import os
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, POST, JSON
from dotenv import load_dotenv
import git
from tqdm import tqdm
import ntpath
import json
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
repo = git.Repo(search_parent_directories=True)
load_dotenv(repo.working_dir + "/.env", override=True)

FUSEKI_URL = os.getenv("FUSEKI_URL")
FUSEKI_USER = os.getenv("FUSEKI_USER")
FUSEKI_PASSWORD = os.getenv("FUSEKI_PASSWORD")
BATCH_SIZE = 10000  # Number of triples per batch

if not FUSEKI_URL:
    raise ValueError("FUSEKI_URL not found in environment variables")
if not FUSEKI_USER or not FUSEKI_PASSWORD:
    raise ValueError("FUSEKI_USER and FUSEKI_PASSWORD must be set in environment variables")

@dataclass
class SPARQLBinding:
    """
    Represents a single binding in a SPARQL query result
    """
    value: str
    type: str
    datatype: Optional[str] = None
    lang: Optional[str] = None

    @classmethod
    def from_dict(cls, binding_dict: Dict[str, Any]) -> 'SPARQLBinding':
        """
        Create a SPARQLBinding from a dictionary
        """
        return cls(
            value=binding_dict['value'],
            type=binding_dict['type'],
            datatype=binding_dict.get('datatype'),
            lang=binding_dict.get('xml:lang')
        )

@dataclass
class SPARQLResult:
    """
    Represents a single row in a SPARQL query result
    """
    bindings: Dict[str, SPARQLBinding]

    def __getattr__(self, name: str) -> str:
        """
        Allow accessing binding values directly as attributes
        """
        if name in self.bindings:
            return self.bindings[name].value
        raise AttributeError(f"No binding found for {name}")

    @classmethod
    def from_dict(cls, binding_dict: Dict[str, Dict[str, Any]]) -> 'SPARQLResult':
        """
        Create a SPARQLResult from a dictionary of bindings
        """
        return cls(
            bindings={
                var: SPARQLBinding.from_dict(binding)
                for var, binding in binding_dict.items()
            }
        )

@dataclass
class ASKResult:
    """
    Represents the result of an ASK query
    """
    askAnswer: bool

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'ASKResult':
        """
        Create an ASKResult from a dictionary
        """
        return cls(askAnswer=result_dict['boolean'])

class FusekiClient:
    """
    Client for interacting with a specific Fuseki dataset
    """
    def __init__(self, dataset_name: str):
        """
        Initialize the Fuseki client for a specific dataset
        
        Args:
            dataset_name: Name of the dataset to connect to
        """
        self.dataset_name = dataset_name
        self.endpoint = f"{FUSEKI_URL}/{dataset_name}/query"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setCredentials(FUSEKI_USER, FUSEKI_PASSWORD)
        self.sparql.setReturnFormat(JSON)
    
    def query(self, query: str) -> Union[List[SPARQLResult], ASKResult, Dict[str, Any]]:
        """
        Execute a SPARQL query and return the results in a structured format
        
        Args:
            query: SPARQL query string
            
        Returns:
            For SELECT queries: List of SPARQLResult objects
            For ASK queries: ASKResult object
            For CONSTRUCT/DESCRIBE queries: Dictionary containing the constructed graph
        """
        self.sparql.setQuery(query)
        try:
            results = self.sparql.query().convert()
            
            # Handle different query types
            if 'results' in results:  # SELECT query
                bindings = results['results']['bindings']
                # Convert bindings to SPARQLResult objects
                return [SPARQLResult.from_dict(binding) for binding in bindings]
            elif 'boolean' in results:  # ASK query
                return ASKResult.from_dict(results)
            else:  # CONSTRUCT or DESCRIBE query
                return results
                
        except Exception as e:
            logger.error(f"Error executing query on dataset {self.dataset_name}: {str(e)}")
            raise

def get_sparql_wrapper(endpoint):
    """
    Create a SPARQLWrapper instance with authentication
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setCredentials(FUSEKI_USER, FUSEKI_PASSWORD)
    return sparql

def check_datasets():
    """
    Check if both datasets exist and contain triples
    Returns a dictionary with the status of each dataset
    """
    datasets = ["wikidata_predicates", "wikidata_class_hierarchy"]
    status = {}
    
    # First check if datasets exist
    url = f"{FUSEKI_URL}/$/datasets"
    response = requests.get(url, auth=(FUSEKI_USER, FUSEKI_PASSWORD))
    if response.status_code != 200:
        raise Exception(f"Failed to get dataset list: {response.text}")
    
    # Parse the response - Fuseki returns a complex structure with dataset information
    try:
        response_data = response.json()
        if 'datasets' in response_data:
            # Extract dataset names and remove leading slash
            existing_datasets = [ds['ds.name'].lstrip('/') for ds in response_data['datasets']]
        else:
            logger.error(f"Unexpected response format: {response_data}")
            existing_datasets = []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        existing_datasets = []
    
    for dataset in datasets:
        dataset_status = {
            'exists': dataset in existing_datasets,
            'triple_count': 0
        }
        
        if dataset_status['exists']:
            # Check triple count
            client = FusekiClient(dataset)
            try:
                results = client.query("""
                    SELECT (COUNT(*) as ?count)
                    WHERE {
                        ?s ?p ?o .
                    }
                """)
                dataset_status['triple_count'] = int(results[0].count)
            except Exception as e:
                logger.error(f"Error counting triples in {dataset}: {str(e)}")
                dataset_status['triple_count'] = -1
        
        status[dataset] = dataset_status
        logger.info(f"Dataset {dataset}: {'exists' if dataset_status['exists'] else 'does not exist'}, "
                   f"triple count: {dataset_status['triple_count']}")
    
    return status

def create_dataset(dataset_name):
    """
    Create a new dataset in Fuseki using the REST API
    """
    url = f"{FUSEKI_URL}/$/datasets"
    data = {
        'dbName': dataset_name,
        'dbType': 'tdb2'
    }
    response = requests.post(url, data=data, auth=(FUSEKI_USER, FUSEKI_PASSWORD))
    if response.status_code != 200:
        raise Exception(f"Failed to create dataset {dataset_name}: {response.text}")
    logger.info(f"Created dataset {dataset_name}")

def count_lines(file_path):
    """
    Count the number of lines in a file
    """
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def upload_file_in_batches(file_path, dataset_name):
    """
    Upload a file to Fuseki in batches with progress bar
    """
    total_lines = count_lines(file_path)
    file_name = ntpath.basename(file_path)
    file_format = file_name.split('.')[-1]
    
    logger.info(f"Uploading {file_name} to {dataset_name} dataset")
    
    with open(file_path, 'r') as f:
        batch = []
        sparql = get_sparql_wrapper(f"{FUSEKI_URL}/{dataset_name}/update")
        sparql.setMethod(POST)
        
        with tqdm(total=total_lines, desc=f"Uploading {file_name}") as pbar:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    batch.append(line)
                
                if len(batch) >= BATCH_SIZE:
                    # Upload batch
                    query = f"""
                        INSERT DATA {{
                            {' '.join(batch)}
                        }}
                    """
                    sparql.setQuery(query)
                    sparql.query()
                    pbar.update(len(batch))
                    batch = []
            
            # Upload remaining triples
            if batch:
                query = f"""
                    INSERT DATA {{
                        {' '.join(batch)}
                    }}
                """
                sparql.setQuery(query)
                sparql.query()
                pbar.update(len(batch))

def init_db():
    """
    Initialize two datasets in Jena Fuseki:
    1. wikidata_predicates
    2. wikidata_class_hierarchy
    """
    logger.info("Initializing Fuseki datasets...")
    
    # Create datasets
    create_dataset("wikidata_predicates")
    create_dataset("wikidata_class_hierarchy")

    # Load data into datasets
    upload_file_in_batches(
        os.path.join(repo.working_dir, "infrastructure/all_properties.ttl"),
        "wikidata_predicates"
    )
    
    upload_file_in_batches(
        os.path.join(repo.working_dir, "infrastructure/classes.nt"),
        "wikidata_class_hierarchy"
    )

def reinit_db():
    """
    Delete both datasets and reinitialize them
    """
    logger.info("Reinitializing Fuseki datasets...")
    
    # Delete datasets using REST API
    for dataset in ["wikidata_predicates", "wikidata_class_hierarchy"]:
        url = f"{FUSEKI_URL}/$/datasets/{dataset}"
        response = requests.delete(url, auth=(FUSEKI_USER, FUSEKI_PASSWORD))
        if response.status_code != 200:
            raise Exception(f"Failed to delete dataset {dataset}: {response.text}")
        logger.info(f"Deleted dataset {dataset}")

    # Reinitialize datasets
    init_db()
    logger.info("Datasets reinitialized successfully")

if __name__ == "__main__":
    client = FusekiClient("wikidata_predicates")
    print(client.query("""
        ASK {
            ?s ?p ?o .
        }
    """))