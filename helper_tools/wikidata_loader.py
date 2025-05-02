from distutils.core import run_setup

from SPARQLWrapper import SPARQLWrapper, JSON
from helper_tools.base_setup import sparql
from helper_tools.redis_handler import get_element_info, element_info_upload

def send_query(query):
    """Helper function to send a SPARQL query and handle retries
    
    Args:
        query (str): The SPARQL query to execute
        
    Returns:
        dict: The query results or empty string if failed
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = ""
    retry = 0
    max_retries = 10
    while results == "" and retry < max_retries:
        try:
            results = sparql.query().convert()
        except Exception as e:
            retry += 1
    return results

def get_description(uri):
    # Check if the description exists in Redis
    redis_info = get_element_info(uri)
    if redis_info and "description" in redis_info:
        return redis_info["description"]
    
    # If not in Redis, perform SPARQL query
    query = f"""
        PREFIX schema: <http://schema.org/>
        SELECT ?o WHERE {{
            <{uri}> schema:description ?o .
            FILTER(langmatches(lang(?o), "en"))
        }}
    """
    results = send_query(query)
    
    try:
        description = results["results"]["bindings"][0]["o"]["value"]
    except Exception:
        description = "No Description Found"
    
    # Check if we already have a label in Redis
    if redis_info and "label" in redis_info:
        label = redis_info["label"]
    else:
        # Get label from SPARQL
        label = fetch_label_from_sparql(uri)
    
    # Store both in Redis
    element_info_upload(uri, label, description)
    
    return description

def get_label(uri):
    # Check if the label exists in Redis
    redis_info = get_element_info(uri)
    if redis_info and "label" in redis_info:
        return redis_info["label"]
    
    # If not in Redis, perform SPARQL query to get label
    label = fetch_label_from_sparql(uri)
    
    # Check if we already have a description in Redis
    if redis_info and "description" in redis_info:
        description = redis_info["description"]
    else:
        # Get description from SPARQL
        description = fetch_description_from_sparql(uri)
    
    # Store both in Redis
    element_info_upload(uri, label, description)
    
    return label

def fetch_label_from_sparql(uri):
    """Helper function to fetch a label directly from SPARQL without Redis checks"""
    query = f"""
            SELECT ?o WHERE {{
                <{uri}> rdfs:label ?o .
                FILTER(langmatches(lang(?o), "en"))
            }}
        """
    results = send_query(query)
    try:
        return results["results"]["bindings"][0]["o"]["value"]
    except Exception:
        return "No Label Found"

def fetch_description_from_sparql(uri):
    """Helper function to fetch a description directly from SPARQL without Redis checks"""
    query = f"""
        PREFIX schema: <http://schema.org/>
        SELECT ?o WHERE {{
            <{uri}> schema:description ?o .
            FILTER(langmatches(lang(?o), "en"))
        }}
        ORDER BY (lang(?l) != "en")
        LIMIT 1
    """
    results = send_query(query)
    try:
        return results["results"]["bindings"][0]["o"]["value"]
    except Exception:
        return "No Description Found"

def get_types(uri):
    query = f"""
        SELECT ?o WHERE {{
            <{uri}> wdt:P31 ?o .
        }}
    """
    results = send_query(query)
    try:
        type_list = get_superclasses(uri)
        for binding in results["results"]["bindings"]:
            type_list.append(get_label(binding["o"]["value"]))
        return type_list
    except Exception:
        return ["No Types Found"]

def get_superclasses(uri):
    query = f"""
            SELECT ?o WHERE {{
                <{uri}> wdt:P279 ?o .
            }}
        """
    results = send_query(query)
    try:
        super_class_list = []
        for binding in results["results"]["bindings"]:
            super_class_list.append(get_label(binding["o"]["value"]))
        return super_class_list
    except Exception:
        return ["No Super-Class Found"]


if __name__ == "__main__":
    print(get_types("http://www.wikidata.org/entity/Q174174"))
