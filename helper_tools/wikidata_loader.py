from SPARQLWrapper import SPARQLWrapper, JSON
from helper_tools.base_setup import sparql
from helper_tools.redis_handler import get_element_info, element_info_upload

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
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = ""
    retry = 0
    max_retries = 10
    while results == "" and retry < max_retries:
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(e)
            retry += 1
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
    try:
        return results["results"]["bindings"][0]["o"]["value"]
    except Exception:
        return "No Description Found"

if __name__ == "__main__":
    print(get_label("http://www.wikidata.org/entity/Q567"))
