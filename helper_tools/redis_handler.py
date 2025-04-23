import redis
from helper_tools.base_setup import repo
from dotenv import load_dotenv
import os

load_dotenv(repo.working_dir + "/.env", override=True)

creds_provider = redis.UsernamePasswordCredentialProvider(os.getenv("REDIS_USER"), os.getenv("REDIS_PASSWORD"))
r = redis.Redis(host=os.getenv("REDIS_HOST"), decode_responses=True, credential_provider=creds_provider,db=1)

def element_info_upload(uri: str, label: str, description: str) -> None:
    """
    Track the upload status of a URI by setting both label and description upload flags to True.
    
    Args:
        uri (str): The URI to track
    """
    element_type = "predicate" if "entity/P" in uri else "entity" if "entity/Q" in uri else "unknown"

    tracking_data = {
        "label": label,
        "description": description,
        "type": element_type
    }
    r.hset(uri, mapping=tracking_data)

def get_element_info(uri: str) -> dict:
    """
    Get the tracking information for a given URI.
    
    Args:
        uri (str): The URI to check
        
    Returns:
        dict: The tracking information if found, empty dict if not found
    """
    if r.exists(uri):
        return r.hgetall(uri)
    return {}

def clear_redis() -> None:
    """
    Clear all data from the Redis database.
    """
    r.flushdb()

if __name__ == "__main__":
    print(get_element_info("http://www.wikidata.org/entity/P618"))
    