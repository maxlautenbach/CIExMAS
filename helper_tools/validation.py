import re

from rdflib import Graph, URIRef
from helper_tools.wikidata_loader import get_types, send_query
from helper_tools.base_setup import wikidata_class_hierarchy


def validate_turtle_response(response):
    turtle_string_match = re.search(r'<ttl>(.*?)</ttl>', response, re.DOTALL)
    if turtle_string_match:
        turtle_string = turtle_string_match.group(1)
    else:
        turtle_string = response
    try:
        # Load the Turtle file into an RDF graph
        result_graph = Graph()
        result_graph.parse(data=turtle_string, format="turtle")
        return True, ""
    except Exception as e:
        return False, str(e)


def check_type_constraint(entity_uri, constraint_uri):
    """
    Check if an entity's type matches a constraint by checking the class hierarchy.
    
    Args:
        entity_uri: URI of the entity to check
        constraint_uri: URI of the type constraint
        
    Returns:
        bool: True if the entity's type matches the constraint, False otherwise
    """
    # Get all types of the entity
    entity_types = get_types(entity_uri, output_uri=True)
    
    # For each type of the entity, check if it matches the constraint
    for type_uri in entity_types:
        # Check if the type is a subclass of the constraint
        query = f"""
        ASK {{
            <{type_uri}> <http://www.w3.org/2000/01/rdf-schema#subClassOf>* <{constraint_uri}> .
        }}
        """
        ask_answer = wikidata_class_hierarchy.query(query).askAnswer
        if ask_answer:
            return True
            
    return False


def validate_triple(subject_uri, property_uri, object_uri):
    """
    Validate a triple by checking if its subject and object types match the property's type constraints.
    Returns True for a constraint if it's empty or if at least one constraint is matched.
    
    Args:
        subject_uri: URI of the subject
        property_uri: URI of the property
        object_uri: URI of the object
        
    Returns:
        tuple containing:
        - bool: Whether subject type constraints are matched
        - bool: Whether object type constraints are matched
    """
    # Get subject type constraints
    subject_constraints_query = f"""
    SELECT DISTINCT ?subjectTypeConstraint ?subjectTypeLabel
    WHERE {{
      <{property_uri}> p:P2302 ?csST .
      ?csST ps:P2302 wd:Q21503250 .  # subject type constraint
      ?csST pq:P2308 ?subjectTypeConstraint .
      OPTIONAL {{ ?subjectTypeConstraint rdfs:label ?subjectTypeLabel . FILTER(LANG(?subjectTypeLabel) = "en") }}
    }}
    """
    
    # Get object type constraints
    object_constraints_query = f"""
    SELECT DISTINCT ?objectTypeConstraint ?objectTypeLabel
    WHERE {{
      <{property_uri}> p:P2302 ?csVT .
      ?csVT ps:P2302 wd:Q21510865 .  # value type constraint
      ?csVT pq:P2308 ?objectTypeConstraint .
      OPTIONAL {{ ?objectTypeConstraint rdfs:label ?objectTypeLabel . FILTER(LANG(?objectTypeLabel) = "en") }}
    }}
    """
    
    # Execute queries
    subject_results = send_query(subject_constraints_query)
    object_results = send_query(object_constraints_query)
    
    # Check subject type constraints
    subject_restriction_matched = True  # Default to True if no constraints
    try:
        if subject_results["results"]["bindings"]:
            subject_restriction_matched = False  # Set to False if there are constraints
            for binding in subject_results["results"]["bindings"]:
                if "subjectTypeConstraint" in binding:
                    constraint_uri = binding["subjectTypeConstraint"]["value"]
                    if check_type_constraint(subject_uri, constraint_uri):
                        subject_restriction_matched = True
                        break  # Stop at first match
    except Exception:
        pass
    
    # Check object type constraints
    object_restriction_matched = True  # Default to True if no constraints
    try:
        if object_results["results"]["bindings"]:
            object_restriction_matched = False  # Set to False if there are constraints
            for binding in object_results["results"]["bindings"]:
                if "objectTypeConstraint" in binding:
                    constraint_uri = binding["objectTypeConstraint"]["value"]
                    if check_type_constraint(object_uri, constraint_uri):
                        object_restriction_matched = True
                        break  # Stop at first match
    except Exception:
        pass
            
    return subject_restriction_matched, object_restriction_matched


if __name__ == "__main__":
    results = validate_triple("http://www.wikidata.org/entity/Q567", "http://www.wikidata.org/entity/P4100",
                      "http://www.wikidata.org/entity/Q1023134")
    print(results)