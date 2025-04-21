import re

from rdflib import Graph


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
