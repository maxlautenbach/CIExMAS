from SPARQLWrapper import SPARQLWrapper, JSON

def get_description(uri):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
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
    except IndexError:
        return "No Description Found"

if __name__ == "__main__":
    print(get_description("http://www.wikidata.org/entity/Q61053"))
