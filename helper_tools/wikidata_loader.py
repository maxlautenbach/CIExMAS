from SPARQLWrapper import SPARQLWrapper, JSON

def get_description(uri):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
            SELECT ?o WHERE {'{wd:' + uri} schema:description ?o. FILTER(langmatches(lang(?o), "en")){'}'}
        """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    res = sparql.query().convert()["results"]["bindings"][0]["o"]["value"]
    return res

if __name__ == "__main__":
    print(get_description("Q567"))