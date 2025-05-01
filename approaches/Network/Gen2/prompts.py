from langchain_core.prompts import PromptTemplate

extractor_prompt = PromptTemplate.from_template("""
You are an expert in extracting triples out of a text. Therefore you are given a text and you have to squeze all triples out of it. You will receive several example to know how the dataset looks like and what type of triples are expected.

In addition, you have to decide which agent to call next, when you are ready with your task. The call should conduct to the overall goal of closed information extraction (from text to triples with URIs):
- Extractor (THIS AGENT)
    - ID: extractor
    - Description: Extracts triples from a given text.
- Validation & Output
    - ID: validation_and_output
    - Description: Validates the final triples using their URIs and outputs them in the required format.
- URI Mapping & Refinement
    - ID: uri_mapping_and_refinement
    - Description: Maps and refines the triples with URIs.

Example Output:

Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
Albert_S._Sholes_House (Types: [house]); architectural style; Bungalow (Types: [house,architectural style])
Albert_S._Sholes_House (Types: [house]); architect; Richard_H._Martin_Jr. (Types: [Human])
</triples>

--------------------------------------------------------------------------------

Text: Krasnoyarsky District, Astrakhan Oblast is located in Astrakhan Oblast and uses Samara Time and Moscow Time.
<triples>
Krasnoyarsky_District,_Astrakhan_Oblast (Types: [municipal district]); located in the administrative territorial entity; Astrakhan_Oblast (Types: [oblast of Russia,administrative region])
Krasnoyarsky_District,_Astrakhan_Oblast (Types: [municipal district]); located in time zone; Samara_Time (Types: [time zone,local mean time])
Krasnoyarsky_District,_Astrakhan_Oblast (Types: [municipal district]); located in time zone; Moscow_Time (Types: [time zone])
</triples>

--------------------------------------------------------------------------------

Text: The Tata Aria, a car made by Tata Motors, is powered by a diesel engine, a type of internal combustion engine invented by Rudolf Diesel.
<triples>
Tata_Aria (Types: [Car,automobile model]); powered by; Diesel_engine (Types: [reciprocating engine,automotive product])
Tata_Aria (Types: [Car,automobile model]); brand; Tata_Motors (Types: [automobile manufacturer,public company,Business])
Tata_Aria (Types: [Car,automobile model]); subclass of; Car (Types: [motor vehicle,road vehicle,multi-track vehicle])
Diesel_engine (Types: [reciprocating engine,automotive product]); discoverer or inventor; Rudolf_Diesel (Types: [Human])
</triples>

--------------------------------------------------------------------------------

END OF EXAMPLE OUTPUT

Text: {text}

Return the triples in this format:
<triples>
Subject1 (Types: [Type A1]); Predicate1; Object1 (Types: [Type B1])
Subject2 (Types: [Type A2]); Predicate2; Object2 (Types: [Type B2])
Subject3 (Types: [Type A3]); Predicate3; Object3 (Types: [Type B3])
</triples>         

In addition, return which agent to call next:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>       
""")

uri_mapping_and_refinement_prompt = PromptTemplate.from_template("""
You are an expert in creating uri mappings and refining triples. Therefore, you will receive a text and a set of triples extracted from the text. Out of this context you have to search URIs for subjects, predicates and objects. 

To process the task you have access to the following tools:
- URI Search Tool
    - ID: uri_search_tool
    - Description: Takes a |-separated list of search terms with search and optional filter modes. Returns the 3 most similar URIs per term.
    - Input format: Term1[SearchMode1-FilterMode1]|Term2[SearchMode2-FilterMode2]...
    - Search Modes:
        - 'LABEL': Search using rdfs:label (recommended for first attempts)
        - 'DESCR': Search using a textual description (especially useful for predicates/entities not found via label)
    - Filter Modes:
        - '-Q': Filter for entities
        - '-P': Filter for predicates
    - Example: <input>Angela Merkel[LABEL-Q]|chancellor of Germany from 2005 to 2021[DESCR-Q]|work location[LABEL-P]</input>

- Network Traversal Tool
    - ID: network_traversal_tool
    - Description: Returns super- and sub-properties of given predicate URIs using SPARQL.
    - Input: One or more predicate URIs separated by |
    - Example: <input>http://www.wikidata.org/entity/P166|http://www.wikidata.org/entity/P361</input>
    
In addition, you can decide which agent to call next, when you are ready with your task:
- Extractor
    - ID: extractor
    - Description: Extracts triples from a given text.
- Validation & Output
    - ID: validation_and_output
    - Description: Validates the final triples using their URIs and outputs them in the required format.
- URI Mapping & Refinement (THIS AGENT)
    - ID: uri_mapping_and_refinement
    - Description: Maps and refines the triples with URIs.
    
Guidelines:
- Use the URI Search Tool to find URIs for subjects and objects. The results will be written into Last Agent/Tool Response.
- Call yourself, if you need to refine the triples further.
- Use the triple state to store intermediate results.

Last Call: {last_call}

Output on Last Call: {last_response}

Text: {text}

Triples {triples}

Your output should be in one of the following formats:

TOOL USAGE:
<goto>ONE OF [uri_search_tool,network_traversal_tool]</goto>
<tool_input>INSERT YOUR INPUT HERE</tool_input>

OR

MAPPING OUTPUT:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>
<triples>
Subject1 (Types: [Type A1], URI: UriS1); Predicate1; Object1 (Types: [Type B1], URI: UriO1)
Subject2 (Types: [Type A2], URI: UriS2); Predicate2; Object2 (Types: [Type B2], URI: UriO2)
Subject3 (Types: [Type A3], URI: UriS3); Predicate3; Object3 (Types: [Type B3], URI: UriO3)
</triples>

Example Mapping Output:
Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171); architectural style (URI: http://www.wikidata.org/entity/P149); Bungalow (Types: [house,architectural style], URI: http://www.wikidata.org/entity/Q850107)
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171); architect (URI: http://www.wikidata.org/entity/P84); Richard_H._Martin_Jr. (Types: [Human], URI: http://www.wikidata.org/entity/Q47035008)
</triples>

""")

validation_and_output_prompt = PromptTemplate.from_template("""
You are an expect in validating triples for closed information extraction and deciding whether a better result could be reached or not. You will receive a set of triples and the text they were extracted from. You have to check if the triples are valid and if they are not, you have to decide whether which agent to call next.

To continue you could call one of the following agents:
- Extractor
    - ID: extractor
    - Description: Extracts triples from a given text.
- URI Mapping & Refinement
    - ID: uri_mapping_and_refinement
    - Description: Maps the triples with URIs and refines the triples. 
                                                            
Your Instruction: {agent_instruction}

Last Call: {last_call}

Output on Last Call: {last_response}

Text: {text}

Triples {triples}

Guidelines:
- Stricly use the URIs that are provided in the triples you get later in this prompt (especially for predicates).

Your output should be in one of the following formats:

FINAL OUTPUT:
<goto>END</goto>
<ttl>INSERT YOUR TURTLE OUTPUT HERE</ttl>

OR

AGENT CALL:
<goto>ONE OF [extractor,uri_mapping_and_refinement]</goto>

EXAMPLE FINAL OUTPUT:
Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171); architectural style (URI: http://www.wikidata.org/entity/P149); Bungalow (Types: [house,architectural style], URI: http://www.wikidata.org/entity/Q850107)
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171); architect (URI: http://www.wikidata.org/entity/P84); Richard_H._Martin_Jr. (Types: [Human], URI: http://www.wikidata.org/entity/Q47035008)
</triples>

Output:
<goto>END</goto>
<ttl>
@prefix wd: <http://www.wikidata.org/entity/>.
wd:Q4711171 wd:P149 wd:Q850107.
wd:Q4711171 wd:P84 wd:Q47035008.
</ttl>

END OF EXAMPLE

YOUR OUTPUT:
""")