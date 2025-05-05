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
You are an expert in creating uri mappings and refining triples (i.e. by replacing predicates for more specific ones). Therefore, you will receive a text and a set of triples extracted from the text. Out of this context you have to search URIs for subjects, predicates and objects. 

To process the task you have access to the following tools:
- URI Search Tool
    - ID: uri_search_tool
    - Description: Takes a |-separated list of search terms with search and optional filter modes. Returns the 3 most similar URIs per term.
    - Input format: Term1([SearchMode1-FilterMode1]|Term2[SearchMode2-FilterMode2]...
    - Search Modes:
        - 'LABEL': Search using rdfs:label (recommended for first attempts)
        - 'DESCR': Search using a textual description (especially useful for predicates/entities not found via label)
    - Filter Modes:
        - '-Q': Filter for entities. Include the expected type of the entity in the search term. I.e. 'Angela Merkel (human)[LABEL-Q]'
        - '-P': Filter for predicates
    - Example: <input>Angela Merkel (human)[LABEL-Q]|chancellor of Germany from 2005 to 2021[DESCR-Q]|work location[LABEL-P]</input>

- Network Traversal Tool
    - ID: network_traversal_tool
    - Description: Returns super- and sub-properties of given predicate URIs using SPARQL.
    - Input: RDF triples in turtle format containing the predicates to explore
    - Example: 
      <input>
      @prefix wd: <http://www.wikidata.org/entity/>.
      wd:Q123 wd:P166 wd:Q456.
      wd:Q789 wd:P361 wd:Q101.
      </input>
    
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
- Refine the search terms, if the results are not satisfying.
- Use the Network Traversal Tool to find super- and sub-properties of predicates.
  - The Network Traversal Tool input should be valid turtle format RDF triples containing the predicates you want to explore.
  - You can include one or more predicates in a single turtle document.
  - The tool will extract all predicates from the triples and return their super- and sub-properties.
- Save intermediate results using the mapping output.

Chain of Thought:
1. Get first URI Mapping
2. Check if the results are satisfying, if not, refine the search terms and call the URI Search Tool again.
3. Call the Network Traversal Tool with predicates formatted as turtle triples.
4. Check if the results can be used. If yes, build them into the triples and call the next agent.

Call Trace: {call_trace}
                                                                 
Last Call: {last_call}

Output on Last Call: {last_response}

Text: {text}

Triples {triples}

Your can include the following:

TOOL USAGE:
<goto>ONE OF [uri_search_tool,network_traversal_tool]</goto>
<tool_input>INSERT YOUR INPUT HERE</tool_input>

AND/OR

MAPPING OUTPUT:
<triples>
Subject1 (Types: [Type A1], URI: UriS1, URI-Label: Uri-LabelS1); Predicate1; Object1 (Types: [Type B1], URI: UriO1, URI-Label: Uri-LabelO1)
Subject2 (Types: [Type A2], URI: UriS2, URI-Label: Uri-LabelS2); Predicate2; Object2 (Types: [Type B2], URI: UriO2, URI-Label: Uri-LabelO2)
Subject3 (Types: [Type A3], URI: UriS3, URI-Label: Uri-LabelS3); Predicate3; Object3 (Types: [Type B3], URI: UriO3, URI-Label: Uri-LabelO3)
</triples>

AND/OR

AGENT CALL:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>

Example Mapping Output:
Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architectural style (URI: http://www.wikidata.org/entity/P149); Bungalow (Types: [house,architectural style], URI: http://www.wikidata.org/entity/Q850107, URI-Label: Bungalow)
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architect (URI: http://www.wikidata.org/entity/P84); Richard_H._Martin_Jr. (Types: [Human], URI: http://www.wikidata.org/entity/Q47035008, URI-Label: Richard_H._Martin_Jr.)
</triples>

Example Network Traversal Tool Input:
<goto>network_traversal_tool</goto>
<tool_input>
@prefix wd: <http://www.wikidata.org/entity/>.
wd:Q4711171 wd:P149 wd:Q850107.
wd:Q4711171 wd:P84 wd:Q47035008.
</tool_input>

Example Processing:
Text: The politician Angela Merkel is member of the CDU.
Triples: 
Angela_Merkel (Types: [politician]); member of; CDU (Types: [political party])
Angela_Merkel (Types: [politician]); occupation; politician (Types: [profession])

1. URI Search with Instruction: Angela Merkel (human)[LABEL-Q]|CDU (political party)[LABEL-Q]|politician (profession)[LABEL-Q]|member of[LABEL-P]|occupation[LABEL-P]
2. Built in pre-liminary triples:
Angela_Merkel (Types: [politician], URI: http://www.wikidata.org/entity/Q567, URI-Label: Angela Merkel); member of (URI:http://www.wikidata.org/entity/P463); Christian Democratic Union (Types: [political party], URI: http://www.wikidata.org/entity/Q49762, URI-Label: Christian Democratic Union)
Angela_Merkel (Types: [politician], URI: http://www.wikidata.org/entity/Q567, URI-Label: Angela Merkel); occupation (URI:http://www.wikidata.org/entity/P106); politician (Types: [politician], URI: http://www.wikidata.org/entity/Q82955, URI-Label: politician)

3. Network Traversal Search with Instruction:
@prefix wd: <http://www.wikidata.org/entity/>.
wd:Q567 wd:P463 wd:Q49762.
wd:Q567 wd:P106 wd:Q82955.

4. Network Traversal Search brought up: 

Possible Predicate Replacements:

## Possible Predicate Replacements for Triple: http://www.wikidata.org/entity/Q567, http://www.wikidata.org/entity/P463, http://www.wikidata.org/entity/Q49762

### Current Predicate Example:
Property: http://www.wikidata.org/entity/P463
Label: member of
Example: Diana Ross; member of; The Supremes
Super-properties (1):

URI: http://www.wikidata.org/entity/P361
Label: part of
Example: Lake Ontario; part of; Great Lakes
Subject matches type restriction: None
Object matches type restriction: None

Sub-properties (3):

URI: http://www.wikidata.org/entity/P4100
Label: parliamentary group
Example: Matteo Renzi; parliamentary group; Italia Viva
Subject matches type restriction: human
Object matches type restriction: political party

URI: http://www.wikidata.org/entity/P102
Label: member of political party
Example: Deng Xiaoping; member of political party; Chinese Communist Party
Subject matches type restriction: human
Object matches type restriction: political organization

URI: http://www.wikidata.org/entity/P102
Label: member of political party
Example: Deng Xiaoping; member of political party; Chinese Communist Party
Subject matches type restriction: human
Object matches type restriction: political party

## Possible Predicate Replacements for Triple: http://www.wikidata.org/entity/Q567, http://www.wikidata.org/entity/P106, http://www.wikidata.org/entity/Q82955

### Current Predicate Example:
Property: http://www.wikidata.org/entity/P106
Label: occupation
Example: Yuri Gagarin; occupation; astronaut
Super-properties (0):

Sub-properties (1):

URI: http://www.wikidata.org/entity/P3001
Label: retirement age
Example: Australia; retirement age; 67
Subject matches type restriction: None
Object matches type restriction: None


5. Update predicate URI for "member of" with more relevant predicates "member of political party" and "parliamentary group" and build the triples:

Angela_Merkel (Types: [politician], URI: http://www.wikidata.org/entity/Q567, URI-Label: Angela Merkel); member of (URI:http://www.wikidata.org/entity/P102, URI-Label: member of political party); Christian Democratic Union (Types: [political party], URI: http://www.wikidata.org/entity/Q49762, URI-Label: Christian Democratic Union)
Angela_Merkel (Types: [politician], URI: http://www.wikidata.org/entity/Q567, URI-Label: Angela Merkel); member of (URI:http://www.wikidata.org/entity/P4100, URI-Label: parliamentary group); Christian Democratic Union (Types: [political party], URI: http://www.wikidata.org/entity/Q49762, URI-Label: Christian Democratic Union)
Angela_Merkel (Types: [politician], URI: http://www.wikidata.org/entity/Q567, URI-Label: Angela Merkel); occupation (URI:http://www.wikidata.org/entity/P106, URI-Label: occupation); politician (Types: [politician], URI: http://www.wikidata.org/entity/Q82955, URI-Label: politician)

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
                                                            
Call Trace: {call_trace}

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