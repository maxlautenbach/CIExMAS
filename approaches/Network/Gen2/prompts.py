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

Text: Pozoantiguo is a municipality in Spain, and its patron saint is John the Baptist. It shares borders with Fuentesecas, Abezames, Villardondiego, Toro, Zamora, and Pinilla de Toro. John the Baptist is an iconographic symbol of a cup.
<triples>
1: Pozoantiguo (Municipalities_of_Spain); patron saint; John_the_Baptist (human biblical figure)
2: Pozoantiguo (Municipalities_of_Spain); shares border with; Fuentesecas (Municipalities_of_Spain)
3: Pozoantiguo (Municipalities_of_Spain); instance of; Municipalities_of_Spain (municipality,municipio,administrative territorial entity of Spain,LAU 2,third-order administrative division,local territorial entity)
4: Pozoantiguo (Municipalities_of_Spain); country; Spain (country,nation state,realm,sovereign state,Mediterranean country)
5: Pozoantiguo (Municipalities_of_Spain); shares border with; Abezames (Municipalities_of_Spain)
6: Pozoantiguo (Municipalities_of_Spain); shares border with; Villardondiego (Municipalities_of_Spain)
7: Pozoantiguo (Municipalities_of_Spain); shares border with; Toro,_Zamora (Municipalities_of_Spain)
8: Pozoantiguo (Municipalities_of_Spain); shares border with; Pinilla_de_Toro (Municipalities_of_Spain)
9: John_the_Baptist (human biblical figure); iconographic symbol; Cup (physical container,drinking vessel,vessel)
</triples>

--------------------------------------------------------------------------------

Text: Blood Scent is a groove metal song performed by STEMM.
<triples>
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
</triples>

--------------------------------------------------------------------------------

Text: Miho Nakayama was born in Koganei, Tokyo and is a J-pop singer. She has won the Golden Arrow Award and the Blue Ribbon Award for Best Actress. She is a human. Koganei, Tokyo shares a border with Nishitōkyō.
<triples>
1: Miho_Nakayama (Human); place of birth; Koganei,_Tokyo (city of Japan,big city)
2: Miho_Nakayama (Human); genre; J-pop (A-pop,music genre)
3: Miho_Nakayama (Human); award received; Golden_Arrow_Award (award)
4: Miho_Nakayama (Human); award received; Blue_Ribbon_Award_for_Best_Actress (Blue Ribbon Awards,film award category,award for best leading actress)
5: Miho_Nakayama (Human); instance of; Human (natural person,omnivore,person,individual animal,mammal,organisms known by a particular common name)
6: Koganei,_Tokyo (city of Japan,big city); shares border with; Nishitōkyō (city of Japan,big city)
</triples>

--------------------------------------------------------------------------------

END OF EXAMPLE OUTPUT

Text: {text}

Return the triples in this format:
<triples>
Subject1 (Type A1); Property1; Object1 (Type B1)
Subject2 (Type A2); Property2; Object2 (Type B2)
Subject3 (Type A3); Property3; Object3 (Type B3)
</triples>         

In addition, return which agent to call next:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>       
""")

uri_mapping_and_refinement_prompt = PromptTemplate.from_template("""
You are an expert in creating uri mappings and refining triples. Therefore, you will receive a text and a set of triples extracted from the text.

To process the task you have access to the following tools:
- URI Search Tool
    - ID: uri_search_tool
    - Description: Searches for URIs matching entities or properties.
    - Input format: Term1[Mode]|Term2[Mode]|... (Max 10 terms)
    - Search Modes:
        - [Q]: Entity search with type. Format: 'Entity (type)[Q]'
        - [P]: Property search. Format: 'property[P]'
        - [X]: Property by example. Example must be a sentence in the format: 'Subject (type) property Object (type)[X]'
    - Guidelines:
        - Always use all search terms simultaneously to maximize the chance of finding the correct URIs
        - For properties, use [P] and [X] mode simultaneously to find the most specific property.
    
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

Output Options (can be combined):
- Next Step: <goto>agent_id OR tool_id</goto>
- Tool Input Propagation (if tool is next step): <tool_input>tool_input</tool_input>
- Next Agent Instruction Propagation (if agent is next step): <agent_instruction>agent_instruction</agent_instruction>
- URI Mapping Update: <uri_mapping>uri_mapping</uri_mapping>
    - Format:
        - Subject/Object Extracted Label (Type A1) OR Property Extracted Label - Search mode applied [Q|P|X] - Applicable to [S-Subject|P-Property|O-Object-Triple Numbers] -> i.e. Angela Merkel (Human) - Search mode applied [Q] - Applicable to [S1, S2, O3]
        - URI: uri (from search results)
        - URI-Label: uri_label (corresponding label from the search results)
        - URI-Description: uri_description (corresponding description from the search results)
    - IMPORTANT: The URI Mapping must be provided as a complete set. Include ALL entities and properties from the triples, even if they were previously mapped.
    - For each entity/property in the triples, you must provide a mapping entry with one of these states:
        - If searched and not found: Use "not found" for URI, URI-Label, and URI-Description
        - If not yet searched: Use "not searched" for URI, URI-Label, and URI-Description
        - If found: Use the actual URI, label, and description
    - Do not provide partial updates - the entire mapping must be included each time.

Guidelines:
- Do not assume any search results or created them using your internal knowledge. Instead use the URI Search Tool to find the URIs.
- If you search for an entity/property and don't find a URI, mark it as "not found" and accept this result - do not keep searching for it.
- If you haven't searched for an entity/property yet, mark it as "not searched".
- To use one of the output options, the corresponding tag must be included in the output.
- Keep your output concise and to the point
- IMPORTANT: Track which entities and properties have already been processed in the URI Mapping. Do not search for the same entity or property multiple times.
- If an entity or property is already in the URI Mapping with a valid URI, reuse that mapping in your complete update.
- Only search for entities and properties that are marked as "not searched" in the URI Mapping.
- When updating the URI Mapping, provide a complete mapping that includes ALL entities and properties from the triples.

Additional Instructions/Error Handling: {agent_instruction}

Text: {text}

Triples: {triples}
                                                                 
URI Mapping: {uri_mapping}
                                                                 
Call Trace: {call_trace}

Output for "{last_call}": {last_response}

--------------------------------------------------------------------------------
""")

validation_and_output_prompt = PromptTemplate.from_template("""
You are an expect in validating triples for closed information extraction and deciding whether a better result could be reached or not. You will receive a set of triples and the text they were extracted from. You have to check if the triples are valid and if they are not, you have to decide whether which agent to call next.

To process the task you have access to the following tools:
- Turtle to Labels Tool (Exact 1x required)
    - ID: turtle_to_labels_tool
    - Description: Takes a turtle string and returns the triples with their corresponding labels. The Output will be written into the "Output on Last Call"  field.
    - Example Input:
        @prefix wd: <http://www.wikidata.org/entity/>.
        wd:Q567 wd:P102 wd:Q49762 .
    - IMPORTANT: This tool MUST be called exactly once before ending the process. END can only be called if turtle_to_labels_tool is the last element in the call_trace.

To continue you could call one of the following agents:
- Extractor
    - ID: extractor
    - Description: Extracts triples from a given text.
- URI Mapping & Refinement
    - ID: uri_mapping_and_refinement
    - Description: Maps the triples with URIs and refines the triples. 

Decision Tree for Next Step:
1. If call_trace is empty:
   - Call turtle_to_labels_tool
2. If call_trace ends with 'turtle_to_labels_tool':
   - If labels match URIs OR entities are marked as "not found": Call END
   - If labels don't match AND entities are not marked as "not found": Call extractor or uri_mapping_and_refinement with an instruction that describes the problem.
3. If call_trace ends with any other tool/agent:
   - Call turtle_to_labels_tool
4. Never call turtle_to_labels_tool if it's the last call in the call_trace

Output Options (can be combined):
- Next Step: <goto>agent_id OR tool_id</goto>
- Tool Input Propagation (if tool is next step): <tool_input>tool_input</tool_input>
- Next Agent Instruction Propagation (if agent is next step): <agent_instruction>agent_instruction</agent_instruction>
- Final Output (if END is next step): <ttl>Turtle String</ttl>
    - Only use wd: prefix for URIs.
    - Example Output:
        <ttl>
        @prefix wd: <http://www.wikidata.org/entity/>.
        wd:Q567 wd:P102 wd:Q49762 .
        </ttl>
Guidelines:
- Do only use the URIs from the URI Mapping. If you don't find a URI in the URI Mapping, just don't include the triple in the final output.
- Restrict all turtle strings in your output to the http://wikidata.org/entity namespace and/or wd: prefix.
- Keep your output concise and to the point. Do not repeat the inputs given.
- You MUST call the turtle_to_labels_tool exactly once before ending the process.
- Check the call_trace to ensure turtle_to_labels_tool was the last tool called before using END.
- If the call_trace ends with 'turtle_to_labels_tool', you MUST either end the process or call another agent. Do not loop calling turtle_to_labels_tool.
- NEVER call turtle_to_labels_tool if it's the last call in the call_trace.
- No agent has access on the message history, so give them a meaningful agent instruction.
- If an Entity is mapped as "not found", accept this result and do not try to find it again. These entities should be excluded from the turtle to label input and from the final output.
- IMPORTANT: When entities are marked as "not found" in the URI Mapping, this is a valid final state. Do not try to find these entities again by calling the URI Mapping agent.
- Include exactly one goto tag in your output. If you want to call the turtle_to_labels_tool, use <goto>turtle_to_labels_tool</goto>. If you want to end the process, use <goto>END</goto>. And so on...

START OF INPUT

Additional Instructions/Error Handling: {agent_instruction}

Text: {text}

Triples: {triples}

URI Mapping: {uri_mapping}
                                                            
Call Trace: {call_trace}

Output for "{last_call}": {last_response}
                                                            
END OF INPUT
                                                            
--------------------------------------------------------------------------------
""")