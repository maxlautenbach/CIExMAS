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
1: Pozoantiguo (Types: [Municipalities_of_Spain]); patron saint; John_the_Baptist (Types: [human biblical figure])
2: Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Fuentesecas (Types: [Municipalities_of_Spain])
3: Pozoantiguo (Types: [Municipalities_of_Spain]); instance of; Municipalities_of_Spain (Types: [municipality,municipio,administrative territorial entity of Spain,LAU 2,third-order administrative division,local territorial entity])
4: Pozoantiguo (Types: [Municipalities_of_Spain]); country; Spain (Types: [country,nation state,realm,sovereign state,Mediterranean country])
5: Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Abezames (Types: [Municipalities_of_Spain])
6: Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Villardondiego (Types: [Municipalities_of_Spain])
7: Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Toro,_Zamora (Types: [Municipalities_of_Spain])
8: Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Pinilla_de_Toro (Types: [Municipalities_of_Spain])
9: John_the_Baptist (Types: [human biblical figure]); iconographic symbol; Cup (Types: [physical container,drinking vessel,vessel])
</triples>

--------------------------------------------------------------------------------

Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.
<triples>
1: Blood_Scent (Types: [Album]); genre; Groove_metal (Types: [extreme metal,music genre])
2: Blood_Scent (Types: [Album]); performer; STEMM (Types: [musical group])
3: Blood_Scent (Types: [Album]); follows; Songs_for_the_Incurable_Heart (Types: [Album])
</triples>

--------------------------------------------------------------------------------

Text: Miho Nakayama was born in Koganei, Tokyo and is a J-pop singer. She has won the Golden Arrow Award and the Blue Ribbon Award for Best Actress. She is a human. Koganei, Tokyo shares a border with Nishitōkyō.
<triples>
1: Miho_Nakayama (Types: [Human]); place of birth; Koganei,_Tokyo (Types: [city of Japan,big city])
2: Miho_Nakayama (Types: [Human]); genre; J-pop (Types: [A-pop,music genre])
3: Miho_Nakayama (Types: [Human]); award received; Golden_Arrow_Award (Types: [award])
4: Miho_Nakayama (Types: [Human]); award received; Blue_Ribbon_Award_for_Best_Actress (Types: [Blue Ribbon Awards,film award category,award for best leading actress])
5: Miho_Nakayama (Types: [Human]); instance of; Human (Types: [natural person,omnivore,person,individual animal,mammal,organisms known by a particular common name])
6: Koganei,_Tokyo (Types: [city of Japan,big city]); shares border with; Nishitōkyō (Types: [city of Japan,big city])
</triples>

--------------------------------------------------------------------------------

END OF EXAMPLE OUTPUT

Text: {text}

Return the triples in this format:
<triples>
Subject1 (Types: [Type A1]); Property1; Object1 (Types: [Type B1])
Subject2 (Types: [Type A2]); Property2; Object2 (Types: [Type B2])
Subject3 (Types: [Type A3]); Property3; Object3 (Types: [Type B3])
</triples>         

In addition, return which agent to call next:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>       
""")

uri_mapping_and_refinement_prompt = PromptTemplate.from_template("""
You are an expert in creating uri mappings and refining triples. Therefore, you will receive a text and a set of triples extracted from the text. Out of this context you have to search URIs for subjects, properties and objects. 

To process the task you have access to the following tools:
- URI Search Tool
    - ID: uri_search_tool
    - Description: Searches for URIs matching entities or properties.
    - Input format: Term1[Mode]|Term2[Mode]|... (Max 10 terms)
    - Search Modes:
        - [Q]: Entity search with type. Format: 'Entity (type)[Q]'
        - [P]: Property search. Format: 'property[P]'
        - [X]: Property by example. Example must be a sentence in the format: 'Subject (type) property Object (type)[X]'
    - Example: <tool_input>Albert_S._Sholes_House (house)[Q]|architectural style[P]|Albert_S._Sholes_House (house) architectural style Bungalow (architectural style)[X]</tool_input>
    
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
- Use the URI Search Tool to find URIs for subjects, objects and properties. The results of the tool will be written into Last Agent/Tool Response.
- Use 'P' and 'X' at the same time to find the best matching property URI.
- Only change or add the URIs, URI-Labels and URI-Descriptions fields strictly based on the Output on Last Call.
- You need to save intermediate results in the triples output, otherwise they will get lost.
- Keep your output short and concise.
- Do not use the URI Search Tool for the same term twice.
- Only use the triples output, when you make an update to the triples.

Chain of Thought:
1. Get first URI Mapping by search for all subject, property and object URIs using the URI Search Tool. Use all modes 'Q' for subjects/objects, 'X' for property example sentences and 'P' for property labels.
2. The search results must be saved into the triples output.
3. Check if the results are satisfying and the descriptions and/or example matches the background of the text and triple. If not, add a note to the non-found entity/property and search using alternative search terms or an alternative search mode 'P' for non-found properties. Stay in Step 3. If yes, go to Step 4.
4. If a property URI was found using [P] mode, make a second search with the full triple using the label from [P] mode using the [X] mode.
5. Check if the results are final. If yes, build them into the triples and call the next agent.

Your Instruction: {agent_instruction}
                                                                 
Call Trace: {call_trace}
                                                                 
Last Call: {last_call}

Output on Last Call: {last_response}

Text: {text}

Triples {triples}

You can include the following (please include exactly one goto tag):

TOOL USAGE:
<goto>ONE OF [uri_search_tool]</goto>
<tool_input>INSERT YOUR INPUT HERE</tool_input>

AND/OR
                                                                 
INSTRUCTION FOR NEXT CALL:
<agent_instruction>INSERT YOUR INSTRUCTION HERE</agent_instruction>

TRIPLES OUTPUT:
<triples>
1. Subject1 (Types: [Type A1], URI: UriS1, URI-Label: Uri-LabelS1, Note: NoteS1); Property1 (URI: UriP1, URI-Label: Uri-LabelP1, URI-Description: URI-DescriptionP1, Note: NoteP1); Object1 (Types: [Type B1], URI: UriO1, URI-Label: Uri-LabelO1, Note: NoteO1)
2. Subject2 (Types: [Type A2], URI: UriS2, URI-Label: Uri-LabelS2, Note: NoteS2); Property2 (URI: UriP2, URI-Label: Uri-LabelP2, URI-Description: URI-DescriptionP2, Note: NoteP2); Object2 (Types: [Type B2], URI: UriO2, URI-Label: Uri-LabelO2, Note: NoteO2)
3. Subject3 (Types: [Type A3], URI: UriS3, URI-Label: Uri-LabelS3, Note: NoteS3); Property3 (URI: UriP3, URI-Label: Uri-LabelP3, URI-Description: URI-DescriptionP3, Note: NoteP3); Object3 (Types: [Type B3], URI: UriO3, URI-Label: Uri-LabelO3, Note: NoteO3)
</triples>

AND/OR

AGENT CALL:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>

Example Mapping Output:
Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
1. Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architectural style (URI: http://www.wikidata.org/entity/P149, URI-Label: 'architectural style', URI-Description: 'architectural style of a structure'); Bungalow (Types: [house,architectural style], URI: http://www.wikidata.org/entity/Q850107, URI-Label: Bungalow)
2. Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architect (URI: http://www.wikidata.org/entity/P84, URI-Label: 'architect', URI-Description: 'person or architectural firm responsible for designing this building'); Richard_H._Martin_Jr. (Types: [Human], URI: http://www.wikidata.org/entity/Q47035008, URI-Label: Richard_H._Martin_Jr.)
</triples>

""")

validation_and_output_prompt = PromptTemplate.from_template("""
You are an expect in validating triples for closed information extraction and deciding whether a better result could be reached or not. You will receive a set of triples and the text they were extracted from. You have to check if the triples are valid and if they are not, you have to decide whether which agent to call next.

To process the task you have access to the following tools:
- Turtle to Labels Tool
    - ID: turtle_to_labels_tool
    - Description: Takes a turtle string and returns the corresponding labels for each URI in the triples. The Output will be written into the Output on Last Call field.
    - Input format: <tool_input>INSERT YOUR TURTLE STRING HERE</tool_input>

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
- ALWAYS validate before output:
  1. Call Turtle to Labels Tool to verify URI labels
  2. Check if property descriptions match triple context
  3. Verify all notes from previous agents are resolved
  4. Only proceed to final output if all validations pass
- Use only URIs provided in the triples, following format: Subject (Types: [Type A], URI: UriS, URI-Label: Uri-LabelS); Property (URI: UriP, URI-Label: Uri-LabelP, URI-Description: URI-DescriptionP); Object (Types: [Type B], URI: UriO, URI-Label: Uri-LabelO)
- If validation fails, call appropriate agent (uri_mapping_and_refinement or extractor)
- Use only @prefix wd: <http://www.wikidata.org/entity/> for the turtle output
- Include exactly one goto tag in your output
- If there is a Note: Not Found, let the triple out and do not re-iterate.
- If the labels has already been verified, do not call the Turtle to Labels Tool again.

Your output should be in one of the following formats:

TOOL USAGE:
<goto>turtle_to_labels_tool</goto>
<tool_input>INSERT YOUR TURTLE STRING HERE</tool_input>

OR

FINAL OUTPUT:
<goto>END</goto>
<ttl>INSERT YOUR TURTLE OUTPUT HERE</ttl>

OR

AGENT CALL:
<goto>ONE OF [extractor,uri_mapping_and_refinement]</goto>
<agent_instruction>INSERT YOUR AGENT INSTRUCTION HERE</agent_instruction>

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