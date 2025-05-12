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
Pozoantiguo (Types: [Municipalities_of_Spain]); patron saint; John_the_Baptist (Types: [human biblical figure])
Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Fuentesecas (Types: [Municipalities_of_Spain])
Pozoantiguo (Types: [Municipalities_of_Spain]); instance of; Municipalities_of_Spain (Types: [municipality,municipio,administrative territorial entity of Spain,LAU 2,third-order administrative division,local territorial entity])
Pozoantiguo (Types: [Municipalities_of_Spain]); country; Spain (Types: [country,nation state,realm,sovereign state,Mediterranean country])
Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Abezames (Types: [Municipalities_of_Spain])
Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Villardondiego (Types: [Municipalities_of_Spain])
Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Toro,_Zamora (Types: [Municipalities_of_Spain])
Pozoantiguo (Types: [Municipalities_of_Spain]); shares border with; Pinilla_de_Toro (Types: [Municipalities_of_Spain])
John_the_Baptist (Types: [human biblical figure]); iconographic symbol; Cup (Types: [physical container,drinking vessel,vessel])
</triples>

--------------------------------------------------------------------------------

Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.
<triples>
Blood_Scent (Types: [Album]); genre; Groove_metal (Types: [extreme metal,music genre])
Blood_Scent (Types: [Album]); performer; STEMM (Types: [musical group])
Blood_Scent (Types: [Album]); follows; Songs_for_the_Incurable_Heart (Types: [Album])
</triples>

--------------------------------------------------------------------------------

Text: Miho Nakayama was born in Koganei, Tokyo and is a J-pop singer. She has won the Golden Arrow Award and the Blue Ribbon Award for Best Actress. She is a human. Koganei, Tokyo shares a border with Nishitōkyō.
<triples>
Miho_Nakayama (Types: [Human]); place of birth; Koganei,_Tokyo (Types: [city of Japan,big city])
Miho_Nakayama (Types: [Human]); genre; J-pop (Types: [A-pop,music genre])
Miho_Nakayama (Types: [Human]); award received; Golden_Arrow_Award (Types: [award])
Miho_Nakayama (Types: [Human]); award received; Blue_Ribbon_Award_for_Best_Actress (Types: [Blue Ribbon Awards,film award category,award for best leading actress])
Miho_Nakayama (Types: [Human]); instance of; Human (Types: [natural person,omnivore,person,individual animal,mammal,organisms known by a particular common name])
Koganei,_Tokyo (Types: [city of Japan,big city]); shares border with; Nishitōkyō (Types: [city of Japan,big city])
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
    - Description: Takes a |-separated list of search terms with search and optional filter modes. Returns the 3 most similar URIs per term.
    - Input format: Term1[SearchMode1]|Term2[SearchMode2]...
    - Search Modes:
        - 'Q': Searches for entities. The term must include the expected type of the entity. I.e. 'Angela Merkel (human)[Q]'
        - 'X': Search for property example triples. The term should be in the form of an example sentence like this: 'Subject (Type) Property Object (Type)[X]'. I.e. <tool_input>Angela Merkel (human) is member of the political party CDU (political party)[X]</tool_input>
        - 'P': Searches for property labels - Do not include the whole triple. I.e. <tool_input>architectural style[P]</tool_input>
    Albert_S._Sholes_House (Types: [house]); architectural style; Bungalow (Types: [house,architectural style])
    - Example: <tool_input>Albert_S._Sholes_House (house)[Q]|Bungalow (house,architectural style)[Q]|Albert_S._Sholes_House (house) architectural style Bungalow (house,architectural style)[X]|architectural style[P]</tool_input>
    
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
- Only change or add the URIs, URI-Labels and URI-Descriptions fields. Strictly use them from the search results.
- You need to save intermediate results in the triples output, otherwise they will get lost.
- Keep your output short and concise.

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

Your can include the following:

TOOL USAGE:
<goto>ONE OF [uri_search_tool]</goto>
<tool_input>INSERT YOUR INPUT HERE</tool_input>

AND/OR

TRIPLES OUTPUT:
<triples>
Subject1 (Types: [Type A1], URI: UriS1, URI-Label: Uri-LabelS1, Note: NoteS1); Property1 (URI: UriP1, URI-Label: Uri-LabelP1, URI-Description: URI-DescriptionP1, Note: NoteP1); Object1 (Types: [Type B1], URI: UriO1, URI-Label: Uri-LabelO1, Note: NoteO1)
Subject2 (Types: [Type A2], URI: UriS2, URI-Label: Uri-LabelS2, Note: NoteS2); Property2 (URI: UriP2, URI-Label: Uri-LabelP2, URI-Description: URI-DescriptionP2, Note: NoteP2); Object2 (Types: [Type B2], URI: UriO2, URI-Label: Uri-LabelO2, Note: NoteO2)
Subject3 (Types: [Type A3], URI: UriS3, URI-Label: Uri-LabelS3, Note: NoteS3); Property3 (URI: UriP3, URI-Label: Uri-LabelP3, URI-Description: URI-DescriptionP3, Note: NoteP3); Object3 (Types: [Type B3], URI: UriO3, URI-Label: Uri-LabelO3, Note: NoteO3)
</triples>

AND/OR

AGENT CALL:
<goto>ONE OF [extractor,validation_and_output,uri_mapping_and_refinement]</goto>

Example Mapping Output:
Text: The Albert S. Sholes House is a bungalow designed by architect Richard H. Martin Jr.
<triples>
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architectural style (URI: http://www.wikidata.org/entity/P149, URI-Label: 'architectural style', URI-Description: 'architectural style of a structure'); Bungalow (Types: [house,architectural style], URI: http://www.wikidata.org/entity/Q850107, URI-Label: Bungalow)
Albert_S._Sholes_House (Types: [house], URI: http://www.wikidata.org/entity/Q4711171, URI-Label: Albert_S._Sholes_House); architect (URI: http://www.wikidata.org/entity/P84, URI-Label: 'architect', URI-Description: 'person or architectural firm responsible for designing this building'); Richard_H._Martin_Jr. (Types: [Human], URI: http://www.wikidata.org/entity/Q47035008, URI-Label: Richard_H._Martin_Jr.)
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
- Strictly use the URIs that are provided in the triples you get later in this prompt (especially for properties).
- Each triple is in the following format: Subject (Types: [Type A], URI: UriS, URI-Label: Uri-LabelS); Property (URI: UriP, URI-Label: Uri-LabelP, URI-Description: URI-DescriptionP); Object (Types: [Type B], URI: UriO, URI-Label: Uri-LabelO)
- Check if there are any notes left from the previous agents. If yes, think about how to solve them.
- Every field starting with "URI" is the corresponding URI, that was found by the URI Search Tool.
- Check if the description of the property matches the context of the triple. If not, you can call one of the mentioned agents to refine the triples.
- Only use the @prefix wd: <http://www.wikidata.org/entity/>. Do not use any other prefixes.
- Use the Turtle to Labels Tool exactly once before outputting the final triples. IF THE LAST CALL WAS TURTLE TO LABELS TOOL, DO NOT CALL IT AGAIN.
- Do not include two-gotos in your output. Only one goto is allowed.

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