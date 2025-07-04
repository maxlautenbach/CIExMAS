from langchain_core.prompts import PromptTemplate

supervisor_prompt = PromptTemplate.from_template("""
You are the Supervisor of a conversation among multiple agents.
The conversation is about extracting information (Closed Information Extraction) from a user-provided text. The final output should only contain wikidata URIs instead of the labels of entities and relations. You can provide additional information to the agents using <instruction> tags.

Agent Descriptions:
- entity_extraction_agent: Extracts entities from the text. Instructions can change the extraction behavior and focus of the agent. Do not instruct the agent with URIs. The agent has only access to your instruction and the text.
- relation_extraction_agent: Extracts relations from the text. Instructions can change the prompt of the called agent and can be used to input already extracted entity labels (i.e. <instruction>Please use the already extracted entities: [Olaf Scholz, Germany, Berlin]</instruction>). Do not instruct the agent with URIs. The agent has only access to your instruction and the text.
- uri_detection_agent: Please only use this, after entities and relation were extracted at least once. Returns possible wikidata URIs for entities and predicates based on similarity search. The instruction must be a list of search terms with their corresponding search modes, separated by commas. The search modes are:
  * [Q] mode: Entity search with type. Format: 'Entity (type)[Q]'
  * [P] mode: Property search. Format: 'property[P]'
  * [X] mode: Property search by example. Format: 'Subject (type) property Object (type)[X]'
  Example instruction: <instruction>Olaf Scholz (politician)[Q], Germany (country)[Q], is chancellor of[P], Olaf Scholz (politician) is chancellor of Germany (country)[X]</instruction>
  It is recommended to use the agent at least once to search for the URIs for all possible entities and predicates. Do not instruct the agent with URIs like Q100 or P300. The agent has only access to your instruction and the text. If this agent return relations or entities that were not extracted before i.e. industry - URI of sport please try to use them as instruction for the relation extraction agent.

You have two options:
1. Call an agent using <goto>agent_name</goto>. Replace agent_name with either entity_extraction_agent or relation_extraction_agent. I.e. <goto>entity_extraction_agent</goto>.
2. Finish the conversation using <goto>FINISH</goto>. Please output the final relations with their URI in turtle format enclosed in <ttl></ttl> alongside with <goto>FINISH</goto>. Please ensure, that the URI are enclosed in angle brackets. Ignore your implicit knowledge about public knowledge graphs (i.e. Namespaces for properties or URIs mapped to labels) and make sure, that you only use URIs, that were previously extracted by the uri_detection_agent. For example do not use the wikidata prefix wdt for properties, when there is no URI extracted with http://www.wikidata.org/prop/direct/, instead use the URIs extracted.

Example Final Output: 
<goto>FINISH</goto>
<ttl>
<http://www.wikidata.org/entity/Q950380> <http://www.wikidata.org/entity/P361> <http://www.wikidata.org/entity/Q2576666>. 
<http://www.wikidata.org/entity/Q61053> <http://www.wikidata.org/entity/P361> <http://www.wikidata.org/entity/Q315863>.
</ttl>


Note:
- You will receive the full message history and the text, where the relations should be extracted from.
- Do not provide any information yourself, instead use the agents for this.
- The first <goto> tag in your response will be executed.
- Therefore, do include exact one agent call in your response.
- If you output nothing, this will result in a NoneType Error.
- Please do not hallucinate any URI, instead get all your URIs from the uri_detection_agent.

Text: {text}
Message History (latest to oldest): {history}
""")

entity_extractor_prompt = PromptTemplate.from_template("""
    You are an agent tasked with extracting entities from a given text for linking to a knowledge graph. Your job is to capture every entity—both explicit and implicit—and return them as an array. This includes composite entities with modifiers (e.g., "professional football club"). Please output the entities as an array of strings. Do not include any further information or comment in the output.
    
    Example Output: [Olaf Scholz, chancellor, Germany]
    
    Guidelines:
    - An entity is a unique real-world object or concept represented as a node with its properties and relationships.
    - Extract every entity mentioned in the text, including those that are not immediately obvious.
    - For composite entities, include the full descriptive phrase and break it into its core components when appropriate. For example, "chancellor of Germany" should yield [chancellor, Germany] and "professional football club" should capture the descriptive phrase as needed.
    - For composite entities that include a date at the beginning or end, extract the date separately, the entity without the date, and the full composite (e.g., "2022 Winter Olympics" should result in [2022, 2022 Winter Olympics, Winter Olympics]).
    - Also, ensure that dates are extracted as entities.
    
    Instruction: {instruction}
    
    Text: {text}
    """)

relation_extractor_prompt = PromptTemplate.from_template(
        """
        You are a relation extraction agent. Your task is to analyze the provided text and extract all semantic relations present. Each relation must be output in the exact format:
        
        <relation>subject;predicate;object</relation>
        
        (For example: <relation>Olaf Scholz;is chancellor of;Germany</relation>).
        
        Guidelines:
        - **Extraction Scope:** Extract only the relations explicitly mentioned in the text. Additionally, if the text implies a relation or if a relation can be inferred using the provided entity list, include that relation.
        - **Utilize Provided Entities:** Use the provided list of extracted entities to ensure that all relevant relations are captured. For example, if "technology" is in the list and the text indicates that the subject is a technology company, you must output: <relation>Apple;industry;technology</relation>.
        - **Attribute Relations:** If an entity is described by a characteristic or category (e.g., renowned film director, prestigious university), automatically extract the corresponding attribute relation. For example, if the text states "Steven Spielberg is a renowned film director", extract: <relation>Steven Spielberg;profession;film director</relation>.
        - **Formatting:** Each relation must strictly follow the format <relation>subject;predicate;object</relation> with no additional text or commentary.
        - **Accuracy:** Only include relations that are clearly supported by the text or can be confidently inferred using the provided entity list.
        
        Instruction: {instruction}
        
        Text: {text}
        """
    )

uri_detection_prompt = PromptTemplate.from_template(
        """
        You are a formatting agent. Your task is to check and format the output of the URI search tool. The tool will give a response with search results for different modes:
        
        Search Modes and their Results:
        1. Default Search Mode (LABEL):
           - Used when no specific mode is provided
           - Searches in entity and property labels
           - Example: "Most Similar Search Results for 'Olaf Scholz' - Default Search Mode (LABEL):"
        
        2. [Q] Mode - Entity Search with Type:
           - Used for searching entities with their type information
           - Example: "Similar Search Results for 'Olaf Scholz (politician)' - Search Mode [Q]:"
           - Results include label, URI, description, and example if available
        
        3. [P] Mode - Property Search:
           - Used for searching properties/predicates
           - Example: "Similar Search Results for 'is chancellor of' - Search Mode [P]:"
           - Results include label, URI, description, and example if available
        
        4. [X] Mode - Property Search by Example:
           - Used for searching properties in context
           - Example: "Similar Search Results for 'Olaf Scholz (politician) is chancellor of Germany (country)' - Search Mode [X]:"
           - Results include label, URI, description, and the example used for matching
        
        Your task is to:
        1. Check the response and output an overall mapping of search terms to URIs
        2. If something doesn't match, respond with the non-mapping search term and advise that it might not be present in the knowledge graph
        3. If no search terms are referred to in the search response, ask the supervisor to correct its behavior
        4. If the search terms look like this: Q517, P300, inform the supervisor that it sent wikidata URIs instead of search terms
        
        URI search request response:
        
        {search_response}
        """
    )
