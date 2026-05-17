from langchain_core.prompts import PromptTemplate

entity_extractor_prompt = PromptTemplate.from_template("""Extract all entities from the text that could be subjects or objects in a knowledge graph.

Text: {text}
Instruction: {instruction}

Rules:
- Extract named entities, concepts, and types explicitly mentioned
- Use natural language with spaces (e.g., "Ice hockey", not "Ice_hockey")
- Each entity must be atomic — one concept per entry
- Only extract entities specific enough to have a dedicated encyclopedia entry

Example:
Text: "Austrian skater Willy Böckl won the men's singles event at the 1924 Winter Olympics, held in Chamonix, France."
Entities: ["Figure skating at the 1924 Winter Olympics – Men's singles", "Willy Böckl", "Figure skating at the 1924 Winter Olympics", "1924 Winter Olympics", "Chamonix", "France"]
""")

triple_extractor_prompt = PromptTemplate.from_template("""Extract predicates and form triples from the text using the available entities.

Text: {text}
Available Entities: {entities}
Instruction: {instruction}

Rules:
- Use only entities from the available list as subjects/objects
- Use specific Wikidata-style property names as predicates (e.g., "headquarters location", "country of origin", "parent organization", "language of work or name", "replaced by", "sport", "publisher", "owned by", "instance of", "indexed in bibliographic review", "file format", "developer", "screenwriter", "original language of film or TV show")
- The primary topic of the sentence is the subject. For creative works (films, journals, software): the WORK is always the subject (e.g., "Film; screenwriter; Person" NOT "Person; screenwriter of; Film")
- Format each triple as: subject; predicate; object
""")

turtle_generator_prompt = PromptTemplate.from_template("""You are given triples extracted from a text. For each triple component (subject, predicate, object), URI candidates from a knowledge base are provided. Select the best URI for each component and output valid Turtle RDF.

Source Text: {text}

Triples with URI candidates:
{triples_with_candidates}

{instruction}

Rules:
- Convert ALL triples to Turtle — do not skip any unless a component has zero candidates
- For each triple, pick the most appropriate URI candidate for subject, predicate, and object based on the source text context
- Use @prefix declarations for common prefixes
- Every statement must end with a period
- Output ONLY valid Turtle syntax
""")
