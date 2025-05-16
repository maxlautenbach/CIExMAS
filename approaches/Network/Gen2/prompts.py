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

Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.
<triples>
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
3: Blood_Scent (Album); follows; Songs_for_the_Incurable_Heart (Album)
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
    
START OF EXAMPLES

START OF INPUT 1

Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.

Triples:
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
3: Blood_Scent (Album); follows; Songs_for_the_Incurable_Heart (Album)
                                                        
END OF INPUT 1

START OF OUTPUT 1
<goto>uri_search_tool</goto>
<tool_input>Blood_Scent (Album)[Q]|genre[P]|Blood_Scent (Album) genre Groove_metal (extreme metal,music genre)[X]|Groove_metal (extreme metal,music genre)[Q]|performer[P]|Blood_Scent (Album) performer STEMM (musical group)[X]|STEMM (musical group)[Q]|follows[P]|Blood_Scent (Album) follows Songs_for_the_Incurable_Heart (Album)[X]|Songs_for_the_Incurable_Heart (Album)[Q]</tool_input>

END OF OUTPUT 1

START OF INPUT 2
Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.

Triples:
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
3: Blood_Scent (Album); follows; Songs_for_the_Incurable_Heart (Album)

Output of "URI Search Tool - INPUT: Blood_Scent (Album)[Q]|genre[P]|Blood_Scent (Album) genre Groove_metal (extreme metal,music genre)[X]|Groove_metal (extreme metal,music genre)[Q]|performer[P]|Blood_Scent (Album) performer STEMM (musical group)[X]|STEMM (musical group)[Q]|follows[P]|Blood_Scent (Album) follows Songs_for_the_Incurable_Heart (Album)[X]|Songs_for_the_Incurable_Heart (Album)[Q]":
Most Similar Search Results for "Blood_Scent (Album)" - Search Mode [Q]:
  1. Label: Blood_Scent
     URI: http://www.wikidata.org/entity/Q4927704
     Description: album by STEMM
  2. Label: Album
     URI: http://www.wikidata.org/entity/Q482994
     Description: collection of recorded music, words, sounds
  3. Label: Single_(music)
     URI: http://www.wikidata.org/entity/Q134556
     Description: group of single releases by an artist usually released at the same time with the same title and tracks but in different formats for consumption (digital, CD, LP)

Most Similar Search Results for "genre" - Search Mode [P]:
  1. Label: genre
     URI: http://www.wikidata.org/entity/P136
     Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic
     Example: Grand Theft Auto V genre first-person shooter
  2. Label: voice type
     URI: http://www.wikidata.org/entity/P412
     Description: person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)
     Example: Sarah Brightman voice type soprano
  3. Label: architectural style
     URI: http://www.wikidata.org/entity/P149
     Description: architectural style of a structure
     Example: Notre-Dame de Paris architectural style French Gothic architecture

Most Similar Search Results for "Blood_Scent (Album) genre Groove_metal (extreme metal,music genre)" - Search Mode [X]:
  1. Label: genre
     URI: http://www.wikidata.org/entity/P136
     Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic
     Example: Grand Theft Auto V genre first-person shooter
  2. Label: country
     URI: http://www.wikidata.org/entity/P17
     Description: sovereign state that this item is in (not to be used for human beings)
     Example: Germany country Germany
  3. Label: peak bagging classification
     URI: http://www.wikidata.org/entity/P8450
     Description: recognised peak bagging classification of a mountain or hill
     Example: Mount Everest peak bagging classification eight-thousander

Most Similar Search Results for "Groove_metal (extreme metal,music genre)" - Search Mode [Q]:
  1. Label: Groove_metal
     URI: http://www.wikidata.org/entity/Q241662
     Description: subgenre of heavy metal
  2. Label: Groovin'_Blue
     URI: http://www.wikidata.org/entity/Q25095540
     Description: album by Curtis Amy
  3. Label: Rock_music
     URI: http://www.wikidata.org/entity/Q11399
     Description: popular music genre

Most Similar Search Results for "performer" - Search Mode [P]:
  1. Label: performer
     URI: http://www.wikidata.org/entity/P175
     Description: actor, musician, band or other performer associated with this role or musical work
     Example: Luke Skywalker performer Mark Hamill
  2. Label: cast member
     URI: http://www.wikidata.org/entity/P161
     Description: actor performing live for a camera or audience [use "character role" (P453) as qualifier] [use "voice actor" (P725) for voice-only role]
     Example: Titanic cast member Frances Fisher
  3. Label: producer
     URI: http://www.wikidata.org/entity/P162
     Description: person(s) who produced the film, musical work, theatrical production, etc. (for film, this does not include executive producers, associate producers, etc.) [for production company, use P272, video games - use P178]
     Example: Citizen Kane producer Orson Welles

Most Similar Search Results for "Blood_Scent (Album) performer STEMM (musical group)" - Search Mode [X]:
  1. Label: performer
     URI: http://www.wikidata.org/entity/P175
     Description: actor, musician, band or other performer associated with this role or musical work
     Example: Luke Skywalker performer Mark Hamill
  2. Label: field of this occupation
     URI: http://www.wikidata.org/entity/P425
     Description: field corresponding to this occupation or profession (use only for occupations/professions - for people use Property:P101, for companies use P452)
     Example: painter field of this occupation art of painting
  3. Label: organizer
     URI: http://www.wikidata.org/entity/P664
     Description: person or institution organizing an event
     Example: Eurovision Song Contest organizer European Broadcasting Union

Most Similar Search Results for "STEMM (musical group)" - Search Mode [Q]:
  1. Label: STEMM
     URI: http://www.wikidata.org/entity/Q596622
     Description: American metal band
  2. Label: Music
     URI: http://www.wikidata.org/entity/Q638
     Description: art/activity of creating art using sound
  3. Label: Intelligent_dance_music
     URI: http://www.wikidata.org/entity/Q660984
     Description: style of electronic dance music

Most Similar Search Results for "follows" - Search Mode [P]:
  1. Label: follows
     URI: http://www.wikidata.org/entity/P155
     Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]
     Example: April follows March
  2. Label: followed by
     URI: http://www.wikidata.org/entity/P156
     Description: immediately following item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]
     Example: March followed by April
  3. Label: including
     URI: http://www.wikidata.org/entity/P1012
     Description: usually used as a qualifier

Most Similar Search Results for "Blood_Scent (Album) follows Songs_for_the_Incurable_Heart (Album)" - Search Mode [X]:
  1. Label: follows
     URI: http://www.wikidata.org/entity/P155
     Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]
     Example: April follows March
  2. Label: followed by
     URI: http://www.wikidata.org/entity/P156
     Description: immediately following item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]
     Example: March followed by April
  3. Label: conferred by
     URI: http://www.wikidata.org/entity/P1027
     Description: person or organization who grants an award, certification, grant, or role
     Example: Juno Awards conferred by Canadian Academy of Recording Arts and Sciences

Most Similar Search Results for "Songs_for_the_Incurable_Heart (Album)" - Search Mode [Q]:
  1. Label: Songs_for_the_Incurable_Heart
     URI: http://www.wikidata.org/entity/Q7561421
     Description: album by STEMM
  2. Label: Album
     URI: http://www.wikidata.org/entity/Q482994
     Description: collection of recorded music, words, sounds
  3. Label: Single_(music)
     URI: http://www.wikidata.org/entity/Q134556
     Description: group of single releases by an artist usually released at the same time with the same title and tracks but in different formats for consumption (digital, CD, LP)
                                                        
END OF INPUT 2

START OF OUTPUT 2
<uri_mapping>
Blood_Scent (Album) - Search mode applied [Q] - Applicable to [S1, S2, S3]
URI: http://www.wikidata.org/entity/Q4927704
URI-Label: Blood_Scent
URI-Description: album by STEMM

genre - Search mode applied [P, X] - Applicable to [P1]
URI: http://www.wikidata.org/entity/P136
URI-Label: genre
URI-Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic

Groove_metal (extreme metal,music genre) - Search mode applied [Q] - Applicable to [O1]
URI: http://www.wikidata.org/entity/Q241662
URI-Label: Groove_metal
URI-Description: subgenre of heavy metal

performer - Search mode applied [P, X] - Applicable to [P2]
URI: http://www.wikidata.org/entity/P175
URI-Label: performer
URI-Description: actor, musician, band or other performer associated with this role or musical work

STEMM (musical group) - Search mode applied [Q] - Applicable to [O2]
URI: http://www.wikidata.org/entity/Q596622
URI-Label: STEMM
URI-Description: American metal band

follows - Search mode applied [P, X] - Applicable to [P3]
URI: http://www.wikidata.org/entity/P155
URI-Label: follows
URI-Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]

Songs_for_the_Incurable_Heart (Album) - Search mode applied [Q] - Applicable to [O3]
URI: http://www.wikidata.org/entity/Q7561421
URI-Label: Songs_for_the_Incurable_Heart
URI-Description: album by STEMM
</uri_mapping>
<goto>validation_and_output</goto>
END OF OUTPUT 2

END OF EXAMPLES
                                                                 
Additional Instructions: {agent_instruction}

Text: {text}

Triples: {triples}

Output for "{last_call}": {last_response}

START OF YOUR OUTPUT:
""")

validation_and_output_prompt = PromptTemplate.from_template("""
You are an expect in validating triples for closed information extraction and deciding whether a better result could be reached or not. You will receive a set of triples and the text they were extracted from. You have to check if the triples are valid and if they are not, you have to decide whether which agent to call next.

To process the task you have access to the following tools:
- Turtle to Labels Tool
    - ID: turtle_to_labels_tool
    - Description: Takes a turtle string and returns the corresponding labels for each URI in the triples. The Output will be written into the Output on Last Call field.

To continue you could call one of the following agents:
- Extractor
    - ID: extractor
    - Description: Extracts triples from a given text.
- URI Mapping & Refinement
    - ID: uri_mapping_and_refinement
    - Description: Maps the triples with URIs and refines the triples. 

START OF EXAMPLES

START OF INPUT 1
Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.

Triples:
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
3: Blood_Scent (Album); follows; Songs_for_the_Incurable_Heart (Album)

URI Mapping:
Blood_Scent (Album) - Search mode applied [Q] - Applicable to [S1, S2, S3]
URI: http://www.wikidata.org/entity/Q4927704
URI-Label: Blood_Scent
URI-Description: album by STEMM

genre - Search mode applied [P, X] - Applicable to [P1]
URI: http://www.wikidata.org/entity/P136
URI-Label: genre
URI-Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic

Groove_metal (extreme metal,music genre) - Search mode applied [Q] - Applicable to [O1]
URI: http://www.wikidata.org/entity/Q241662
URI-Label: Groove_metal
URI-Description: subgenre of heavy metal

performer - Search mode applied [P, X] - Applicable to [P2]
URI: http://www.wikidata.org/entity/P175
URI-Label: performer
URI-Description: actor, musician, band or other performer associated with this role or musical work

STEMM (musical group) - Search mode applied [Q] - Applicable to [O2]
URI: http://www.wikidata.org/entity/Q596622
URI-Label: STEMM
URI-Description: American metal band

follows - Search mode applied [P, X] - Applicable to [P3]
URI: http://www.wikidata.org/entity/P155
URI-Label: follows
URI-Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]

Songs_for_the_Incurable_Heart (Album) - Search mode applied [Q] - Applicable to [O3]
URI: http://www.wikidata.org/entity/Q7561421
URI-Label: Songs_for_the_Incurable_Heart
URI-Description: album by STEMM

Output for "URI Search Tool - INPUT: Blood_Scent (Album)[Q]|genre[P]|Blood_Scent (Album) genre Groove_metal (extreme metal,music genre)[X]|Groove_metal (extreme metal,music genre)[Q]|performer[P]|Blood_Scent (Album) performer STEMM (musical group)[X]|STEMM (musical group)[Q]|follows[P]|Blood_Scent (Album) follows Songs_for_the_Incurable_Heart (Album)[X]|Songs_for_the_Incurable_Heart (Album)[Q]": <uri_mapping>
Blood_Scent (Album) - Search mode applied [Q] - Applicable to [S1, S2, S3]
URI: http://www.wikidata.org/entity/Q4927704
URI-Label: Blood_Scent
URI-Description: album by STEMM

genre - Search mode applied [P, X] - Applicable to [P1]
URI: http://www.wikidata.org/entity/P136
URI-Label: genre
URI-Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic

Groove_metal (extreme metal,music genre) - Search mode applied [Q] - Applicable to [O1]
URI: http://www.wikidata.org/entity/Q241662
URI-Label: Groove_metal
URI-Description: subgenre of heavy metal

performer - Search mode applied [P, X] - Applicable to [P2]
URI: http://www.wikidata.org/entity/P175
URI-Label: performer
URI-Description: actor, musician, band or other performer associated with this role or musical work

STEMM (musical group) - Search mode applied [Q] - Applicable to [O2]
URI: http://www.wikidata.org/entity/Q596622
URI-Label: STEMM
URI-Description: American metal band

follows - Search mode applied [P, X] - Applicable to [P3]
URI: http://www.wikidata.org/entity/P155
URI-Label: follows
URI-Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]

Songs_for_the_Incurable_Heart (Album) - Search mode applied [Q] - Applicable to [O3]
URI: http://www.wikidata.org/entity/Q7561421
URI-Label: Songs_for_the_Incurable_Heart
URI-Description: album by STEMM
</uri_mapping>
<goto>validation_and_output</goto>

END OF INPUT 1

START OF OUTPUT 1
<goto>turtle_to_labels_tool</goto>
<tool_input>@prefix wd: <http://www.wikidata.org/entity/> .
wd:Q4927707 wd:P136 wd:Q241662.
wd:Q4927707 wd:P175 wd:Q596622.
wd:Q4927707 wd:P155 wd:Q7561421.
</tool_input>

END OF OUTPUT 1

START OF INPUT 2

Text: Blood Scent is a groove metal song performed by STEMM, following their earlier release, Songs for the Incurable Heart.

Triples:
1: Blood_Scent (Album); genre; Groove_metal (extreme metal,music genre)
2: Blood_Scent (Album); performer; STEMM (musical group)
3: Blood_Scent (Album); follows; Songs_for_the_Incurable_Heart (Album)

URI Mapping:
Blood_Scent (Album) - Search mode applied [Q] - Applicable to [S1, S2, S3]
URI: http://www.wikidata.org/entity/Q4927704
URI-Label: Blood_Scent
URI-Description: album by STEMM

genre - Search mode applied [P, X] - Applicable to [P1]
URI: http://www.wikidata.org/entity/P136
URI-Label: genre
URI-Description: creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic

Groove_metal (extreme metal,music genre) - Search mode applied [Q] - Applicable to [O1]
URI: http://www.wikidata.org/entity/Q241662
URI-Label: Groove_metal
URI-Description: subgenre of heavy metal

performer - Search mode applied [P, X] - Applicable to [P2]
URI: http://www.wikidata.org/entity/P175
URI-Label: performer
URI-Description: actor, musician, band or other performer associated with this role or musical work

STEMM (musical group) - Search mode applied [Q] - Applicable to [O2]
URI: http://www.wikidata.org/entity/Q596622
URI-Label: STEMM
URI-Description: American metal band

follows - Search mode applied [P, X] - Applicable to [P3]
URI: http://www.wikidata.org/entity/P155
URI-Label: follows
URI-Description: immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]

Songs_for_the_Incurable_Heart (Album) - Search mode applied [Q] - Applicable to [O3]
URI: http://www.wikidata.org/entity/Q7561421
URI-Label: Songs_for_the_Incurable_Heart
URI-Description: album by STEMM

Output of "Turtle to Labels Tool - INPUT: @prefix wd: <http://www.wikidata.org/entity/> .
wd:Q4927707 wd:P136 wd:Q241662.
wd:Q4927707 wd:P175 wd:Q596622.
wd:Q4927707 wd:P155 wd:Q7561421.":

Turtle to Labels Tool Output - 
Blood Scent follows Songs for the Incurable Heart
Blood Scent genre groove metal
Blood Scent performer STEMM

END OF INPUT 2

START OF OUTPUT 2
<goto>END</goto>
<ttl>
@prefix wd: <http://www.wikidata.org/entity/> .
wd:Q4927707 wd:P136 wd:Q241662.
wd:Q4927707 wd:P175 wd:Q596622.
wd:Q4927707 wd:P155 wd:Q7561421.
</ttl>
END OF OUTPUT 2

END OF EXAMPLES

Text: {text}

Triples: {triples}

URI Mapping: {uri_mapping}

Output for "{last_call}": {last_response}

START OF YOUR OUTPUT:
""")