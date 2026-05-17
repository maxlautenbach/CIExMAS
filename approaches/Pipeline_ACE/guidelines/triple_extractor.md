# Triple Extraction Guidelines

## Rules

- Use specific predicates, not generic ones. Prefer narrow Wikidata-style property names:
  - "headquarters location" for where an organization's office is physically located
  - "country" for the country an organization belongs to
  - "country of origin" for where a publication/work originates from
  - "parent organization" not "under" (for organizational hierarchy)
  - "file format" not "uses" (for data/output formats)
  - "sport" not "consists of" (for the sport a competition involves)
  - "original language of film or TV show" for media language
  - "language of work or name" for publication/journal language
  - "indexed in bibliographic review" not "indexed in" (for bibliographic databases like Scopus)
  - "instance of" for type classifications (e.g., X instance of Government_agency)
  - "publisher" not "published by" (for the publisher of a journal/work)
  - "owned by" not "owns" (owner relationship, work is subject)
  - "replaced by" not "successor" (for entity succession)
  - "developer" not "developed by" (for software/product development)
  - "inspired by" not "based on" (for creative inspiration)
  - "product or material produced" not "produces" (for output of organizations)
  - "main subject" not "subject" or "focused on" (for topics of works/organizations)

- The primary topic of the sentence is the subject. For creative works (films, journals, software):
  - "Film; screenwriter; Person" not "Person; screenwriter of; Film"
  - "Journal; publisher; Company" not "Company; publishes; Journal"
  - "Software; developer; Organization" not "Organization; developed; Software"

- When text describes an entity and its successor/replacement, attribute properties to the PRIMARY entity (the one the text is ABOUT), not its successor
  - "X was a government agency under Z, headquartered in Y. It was replaced by W." → Subject is X for all properties: "X; headquarters location; Y", "X; parent organization; Z", "X; replaced by; W"
  - NEVER assign headquarters/country/parent to the successor entity

- Extract "instance of" triples ONLY when the text explicitly classifies an entity's type (e.g., "X is a government agency" → "X; instance of; Government agency"). Do NOT use "instance of" for topics/focus areas — use "main subject" instead.

- Ownership direction: "Product; owned by; Company" not "Company; owned by; Product"

- Distinguish "country" (sovereign state an org belongs to) from "headquarters location" (physical location of office/HQ)
  - "Agency X of China, headquartered in Y" → both "X; country; China" AND "X; headquarters location; Y"
