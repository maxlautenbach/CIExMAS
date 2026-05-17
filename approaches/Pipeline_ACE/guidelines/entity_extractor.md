# Entity Extraction Guidelines

## Rules
- Each entity must be atomic — one concept per entry, never merge multiple concepts into one phrase
- Only extract entities specific enough to have a dedicated encyclopedia entry (e.g., "Ice hockey" yes, "sport" no; "Film" yes, "film production company" no)
- Extract type classifications when the text explicitly states "X is a Y" (e.g., "Novelist", "Human", "Government agency")
- Do not extract compound descriptors or relational phrases (e.g., "film production company", "Canadian citizen", "bibliographic review")
- Do not extract generic classifiers that are not the direct type of a subject (e.g., "project", "edition")
