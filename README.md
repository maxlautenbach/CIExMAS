# CIExMAS - Closed Information Extraction using Multi-Agent-System
## Project Structure
```
/approaches - Code/Evaluation for various approaches
/datasets - (Optional) Folder to store HF-datasets
/helper-tools - Tools to handle datasets and repetitive tasks
/infrastructure - Infrastructure to handle tracing and storage of graphs
```

## Table of Contents
1. [Setup](#setup)
2. [Infrastructure](#infrastructure)
3. [Approaches](#approaches)

## Setup

### Requirements
- Huggingface Account
- Python 3.11
- [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

### Step-by-Step
1. Run `docker-compose -f ./infrastructure/docker-compose.yaml -d` to start all infrastructure containers
2. Install all required packages `pip install -r requirements.txt`

Now you are able to run all Jupyter Notebooks

## Infrastructure
CIExMAS uses the following tools:
- Jena-Fuseki - Saving a Wikidata Dump with a SPARQL Endpoint
- Langfuse - Tracing MAS Calls & Costs in order to debug
- Qdrant - Vector Database to store and retrieve entity labels and descriptions

## Approaches
### Baseline
The Baseline approach is a very basic MAS to do a cIE. It consists of the following agents:
- Supervisor
- Entity Extractor
- Relation Extractor
- URI Retriever Agent (connected to VectorDB)

Characteristic:
- Full-Sentence
- Network

### Gen 1
The Gen 1 breaks the Baseline approach down to more simpler task. This especially includes the planning and reasoning. As the Baseline approached showed up problems in creating a plan and check the results within one supervisor agent with one general prompt. Therefore, Gen 1 will contain the following agents:
- Planner
- Agent Instructor
- Entity Extractor
- Relation Extractor
- URI Retriever Agent (connected to VectorDB)
- Result Checker

Characteristic:
- Full-Senctence
- Network
