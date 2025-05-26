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

## Datasets & Evaluation

### Performance Metric

Josifoski et. al. 2022 - GenIE (S. 4639)

A fact is regarded as correct if the relation and the two corresponding entities are all correct.

### synthIE - text_davinci_003

| Approach                  | P (Micro) | R (Micro) | F1 (Micro) | P (Macro) | R (Macro) | F1 (Macro) |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| GenIE T5-base             |     49.10 |     26.69 |      34.58 |     29.82 |     11.14 |      13.94 |
| SynthIE T5-base           |     92.08 |     90.75 |      91.41 |     94.10 |     92.42 |      93.05 |
| SynthIE T5-base-SC        |     92.79 |     90.50 |      91.63 |     94.35 |     92.39 |      93.15 |
| **SynthIE T5-large**      | **93.38** | **92.69** |  **93.04** | **95.27** | **94.95** |  **94.99** |
| CIExMAS Gen2 (50 Samples) |     73.24 |     61.54 |      66.88 |     72.77 |     63.52 |      66.54 |

### synthIE - code_davinci_002

| Approach                  | P (Micro) | R (Micro) | F1 (Micro) | P (Macro) | R (Macro) | F1 (Macro) |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| GenIE T5-base             |     41.56 |     23.94 |      30.38 |     25.78 |      9.81 |      12.12 |
| SynthIE T5-base           |     79.99 |     70.47 |      74.93 |     83.76 |     74.05 |      77.91 |
| SynthIE T5-base-SC        |     81.58 |     69.48 |      75.05 |     84.32 |     73.57 |      77.88 |
| **SynthIE T5-large**      | **82.60** | **73.15** |  **77.59** | **86.43** | **78.78** |  **81.95** |
| CIExMAS Gen2 (50 Samples) |     48.25 |     40.83 |      44.23 |     47.83 |     46.87 |      45.73 |

### REBEL

| Approach               | P (Micro) | R (Micro) | F1 (Micro) | P (Macro) | R (Macro) | F1 (Macro) |
| ---------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| SOTA-Pipeline (GenIE)  |     43.30 |     41.73 |      42.50 |     12.20 |     10.44 |       9.48 |
| GenIE                  |     68.02 |     69.87 |      68.93 |     33.90 |     30.48 |      30.46 |
| GenIE - PLM            |     59.32 |     77.78 |      67.31 |         – |         – |          – |
| DISCIE (F2 calibrated) |     62.13 | **81.93** |      70.67 |     35.84 |     43.99 |      39.50 |
| DISCIE (F1 calibrated) | **77.41** |     72.68 |      74.97 | **44.05** | **42.29** |  **34.11** |
| ReLiK_L                |         - |         - |  **75.60** |         - |         - |          - |

## Model Providers and IDs

### OpenAI

- GPT 4o 2025 Preview - gpt-4o-search-preview-2025-03-11

### SambaNova

- Llama 3.3 70B - Meta-Llama-3.3-70B-Instruct
- QwQ 32B - QwQ-32B
- Llama 4 Maverick - Llama-4-Maverick-17B-128E-Instruct
- LLama 4 Scout - Llama-4-Scout-17B-16E-Instruct

### DeepInfra

- Llama 3.3 70B - meta-llama/Llama-3.3-70B-Instruct
- Gemma 3 27b - google/gemma-3-27b-it
- DeepSeek R1 (Llama Distill) - deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- DeepSeek R1 (Qwen Distill) - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- QwQ 32B - Qwen/QwQ-32B

### vLLM

- Gemma 3 27b - ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g (GPTQ - 4bit)
- Llama 3.3 70B - kosbu/Llama-3.3-70B-Instruct-AWQ (AWQ)
- Command A - unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit (bitsandbytes)

### Cerebras

- Llama 4 Scout - llama-4-scout-17b-16e-instruct
- Llama 3.3 70B - llama-3.3-70b
