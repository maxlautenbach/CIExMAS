# CIExMAS - Closed Information Extraction using Multi-Agent-System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange.svg)]()

> **Advanced Multi-Agent System for Closed Information Extraction from Text**

CIExMAS is a sophisticated framework that leverages multiple AI agents to perform closed information extraction (cIE) tasks. The system extracts structured knowledge from unstructured text by identifying entities, relations, and properties, then mapping them to Wikidata URIs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Infrastructure](#infrastructure)
3. [Multi-Agent Approaches](#multi-agent-approaches)
4. [Project Structure](#project-structure)
5. [Datasets & Evaluation](#datasets--evaluation)
6. [Helper Tools](#helper-tools)
7. [Model Providers](#model-providers)

---

## Quick Start

### Requirements

- **Huggingface Account** with CLI access
- **Python 3.11**
- **Docker & Docker Compose**
- **Langfuse Local or Cloud** for tracing and monitoring

### Step-by-Step Setup

1. **Clone and Install Dependencies**

   ```bash
   git clone <repository-url>
   cd CIExMAS
   pip install -r requirements.txt
   ```

2. **Start Infrastructure**

   ```bash
   docker-compose -f ./infrastructure/docker-compose.yaml up -d
   ```

3. **Configure Environment**

   ```bash
   cp template.env .env
   # Edit .env with your API keys and configurations
   # Not all fields need to be filled, especially those referring to LLM providers you don't want to use
   ```

4. **Setup Langfuse**

   - Configure a project in [Langfuse](https://langfuse.com/docs/observability/get-started)
   - Extract the API keys into your `.env` file

5. **Configure LLM Settings**

   ```bash
   python helper_tools/set_llm_config.py
   # This loads preconfigured LLM models with their corresponding rate limits
   ```

6. **Verify Setup**
   ```bash
   # The triple and vector stores will be filled automatically if base_setup is imported
   python -c "from helper_tools.base_setup import *; print('Setup complete!')"
   ```

### Run Your First Evaluation

```bash
python ./approaches/Network/Gen2/slurm/agent_system.py \
  --dataset synthie_text \
  --num_samples 10 \
  --description "Quick test run"
```

> **Note**: You can now run all Jupyter Notebooks except for the synthIE and GenIE notebooks. For those, refer to [https://github.com/epfl-dlab/SynthIE](https://github.com/epfl-dlab/SynthIE).

---

## Infrastructure

CIExMAS uses the following infrastructure:

| Service         | Purpose                                                                                         | Status    |
| --------------- | ----------------------------------------------------------------------------------------------- | --------- |
| **Jena-Fuseki** | Saving a Wikidata Dump with a SPARQL Endpoint                                                   | ✅ Active |
| **Langfuse**    | Tracing MAS Calls & Costs for debugging                                                         | ✅ Active |
| **Qdrant**      | Vector Database to store and retrieve entity labels and descriptions                            | ✅ Active |
| **Redis**       | Fast in-memory key-value store for fast retrieval of labels or existence of entities/properties | ✅ Active |

---

## Multi-Agent Approaches

### Baseline Architecture

The Baseline Architecture is an approach to cIE that is inspired by common pipeline approaches.

**Agents:**

- Supervisor
- Entity Extractor
- Relation Extractor

**Tools:**

- URI Retrieval Tool (with LLM Summary)

### Splitted Supervisor Architecture (Gen1)

The Splitted Supervisor Architecture breaks the Baseline approach down to simpler tasks. This especially includes splitting the supervisor to the planning and reasoning. As the Baseline approach showed problems in creating a plan and checking the results within one supervisor agent with one general prompt.

**Agents:**

- Planner
- Agent Instructor
- Entity Extractor
- Relation Extractor
- Result Checker
- Result Formatter

**Tools:**

- URI Retrieval Tool (with LLM Summary)

### Splitted Supervisor Architecture with Property Extraction (Gen1_PredEx)

The Splitted Supervisor Architecture showed immense problems especially with property extraction. That is why another extension was done to this architecture with another agent for property extraction.

**Agents:**

- Planner
- Agent Instructor
- Entity Extractor
- Relation Extractor
- Property Extractor (alias Predicate Extractor)
- URI Retriever Agent (connected to VectorDB)
- Result Checker
- Result Formatter

**Tools:**

- URI Retrieval Tool (with LLM Summary)

### Simplified Splitted Supervisor Architecture (Gen1v2)

The Simplified Splitted Supervisor Architecture simplifies the planner, agent instructor and result checker back into one planner agent. This is grounded in the context bloating and miserable results of the splitted supervisor architecture.

**Agents:**

- Planner
- Entity Extractor
- Relation Extractor
- Property Extractor (alias Predicate Extractor)
- Result Formatter

**Tools:**

- URI Retrieval Tool (with LLM Summary)

### ReAct Architecture (One Agent)

The ReAct architecture is the clear opposite draft of the both splitted supervisor architectures. It was initially provided with a single agent and the URI retrieval tool without LLM summary. During the experiments, the ReAct architecture got additional tools for the triple validation.

**Agents:**

- ReAct Agent

**Tools:**

- URI Retrieval Tool
- Semantic Validation Tool
- Turtle to Label Tool

### Network Architecture (Gen2)

The network architecture merges the learnings of all architectures into three agents, that cover triple extraction, the mapping of URI and the validation and formation of the final output.

**Agents:**

- Triple Extractor
- URI Mapping & Refinement Agent
- Validation & Output Agent

**Tools:**

- URI Retrieval Tool
- Semantic Validation Tool
- Turtle to Label Tool

---

## Project Structure

```
CIExMAS/
├── approaches/                    # Code/Evaluation for various approaches
├── datasets/                     # (only local) Folder to store HF-datasets
├── helper_tools/                 # Tools to handle datasets and repetitive tasks
├── infrastructure/               # Infrastructure to handle tracing, initial population and storage of graphs
└── results/                      # Result of configurations used for the report
```

### General Files per Approach

Each approach in CIExMAS follows a consistent file structure to ensure modularity and maintainability:

**Agents Directory:**

```
agents/
├── Agent1.py          # Main agent implementation
├── Agent2.py          # Secondary agent implementation
├── Tool1.py           # Custom tools for agents
└── Tool2.py           # Additional tools
```

**SLURM Execution:**

```
slurm/
├── agent_system.py        # Combines all agents and tools into LangGraph and executes evaluation
└── execution_script.sh    # SLURM cluster execution script (boilerplate for other configurations)
```

**Core Files:**

| File         | Purpose                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------- |
| `prompts.py` | All prompts as LangChain prompt templates (imported by Agent.py files)                         |
| `setup.py`   | Defines agent system state and inherits base setup (LLM, Vector Store, Triple Store & Tracing) |

---

## Datasets & Evaluation

### Run Evaluation

To run an evaluation use:

```bash
python ./approaches/<INSERT_YOUR_APPROACH>/slurm/agent_system.py \
  --dataset <DATASET_OF_CHOICE> \
  --num_samples <NUMBER_OF_SAMPLES> \
  --description <DESCRIPTION_FOR_LOGGING>
```

### Supported Datasets

| Dataset          | ID             | Description                      | Split Mapping                   |
| ---------------- | -------------- | -------------------------------- | ------------------------------- |
| **synthIE Text** | `synthie_text` | synthIE dataset text_davinci_003 | test → test, train → validation |
| **synthIE Code** | `synthie_code` | synthIE dataset code_davinci_002 | test → test, train → train      |
| **REBEL**        | `rebel`        | Relation extraction dataset      | test → test, train → train      |
| **REDFM**        | `redfm`        | Refined relation dataset         | test → test, train → train      |

### Performance Metric

Josifoski et al. 2022 - GenIE (S. 4639)

A fact is regarded as correct if the relation and the two corresponding entities are all correct.

### synthIE - text_davinci_003

| Approach                  | P (Micro) | R (Micro) | F1 (Micro) | P (Macro) | R (Macro) | F1 (Macro) |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| GenIE T5-base             |     49.10 |     26.69 |      34.58 |     29.82 |     11.14 |      13.94 |
| SynthIE T5-base           |     92.08 |     90.75 |      91.41 |     94.10 |     92.42 |      93.05 |
| SynthIE T5-base-SC        |     92.79 |     90.50 |      91.63 |     94.35 |     92.39 |      93.15 |
| **SynthIE T5-large**      | **93.38** | **92.69** |  **93.04** | **95.27** | **94.95** |  **94.99** |
| CIExMAS Gen2 (50 Samples) |     73.24 |     61.54 |      66.88 |     72.77 |     63.52 |      66.54 |

### synthIE - code_davinci_002

| Approach                  | P (Micro) | R (Micro) | F1 (Micro) | P (Macro) | R (Macro) | F1 (Macro) |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| GenIE T5-base             |     41.56 |     23.94 |      30.38 |     25.78 |      9.81 |      12.12 |
| SynthIE T5-base           |     79.99 |     70.47 |      74.93 |     83.76 |     74.05 |      77.91 |
| SynthIE T5-base-SC        |     81.58 |     69.48 |      75.05 |     84.32 |     73.57 |      77.88 |
| **SynthIE T5-large**      | **82.60** | **73.15** |  **77.59** | **86.43** | **78.78** |  **81.95** |
| CIExMAS Gen2 (50 Samples) |     48.25 |     40.83 |      44.23 |     47.83 |     46.87 |      45.73 |

---

## Helper Tools

CIExMAS includes a comprehensive set of helper tools to ensure seamless development and operation. These tools handle various aspects from data processing to evaluation and infrastructure management.

### Agent Builder UI

The agent builder UI provides an interactive interface for managing and evaluating your approaches:

- **Comparison** - Compare all evaluation logs side by side
- **Inspect Eval Log** - Review evaluation logs in detail with optional Langfuse trace IDs
- **Log Notes** - Inspect and modify approach labels and descriptions
- **Step By Step Evaluation** - Test agents incrementally and iterate on improvements

### Core Tools

| Tool                | Description                                                                                             | Status    |
| ------------------- | ------------------------------------------------------------------------------------------------------- | --------- |
| `base_setup.py`     | Initializes access to LLM, Vector Store, Triple Store, and Tracing. Populates Triple Store if necessary | ✅ Active |
| `evaluation.py`     | Handles all evaluation steps and generates reports per test run                                         | ✅ Active |
| `set_llm_config.py` | Sets preconfigured LLM IDs, providers and rate limits in `.env`                                         | ✅ Active |
| `parser.py`         | Parses datasets and initiates embeddings to the vector store                                            | ✅ Active |

### Storage Handlers

| Handler             | Purpose                                          | Status        |
| ------------------- | ------------------------------------------------ | ------------- |
| `qdrant_handler.py` | Qdrant Vector store communication and population | ✅ Active     |
| `redis_handler.py`  | Redis store communication and population         | ✅ Active     |
| `fuseki_handler.py` | Fuseki SPARQL endpoint communication             | ✅ Active     |
| `faiss_handler.py`  | FAISS vector store handling                      | ❌ Deprecated |

### Data Processing

| Tool                   | Description                                             | Status        |
| ---------------------- | ------------------------------------------------------- | ------------- |
| `sort_jsonl.py`        | Maintains dataset sorting for consistent sample subsets | ✅ Active     |
| `upload_predicates.py` | Upload predicates from turtle files to Qdrant           | ❌ Deprecated |
| `validation.py`        | Collection of methods to validate triples               | ✅ Active     |
| `wikidata_loader.py`   | Handler for Wikidata SPARQL endpoint communication      | ✅ Active     |

### Templates & Scripts

- **Chat Templates** - Templates for running Llama 3 and Gemma models on vLLM
- **Scripts**:
  - `display_document_triples.py` - Generate few-shot examples
  - `rename_evaluation_logs.txt` - Convert between synthIE dataset IDs

---

## Model Providers

### OpenAI

| Model               | ID                                 |
| ------------------- | ---------------------------------- |
| GPT 4o 2025 Preview | `gpt-4o-search-preview-2025-03-11` |

### SambaNova

| Model            | ID                                   |
| ---------------- | ------------------------------------ |
| Llama 3.3 70B    | `Meta-Llama-3.3-70B-Instruct`        |
| QwQ 32B          | `QwQ-32B`                            |
| Llama 4 Maverick | `Llama-4-Maverick-17B-128E-Instruct` |
| Llama 4 Scout    | `Llama-4-Scout-17B-16E-Instruct`     |

### DeepInfra

| Model                       | ID                                          |
| --------------------------- | ------------------------------------------- |
| Llama 3.3 70B               | `meta-llama/Llama-3.3-70B-Instruct`         |
| Gemma 3 27b                 | `google/gemma-3-27b-it`                     |
| DeepSeek R1 (Llama Distill) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |
| DeepSeek R1 (Qwen Distill)  | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`  |
| QwQ 32B                     | `Qwen/QwQ-32B`                              |

### vLLM

| Model         | ID                                                | Quantization |
| ------------- | ------------------------------------------------- | ------------ |
| Gemma 3 27b   | `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g`         | GPTQ - 4bit  |
| Llama 3.3 70B | `kosbu/Llama-3.3-70B-Instruct-AWQ`                | AWQ          |
| Command A     | `unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit` | bitsandbytes |

### Cerebras

| Model         | ID                               |
| ------------- | -------------------------------- |
| Llama 4 Scout | `llama-4-scout-17b-16e-instruct` |
| Llama 3.3 70B | `llama-3.3-70b`                  |
