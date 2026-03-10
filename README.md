# CIExMAS - Closed Information Extraction using Multi-Agent Systems

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange.svg)]()

> **Multi-Agent Systems for Closed Information Extraction on Wikidata**

CIExMAS explores multi-agent system (MAS) architectures for closed information extraction (cIE). Given unstructured text, the system extracts structured knowledge triples (subject, predicate, object) and maps them to Wikidata URIs. We systematically compare three MAS architectures - Supervisor, ReAct, and Network - and show that the Network Architecture achieves the best performance, reaching a triple-level F1 score of 0.693 on 50 samples from the Wiki-cIE text test split.

**Paper:** *Closed Information Extraction with Multi-Agent Systems* (ESWC 2026)

## Table of Contents

1. [Multi-Agent Architectures](#multi-agent-architectures)
2. [Additional Architectures](#additional-architectures)
3. [Results](#results)
4. [Quick Start](#quick-start)
5. [Infrastructure](#infrastructure)
6. [Project Structure](#project-structure)
7. [Datasets & Evaluation](#datasets--evaluation)
8. [Helper Tools](#helper-tools)
9. [Model Providers](#model-providers)

---

## Multi-Agent Architectures

CIExMAS implements three main architectures for closed information extraction, each representing a different strategy for distributing the cIE task across agents and tools.

### Supervisor Architecture (`approaches/Supervisor/`)

The Supervisor Architecture follows a hierarchical design where a central planner agent delegates extraction subtasks to specialized agents.

**Agents:**
- **Planner** - Orchestrates the extraction process and delegates to specialized agents
- **Entity Extractor** - Extracts subject and object entities from text
- **Triple Extractor** - Combines entities and predicates into structured triples

**Tools:**
- URI Retrieval Tool

### ReAct Architecture (`approaches/ReAct/`)

The ReAct Architecture uses a single agent with reasoning-and-acting capabilities, equipped with multiple tools for extraction and validation.

**Agents:**
- **ReAct Agent** - Performs the full cIE task through iterative reasoning and tool use

**Tools:**
- URI Retrieval Tool
- Semantic Validation Tool
- Turtle-to-Label Tool

### Network Architecture (`approaches/Network/`)

The Network Architecture distributes the cIE task across three specialized agents in a network topology, combining the strengths of both previous approaches.

**Agents:**
- **Extraction Agent** - Extracts triples from text
- **URI Mapping & Refinement Agent** - Maps extracted entities/predicates to Wikidata URIs
- **Validation & Output Agent** - Validates triples and produces final Turtle output

**Tools:**
- URI Retrieval Tool
- Semantic Validation Tool
- Turtle-to-Label Tool

---

## Additional Architectures

The `approaches/additional_architectures/` directory contains architectures that were explored during the iterative development process:

| Architecture | Directory | Description |
|---|---|---|
| **Initial Supervisor** | `Initial_Supervisor/` | Initial baseline with a single supervisor agent (Paper Baseline) |
| **Splitted Supervisor** | `Splitted_Supervisor/` | Extended supervisor with task decomposition into specialized agents |
| **Splitted Supervisor with Property Extraction** | `Splitted_Supervisor_PredEx/` | Further extension with dedicated property extraction agent |

These represent intermediate development steps and are included for reproducibility.

---

## Results

All results are evaluated on **50 samples from the Wiki-cIE text test split** using **Llama 3.3 70B**.

### Table 1: Triple-Level Performance on Wiki-cIE

| Model | Precision | Recall | F1 | Parental F1 | Related F1 |
|---|---:|---:|---:|---:|---:|
| synthIE T5-large | 0.835 | 0.832 | 0.831 | 0.841 | 0.858 |
| GenIE T5-base | 0.527 | 0.344 | 0.391 | 0.406 | 0.419 |
| Initial Supervisor | 0.228 | 0.252 | 0.233 | 0.267 | 0.267 |
| Supervisor Architecture | 0.519 | 0.675 | 0.572 | 0.610 | 0.617 |
| ReAct Architecture | 0.674 | 0.517 | 0.572 | 0.616 | 0.626 |
| **Network Architecture** | **0.767** | **0.659** | **0.693** | **0.729** | **0.729** |

### Table 2: Category F1 Scores on Wiki-cIE

| Model | Subjects | Objects | Entities | Properties | Parental F1 | Related F1 |
|---|---:|---:|---:|---:|---:|---:|
| synthIE T5-large | 0.929 | 0.921 | 0.933 | 0.882 | 0.894 | 0.912 |
| GenIE T5-base | 0.689 | 0.583 | 0.712 | 0.459 | 0.501 | 0.527 |
| Initial Supervisor | 0.393 | 0.448 | 0.460 | 0.382 | 0.498 | 0.506 |
| Supervisor Architecture | 0.810 | 0.870 | 0.933 | 0.656 | 0.739 | 0.746 |
| ReAct Architecture | 0.910 | 0.759 | 0.846 | 0.658 | 0.727 | 0.746 |
| **Network Architecture** | **0.905** | **0.854** | **0.897** | **0.735** | **0.790** | **0.800** |

### Additional Results

<details>
<summary>Detailed per-category Precision, Recall, F1</summary>

#### Subjects

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| synthIE T5-large | 0.927 | 0.943 | 0.929 |
| GenIE T5-base | 0.710 | 0.742 | 0.689 |
| Initial Supervisor | 0.380 | 0.420 | 0.393 |
| Supervisor Architecture | 0.742 | 0.968 | 0.810 |
| ReAct Architecture | 0.953 | 0.893 | 0.910 |
| Network Architecture | 0.930 | 0.912 | 0.905 |

#### Objects

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| synthIE T5-large | 0.926 | 0.930 | 0.921 |
| GenIE T5-base | 0.858 | 0.484 | 0.583 |
| Initial Supervisor | 0.441 | 0.469 | 0.448 |
| Supervisor Architecture | 0.833 | 0.948 | 0.870 |
| ReAct Architecture | 0.917 | 0.677 | 0.759 |
| Network Architecture | 0.955 | 0.803 | 0.854 |

#### Entities

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| synthIE T5-large | 0.936 | 0.937 | 0.933 |
| GenIE T5-base | 0.932 | 0.606 | 0.712 |
| Initial Supervisor | 0.456 | 0.468 | 0.460 |
| Supervisor Architecture | 0.909 | 0.972 | 0.933 |
| ReAct Architecture | 0.953 | 0.778 | 0.846 |
| Network Architecture | 0.966 | 0.853 | 0.897 |

#### Properties (incl. Hierarchical Scores)

| Model | Precision | Recall | F1 | Parental F1 | Related F1 |
|---|---:|---:|---:|---:|---:|
| synthIE T5-large | 0.882 | 0.890 | 0.882 | 0.894 | 0.912 |
| GenIE T5-base | 0.646 | 0.395 | 0.459 | 0.501 | 0.527 |
| Initial Supervisor | 0.382 | 0.392 | 0.382 | 0.498 | 0.506 |
| Supervisor Architecture | 0.629 | 0.716 | 0.656 | 0.739 | 0.746 |
| ReAct Architecture | 0.775 | 0.596 | 0.658 | 0.727 | 0.746 |
| Network Architecture | 0.832 | 0.684 | 0.735 | 0.790 | 0.800 |

</details>

Full result data is available in `results/result_reports/` (CSV) and `results/result_evaluation_logs/` (Excel).

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
   python -c "from helper_tools.base_setup import *; print('Setup complete!')"
   ```

### Alternative: SLURM Cluster Setup (Step 1 Alternative)

For running CIExMAS on a SLURM cluster, you can use the provided setup script as an alternative to Step 1:

```bash
git clone <repository-url>
cd CIExMAS
bash infrastructure/setup_slurm.sh
```

This script will:

- Install Miniconda3
- Clone the CIExMAS repository
- Create a conda environment with Python 3.11
- Install all required dependencies including vLLM and FlashInfer
- Set up model and dataset directories
- Install Ollama for local model serving

**Note**: The SLURM setup is designed for Linux environments with CUDA support. After running this script, continue with Steps 2-6 from above.

### Run Your First Evaluation

```bash
python ./approaches/Network/slurm/agent_system.py \
  --dataset wiki_cie_text \
  --split test \
  --num_samples 10 \
  --description "Quick test run"
```

> **Note**: For running the synthIE and GenIE benchmark notebooks, refer to [https://github.com/epfl-dlab/SynthIE](https://github.com/epfl-dlab/SynthIE).

### Running synthIE / GenIE Baselines

The fine-tuned synthIE and GenIE model evaluations are run via dedicated notebooks in `results/`:

| Notebook | Model | Evaluation |
|---|---|---|
| `synthIE-large-fe.ipynb` | synthIE T5-large | Full extraction |
| `synthIE-base-fe.ipynb` | synthIE T5-base | Full extraction |
| `synthIE-base-sc.ipynb` | synthIE T5-base | Set constrained |
| `genIE-base-fe.ipynb` | GenIE T5-base | Full extraction |
| `genIE-base-sc.ipynb` | GenIE T5-base | Set constrained |

These notebooks require the model checkpoints from the [SynthIE repository](https://github.com/epfl-dlab/SynthIE). Follow their setup instructions to download the pretrained models, then run the notebooks to generate evaluation logs in `approaches/evaluation_logs/`.

---

## Infrastructure

| Service | Purpose | Status |
|---|---|---|
| **Jena-Fuseki** | Wikidata Dump with SPARQL Endpoint | Active |
| **Langfuse** | Tracing MAS Calls & Costs for debugging | Active |
| **Qdrant** | Vector Database for entity labels and descriptions | Active |
| **Redis** | Fast in-memory key-value store for labels/entity existence checks | Active |

---

## Project Structure

```
CIExMAS/
├── approaches/                          # Multi-agent architectures
│   ├── Network/                         # Network Architecture (Paper)
│   ├── ReAct/                           # ReAct Architecture (Paper)
│   ├── Supervisor/                      # Supervisor Architecture (Paper)
│   ├── additional_architectures/        # Additional explored architectures
│   └── evaluation_logs/                 # Per-run evaluation logs (xlsx)
├── datasets/                            # HuggingFace dataset loader (Wiki-cIE)
├── helper_tools/                        # Evaluation, parsing, infrastructure tools
├── infrastructure/                      # Docker Compose, SLURM setup, Fuseki config
└── results/                             # Final evaluation results and reports
    ├── A_Initial_Supervisor/            # Initial Supervisor snapshot
    ├── result_evaluation_logs/          # Final evaluation Excel files (A-D, Z benchmarks)
    └── result_reports/                  # Generated CSV reports per metric
```

### General Files per Approach

Each approach follows a consistent structure:

```
<approach>/
├── agents/                    # Agent and tool implementations
├── slurm/
│   ├── agent_system.py        # LangGraph composition + evaluation runner
│   └── execution_script.sh    # SLURM execution script
├── prompts.py                 # All LangChain prompt templates
└── setup.py                   # Agent state definition + base setup import
```

---

## Datasets & Evaluation

### Supported Datasets

The primary dataset is **Wiki-cIE** (referred to as SynthIE in the original dataset publication). In code, the dataset IDs are:

| Dataset | ID | Description |
|---|---|---|
| **Wiki-cIE Text** | `wiki_cie_text` | text_davinci_003 generated text (validation & test splits) |
| **Wiki-cIE Code** | `wiki_cie_code` | code_davinci_002 generated text (train, validation & test splits) |
| **REBEL** | `rebel` | Distantly supervised relation extraction dataset |
| **REDFM** | `redfm` | Refined relation extraction dataset |

### Run Evaluation

```bash
python ./approaches/<ARCHITECTURE>/slurm/agent_system.py \
  --dataset <DATASET_ID> \
  --split <SPLIT> \
  --num_samples <N> \
  --description <DESCRIPTION>
```

### Performance Metric

Following Josifoski et al. (2023), a triple is regarded as correct if the subject, predicate, and object URIs all match the gold standard. We report macro-averaged Precision, Recall, and F1 scores. Additionally, we compute hierarchical property scores (Parental F1, Related F1) using the Wikidata class hierarchy.

---

## Helper Tools

### Agent Builder UI

Interactive Streamlit interface for managing and evaluating approaches:

- **Comparison** - Compare evaluation logs side by side
- **Inspect Eval Log** - Review evaluation logs in detail with Langfuse trace IDs
- **Log Notes** - Inspect and modify approach labels and descriptions
- **Step By Step Evaluation** - Test agents incrementally

### Core Tools

| Tool | Description |
|---|---|
| `base_setup.py` | Initializes LLM, Vector Store, Triple Store, and Tracing |
| `evaluation.py` | Evaluation pipeline and report generation |
| `set_llm_config.py` | Sets preconfigured LLM IDs, providers and rate limits in `.env` |
| `parser.py` | Dataset parsing and vector store embedding |
| `validation.py` | Triple validation methods |
| `sort_jsonl.py` | Dataset sorting for consistent sample subsets |

---

## Model Providers

### Cerebras

| Model | ID |
|---|---|
| Llama 3.3 70B | `llama-3.3-70b` |
| Llama 4 Scout | `llama-4-scout-17b-16e-instruct` |

### SambaNova

| Model | ID |
|---|---|
| Llama 3.3 70B | `Meta-Llama-3.3-70B-Instruct` |
| Llama 4 Maverick | `Llama-4-Maverick-17B-128E-Instruct` |

### DeepInfra

| Model | ID |
|---|---|
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` |

### vLLM (Self-Hosted)

| Model | ID | Quantization |
|---|---|---|
| Llama 3.3 70B | `kosbu/Llama-3.3-70B-Instruct-AWQ` | AWQ |
