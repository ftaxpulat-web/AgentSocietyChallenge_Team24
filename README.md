# AgentSocietyChallenge Team 24 Setup & Usage Guide

Link to the original repository: <https://github.com/tsinghua-fib-lab/AgentSocietyChallenge>

---

## Prerequisites

- Python **3.10+**
- **Google Cloud / Gemini API Key** (required for the LLM)
- **High-RAM environment**  
  - Recommended: **32GB+ RAM** if processing the full dataset

Make sure requirements are installed through `pip`, and that an environment file is set up with your Gemini API key named `GEMINI_API_KEY` within.

```bash
pip install -r requirements.txt
pip install google-genai langchain-chroma chromadb tqdm pandas
```

---

## Dataset Preparation

Download the datasets from the AgentSocietyChallenge GitHub onto your local machine (or a virtual machine — for virtual machine setup, refer to various guides online for setting up a project on Google Cloud Platform).

Run the data processing script:

```bash
python3 data_process.py --input_dir ./raw_data --output_dir ./dataset
```

Then, run the `data_preprocess.py` file from the **AgentSocietyChallenge** repository, making sure the output files are in a folder called `dataset/` in the root directory of the project.

Also, ensure that you have ground-truth and task JSON files in folders named:

- `example/ground_truth`
- `example/tasks`

These should already be in the base version of the project. Results will be stored accordingly by the experiment scripts.

---

## Running Agent 1 (Simulation)

The simulation agent additionally requires building a global RAG database in the project. Run:

```bash
python3 build_knowledge.py
```

To run the actual tests and see the results of the model, use:

### 50 full tasks **with** RAG memory

```bash
python3 run_experiment.py   --exp_name hybrid_rag_final   --memory hybrid_rag   --reasoning cot   --tasks 50
```

### 50 full tasks **without** memory

```bash
python3 run_experiment.py   --exp_name baseline_no_memory   --memory none   --reasoning cot   --tasks 50
```

Also feel free to try different reasoning methods as well by changing the `--reasoning` argument.

---

## Running Agent 2 (Recommendation)

To run the recommendation agent and see the results of the model, run:

```bash
python3 run_rec_agent.py --task 50
```
