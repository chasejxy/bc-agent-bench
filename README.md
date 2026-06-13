# BC-Agent-Bench

This repository contains the main code and datasets of **BC-Agent-Bench**, a dynamic 
benchmark for Microsoft Dynamics 365 Business Central Agents, with Sales Order Agent as a case study.

## Project Structure

- **`space/`** – Definitions of the benchmark space: scenarios, tones, and personas.
- **`datasets/`** – The generated benchmark set used for evaluation.
- **Python files** – Core components of the dual-agent simulation and evaluation pipeline:
  - `customer_agent.py` – C2 agent (customer).
  - `reviewer_agent.py` – C1 agent (user/reviewer).
  - `evaluator.py` – LLM-based evaluator (LLM judge).
  - `orchestrator.py` – Orchestrates the interaction and evaluation flow.

## Note on Reproducibility

This codebase is provided for review of the methodology and 
implementation. It depends on an internal Microsoft enterprise environment 
(Business Central, Sales Order Agent) and **can only be run 
within that environment**. 

## Confidentiality

- `AgentTaskClient` (internal Business Central API) is omitted.
- Azure OpenAI endpoint / model version / credentials removed.
