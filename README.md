# BC-Agent-Bench

BC-Agent-Bench is a dynamic benchmarking framework for evaluating AI agents in Microsoft Dynamics 365 Business Central, using the Sales Order Agent as a case study.

The framework extends static agent evaluation into dynamic, multi-turn workflow evaluation. It provides a structured methodology for specifying benchmark cases, generating executable test cases, simulating workflow-relevant human roles, orchestrating dynamic interactions, and evaluating target-agent behavior using an LLM-as-a-Judge.

The framework was developed as part of my work on evaluating agentic AI systems in an enterprise environment.

## Motivation

Traditional agent evaluation often relies on a fixed set of prompts or predefined test cases. While useful for regression testing, static tests may not fully capture the variability of real-world user interactions.

BC-Agent-Bench introduces a dynamic evaluation approach where simulated human roles interact with the target agent across different workflow conditions:

- Business scenarios
- Customer personas
- Communication tones
- Multi-turn conversation paths

This allows the same underlying task to evolve differently depending on how the simulated user communicates and how the target agent responds.

The goal is to provide broader behavioral coverage and help identify failure patterns that may not appear in static test cases.

## Architecture

BC-Agent-Bench consists of four main components that define, generate, execute, and evaluate dynamic benchmark cases.

### 1. Benchmark Case Specification Space

The benchmark defines a structured specification space for constructing dynamic evaluation cases while keeping them grounded in real Business Central workflows.

The specification space separates three dimensions:

- Scenario – defines the business situation, customer goal, task constraints, and expected agent behavior.
- Customer Persona – defines behavioral characteristics of the simulated customer, such as cooperative or uncooperative behavior.
- Communication Tone – defines how the business request is expressed, including neutral, polite, informal, and rude or impatient communication.

By combining these dimensions, the benchmark introduces controlled variation into existing business tasks without changing the underlying task objective.

For example, the same sales-order scenario can be evaluated with different customer behaviors and communication styles, producing different interaction trajectories while preserving the same underlying business requirements.

### 2. Agent-Based Benchmark Case Generation

The benchmark uses an LLM-based generation process to transform abstract benchmark specifications into executable test cases.

For each benchmark case, a selected combination of scenario, customer persona, and communication tone is used to generate a concrete initial customer request.

In the Sales Order Agent case study, the Customer Agent generates the initial customer email based on these specifications.

This process allows benchmark cases to remain grounded in defined Business Central sales workflows while introducing controlled variation across cases.

Generated cases cover different workflow situations, including:

- Complete sales order requests
- Requests with incomplete information
- Requests involving unavailable items
- Unsupported requests
- Cases requiring clarification or follow-up

### 3. Dual-Agent Role Simulation and Orchestration

During benchmark execution, two LLM-based agents simulate the human roles involved in the workflow around the Sales Order Agent.

Customer Agent (C2)

The Customer Agent represents the external customer. It communicates with the Sales Order Agent through email and simulates customer-side behavior throughout the interaction.

Depending on the benchmark specification and current workflow state, the customer may:

- Provide an initial order request
- Supply missing information
- Respond to clarification questions
- Revise an existing request
- Provide follow-up information

User / Reviewer Agent (C1)

The User / Reviewer Agent represents the internal Business Central user involved in reviewing the Sales Order Agent's output and making workflow continuation decisions when human intervention is required.

For example, the reviewer may inspect generated sales orders or sales quotes and determine the appropriate next action based on the current workflow state.

Orchestrator

The orchestrator coordinates the interaction between the Sales Order Agent and the simulated human roles.

It manages:

- Conversation and workflow state
- Message routing between participants
- Multi-turn interaction
- Role activation
- Workflow progression
- Termination conditions
- Collection of complete interaction traces

Rather than following a fully pre-scripted conversation, benchmark execution evolves dynamically according to the target agent's behavior and the decisions of the simulated roles.

### 4. LLM-as-a-Judge Evaluation

After benchmark execution, an LLM-as-a-Judge evaluator analyzes the generated interaction trace and evaluates the behavior of the Sales Order Agent within the evolving workflow context.

The evaluation framework uses five diagnostic dimensions:

- Expected Behaviour Accuracy
- Item Correctness Score
- Document Type Accuracy
- Availability Factuality Score
- Trajectory Consistency Score

The evaluator uses evidence from the interaction trajectory, including Sales Order Agent responses, customer emails, and workflow-related outcomes.

Instead of evaluating only the final outcome, the framework performs turn-level evaluation and aggregates these judgments using Multi-Turn Accuracy (MTA). This allows failures occurring at different stages of a multi-turn interaction to be captured rather than treating the workflow as a single input-output task.

The resulting structured evaluation data can be aggregated across benchmark cases to analyze recurring failure patterns and compare agent behavior across different scenarios, customer personas, and communication tones.

## Evaluation Workflow

A typical benchmark run follows this process:

1. Select a scenario, customer persona, and communication tone from the benchmark specification space.
2. Generate an executable benchmark case based on the selected specification.
3. Initialize the simulated Customer Agent and User / Reviewer Agent.
4. The Customer Agent initiates the interaction with the Sales Order Agent through a generated customer request.
5. The interaction evolves dynamically as the Sales Order Agent processes the request, requests additional information, and produces workflow outputs.
6. The Customer Agent responds to clarification or follow-up requests when required, while the User / Reviewer Agent reviews generated sales documents and makes workflow continuation decisions when human intervention is needed.
7. The orchestrator manages the multi-turn execution and records the complete interaction trajectory.
8. The LLM-as-a-Judge evaluator evaluates the interaction using the defined measurement dimensions.
9. Turn-level evaluation results are aggregated using Multi-Turn Accuracy (MTA) for further analysis.

## Project Structure

```text
BC-Agent-Bench/
│
├── space/
│   ├── scenarios/
│   ├── personas/
│   └── tones/
│
├── datasets/
│   └── generated benchmark datasets
│
├── customer_agent.py
├── reviewer_agent.py
├── evaluator.py
├── orchestrator.py
└── README.md
```

### Core Components

customer_agent.py  
Implements the simulated customer (C2) responsible for generating customer requests and responses.

reviewer_agent.py  
Implements the simulated Business Central user (C1) responsible for reviewing Sales Order Agent outputs and making workflow continuation decisions when human intervention is required.

orchestrator.py  
Coordinates the multi-agent interaction, manages conversation state, and records execution traces.

evaluator.py  
Implements the LLM-as-a-Judge evaluation pipeline used to assess completed interactions.

space/  
Defines the configurable benchmark dimensions, including scenarios, personas, and communication tones.

datasets/  
Contains generated benchmark cases used during evaluation.

## Results

The framework was used to generate and evaluate dynamic multi-turn test cases for the Sales Order Agent across different business scenarios, customer personas, and communication tones.

Compared with static test cases, the dynamic benchmark introduced greater variation in interaction trajectories and enabled the evaluation of agent behavior across evolving workflow states. This helped identify failure patterns that may not be observable through isolated input-output testing.

The results demonstrate how role-based simulation and LLM-as-a-Judge evaluation can complement traditional deterministic testing when evaluating enterprise AI agents in dynamic workflows.

## Technologies

- Python
- Azure AI
- Microsoft Dynamics 365 Business Central
- Large Language Models (LLMs)
- Multi-Agent Systems
- LLM-as-a-Judge
- Agent Evaluation

## Reproducibility

This repository is intended to demonstrate the methodology, architecture, and implementation approach of the benchmark.

The original system integrates with internal Microsoft enterprise infrastructure, including Business Central and the Sales Order Agent. These dependencies are not publicly accessible, so the repository cannot be executed end-to-end outside the original environment.

The benchmark generation, orchestration, and evaluation logic included here illustrate the core design of the system.

## Confidentiality

This repository contains a generalized, public-safe version of the framework. No internal API details, credentials, customer data, or environment-specific production configuration are included.
