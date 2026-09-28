# LangChain Semantic Router

A local-first semantic routing engine built with LangChain and Ollama. I developed this project as part of my ongoing studies to explore how to dynamically route user queries to highly specialized LLM prompts based on intent classification. 

## Features

* **Local LLM Integration:** Powered entirely locally via `OllamaLLM` using the `mistral-nemo:12b` model.
* **Semantic Classification:** Uses a dedicated router chain to strictly categorize inputs into predefined intents (`algorithm`, `design`, or `general`).
* **Dynamic Branching:** Utilizes LangChain's `RunnableBranch` to seamlessly divert traffic to the appropriate specialized chain without hardcoded conditional logic.
* **Role-Playing System Prompts:** Features specific personas, including a strict computer science professor and a precise UI/UX designer, reflecting distinct focus areas of my coursework and personal projects.

## Architecture

The pipeline follows a two-step LCEL (LangChain Expression Language) workflow:
1. **The Classifier:** Receives the raw query and forces the LLM to output a single keyword token.
2. **The Master Assembly Chain:** Injects the classification into a `RunnableBranch` alongside the original query, directing it to the correct specialized prompt.

## Environment & Hardware Context

This project was built and executed entirely within PyCharm. Local model execution was optimized for Apple Silicon, specifically developed and tested on a MacBook Pro M5 with 24GB of RAM. Running a 12-billion parameter model like `mistral-nemo` locally requires significant unified memory, and this hardware configuration provides the necessary overhead for smooth, low-latency prompt routing.

## Prerequisites

* Python 3.9+
* PyCharm IDE (recommended for maintaining this specific project structure)
* Ollama installed locally
* The Mistral Nemo model pulled via Ollama:
  ```bash
  ollama run mistral-nemo:12b
