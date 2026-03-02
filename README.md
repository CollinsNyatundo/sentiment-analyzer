# DSPy Sentiment Analyzer & Agentic LLM System

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![DSPy](https://img.shields.io/badge/DSPy-AI-blueviolet)
![MLflow](https://img.shields.io/badge/MLflow-%E2%89%A53.4.0-orange?logo=mlflow)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?logo=openai)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A two-part AI engineering project demonstrating **structured LLM programming** with [DSPy](https://github.com/stanfordnlp/dspy), **experiment tracking** with MLflow, and **agentic multi-turn dialogue** — all powered by OpenAI GPT-4o-mini.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Results](#results)
- [Author](#author)

---

## Overview

This project is composed of two interconnected components:

### 1. Sentiment Classifier
A structured sentiment analysis pipeline using DSPy `Signatures` and `ChainOfThought` reasoning. Text inputs are classified on a **0–10 sentiment scale** using typed Pydantic-constrained outputs, with support for both the default DSPy adapter and a `JSONAdapter` for structured JSON responses.

### 2. Celebrity Guessing Agent (`CelebrityGuess`)
An interactive, multi-turn agentic system that uses **chained DSPy modules** to iteratively guess a celebrity through yes/no questions. The agent:
- Generates targeted binary questions using a `QuestionGenerator` signature
- Tracks conversation history (past questions and answers) across turns
- Produces a post-game `Reflection` using Chain-of-Thought reasoning
- Logs every interaction, metric, and artifact to **MLflow** for full experiment reproducibility

---

## Project Structure

```
sentiment-analyzer/
├── sentiment-analysis.ipynb       # Main notebook: sentiment classifier + celebrity agent
├── dspy_program/                  # Saved DSPy program state (JSON + full program)
├── mlartifacts/                   # MLflow artifact store
├── mlruns/                        # MLflow experiment run metadata
├── celebrity_guess_results.json   # Sample run output (JSON artifact)
├── celebrity_guess_conversation.txt # Sample run conversation log
├── requirements.txt               # Python dependencies
└── .gitignore
```

---

## Features

- **DSPy Signatures** with typed input/output fields and Pydantic constraints (`ge`, `le`)
- **Chain-of-Thought** reasoning for both classification and reflection tasks
- **Multi-turn agent** with stateful context (past Q&A memory)
- **Dual LLM adapter support**: default DSPy adapter and `JSONAdapter`
- **Prompt inspection** via `dspy.inspect_history()` for full transparency
- **MLflow autologging** with `mlflow.dspy.autolog()` for automatic trace capture
- **Custom MLflow metrics**: execution time, efficiency ratio, questions per attempt, success rate
- **Artifact logging**: JSON results and human-readable conversation logs per run
- **DSPy program persistence**: save and reload program state with `save()` / `dspy.load()`
- **LiteLLM** as the underlying LLM routing layer for provider flexibility

---

## Tech Stack

| Tool | Role |
|------|------|
| **DSPy** | LLM programming framework (Signatures, Modules, ChainOfThought) |
| **OpenAI GPT-4o-mini / GPT-4o** | Language model backend |
| **MLflow ≥ 3.4.0** | Experiment tracking, artifact logging, run management |
| **LiteLLM** | LLM provider abstraction layer |
| **Python 3.13** | Runtime |
| **python-dotenv** | Secure API key management via `.env` |

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/CollinsNyatundo/sentiment-analyzer.git
cd sentiment-analyzer

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Start MLflow Tracking Server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) to view experiment runs.

---

## Usage

Open and run `sentiment-analysis.ipynb` in Jupyter or VS Code.

The notebook is structured in the following sequence:

1. **Imports & Config** — Load environment variables, configure MLflow and DSPy LM
2. **Sentiment Classifier** — Build and test `SentimentClassifier` signature with `dspy.Predict`
3. **Chain-of-Thought** — Add reasoning to predictions with `dspy.ChainOfThought`
4. **JSON Adapter** — Switch to structured JSON output format
5. **Celebrity Guessing Agent** — Run the interactive multi-turn `CelebrityGuess` module
6. **Save & Load** — Persist and reload DSPy program state

### Sentiment Classification Example

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
dspy.settings.configure(lm=lm)

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a text."""
    text: str = dspy.InputField(desc="input text to classify sentiment")
    sentiment: int = dspy.OutputField(desc="sentiment score", ge=0, le=10)

predict = dspy.Predict(SentimentClassifier)
output = predict(text="I am feeling pretty happy about this!")
print(output.sentiment)  # Output: 8
```

### Running the Celebrity Guessing Agent

```python
agent = CelebrityGuess(max_tries=20)
result = agent.forward()
# Think of a celebrity, type their name, and answer yes/no questions
# All interactions are automatically tracked in MLflow
```

---

## MLflow Experiment Tracking

All runs are tracked under the experiment **`DSPy Sentiment Analysis`**.

Each `CelebrityGuess` run logs:

| Category | Details |
|----------|---------|
| **Parameters** | Target celebrity, max tries, question text per attempt, guess flags |
| **Metrics** | `execution_time_seconds`, `actual_attempts`, `efficiency_ratio`, `success`, `questions_per_attempt` |
| **Artifacts** | `celebrity_guess_results.json`, `celebrity_guess_conversation.txt` |
| **Tags** | `execution_status`, `outcome`, `celebrity_category`, `attempts_category` |

DSPy traces are captured automatically via `mlflow.dspy.autolog()`.

---

## Results

### Sentiment Classification

| Input Text | Sentiment Score |
|---|---|
| "I am feeling pretty happy about this!" | 8/10 |
| "This is terrible and I hate it." | 0/10 |
| "I feel neutral about this situation." | 5/10 |
| "I'm absolutely thrilled with the results!" | 10/10 |
| "This makes me so angry and frustrated." | 1/10 |

### Celebrity Guessing Agent — Sample Run

```
Target Celebrity : LeBron James
Outcome          : SUCCESS
Attempts Used    : 7 / 20
Execution Time   : 29.56 seconds
Efficiency Ratio : 0.35
```

**Question trace:**
1. Is the celebrity an actor? → No
2. Is the celebrity a musician? → No
3. Is the celebrity an athlete? → Yes
4. Is the athlete known for playing basketball? → Yes
5. Is the athlete currently active in the NBA? → Yes
6. Does the athlete play for the Los Angeles Lakers? → Yes
7. Is the celebrity LeBron James? → Yes ✓

**Agent Reflection:**
> This guessing process highlighted the importance of asking targeted questions that progressively narrow down the options. Focusing on specific attributes such as sport and team significantly streamlined the guessing process.

---

## Author

**Collins Nyagaka**
Data Scientist & ML Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-collins--nyagaka-blueviolet)](https://collins-nyagaka-portfolio.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-CollinsNyatundo-black?logo=github)](https://github.com/CollinsNyatundo)

---

*Built with DSPy, MLflow, and OpenAI — October 2025*
