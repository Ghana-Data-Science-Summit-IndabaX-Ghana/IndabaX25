# LLMs for Text Generation & Summarization

This repository contains materials for a comprehensive workshop on leveraging Large Language Models (LLMs) for text generation and summarization tasks.

---

## Overview

This workshop explores modern approaches to text summarization using transformer-based architectures, covering both extractive and abstractive techniques. Participants will gain practical experience implementing multi-stage summarization pipelines, controlling generation parameters, and evaluating summary quality using various metrics.

---

## Learning Objectives

By the end of this workshop, you will be able to:

- **Compare multiple summarization approaches:** Understand and apply extractive, abstractive, and hybrid techniques for text summarization.
- **Control summary generation:** Adjust summary length, style, and focus using advanced control parameters (temperature, top-k, top-p, etc.).
- **Evaluate summaries comprehensively:** Go beyond ROUGE by assessing factual consistency, semantic similarity, and human-centric metrics.
- **Build multi-stage summarization pipelines:** Combine different techniques to create more effective summarization systems.
- **Handle complex and multimodal documents:** Summarize long, multi-document, and multimodal (text + images) content.
- **Engineer effective prompts:** Use prompt engineering frameworks (like CO-STAR) for structured and controlled output.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Tutorial Outline](#tutorial-outline)
5. [Key Features](#key-features)
6. [Facilitators](#facilitators)
7. [Useful Resources](#useful-resources)

---

## Prerequisites

Before attending the workshop, participants should:

- Have basic knowledge of Python programming
- Be familiar with fundamental NLP concepts
- Complete the pre-reading materials ([PRE-TUTORIAL.md](PRE-TUTORIAL.md))

---

## Installation

**Assumptions & Requirements:**

- **Operating System:** Linux, macOS, or Windows (with WSL recommended for Windows users)
- **Python:** 3.8 or higher ([Download Python](https://www.python.org/downloads/))
- **pip:** Python's package manager (included by default)
- **venv:** For environment isolation (included with Python 3.8+)
- **git:** For cloning the repository ([Download Git](https://git-scm.com/downloads))

**Setup Steps:**

```bash
# Clone the repository
git clone https://github.com/Ghana-Data-Science-Summit-IndabaX-Ghana/IndabaX25.git
cd "IndabaX25/LLMs for Text Generation & Summarization"

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with:
# OPENROUTER_API_KEY=your_api_key_here
```

---

## Project Structure

```
├── LLMs for Text Generation & Summarization.ipynb  # Main tutorial notebook
├── .gitignore                                      # Git ignore file
├── requirements.txt                                # Python dependencies
├── PRE-TUTORIAL.md                                 # Pre-reading materials
├── articles/                                        # Contains articles
└── images/                                         # Images for visualization
```

---

## Tutorial Outline

The workshop is divided into three main parts:

### Part I: Architectural Overview and Extractive Summarization
- Understanding transformer architecture
- Categories of LLMs (Decoder-only, Encoder-only, Encoder-Decoder)
- Implementing extractive summarization
- Establishing evaluation metrics (ROUGE, BLEU)

### Part II: Abstractive Summarization & Control Parameters
- Implementing BART and T5 models for abstractive summarization
- Comparing with autoregressive models
- Controlling summary generation (length, style, focus)
- Building multi-stage summarization pipelines

### Part III: Advanced Techniques
- Multimodal summarization (text + images)
- Advanced evaluation beyond ROUGE
- Practical exercises for building custom systems

---

## Key Features

- **Multiple Summarization Approaches**: Compare extractive, abstractive, and hybrid techniques
- **Advanced Control Parameters**: Learn to control summary length, style, and focus
- **Comprehensive Evaluation**: Go beyond ROUGE with factual consistency and semantic similarity
- **Multi-Stage Pipelines**: Combine techniques for more effective summarization

---

## Facilitators

**Nana Sam Yeboah**  
Email: nanayeb34@gmail.com  
LinkedIn: [Nana Sam Yeboah](https://www.linkedin.com/in/nana-sam-yeboah-0b664484)

**Audrey Eyram Agbeve**  
Email: audreyagbeve02@gmail.com  
LinkedIn: [Audrey (Eyram) Agbeve](https://www.linkedin.com/in/audreyagbeve02/)

---

## Useful Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CO-STAR Prompt Engineering Framework](https://github.com/co-star-framework/co-star)