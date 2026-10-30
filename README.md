# 🛡️ Prompt Injection Prevention & Detection Framework for Local LLM Chatbot

A modular, multi-layered security framework and ChatGPT-style web interface designed to detect, analyze, and prevent direct prompt injection attacks targeting locally running Large Language Models (**Meta Llama 3.2:3B** powered by **Ollama**), with relational **SQLite** audit logging.

---

## 🌟 Key Features

* **ChatGPT-Style Streamlit Interface**: Real-time chat, session management, security badges, and collapsible diagnostic telemetry.
* **Multi-Layer Defense-in-Depth Detection**:
  1. *Preprocessing & Obfuscation Layer:* Unicode NFKC normalization, Base64/Hex/URL-encoding decoders, delimiter injection scanner.
  2. *Rule-Based Heuristic Detector:* Fast regex engine covering 5 injection taxonomies with false-positive mitigation for educational inquiries.
  3. *Machine Learning Classifier:* TF-IDF (word & char n-grams) + Calibrated Logistic Regression model trained on prompt injection vectors.
  4. *Semantic Llama Guard Evaluator:* Zero-shot security analysis with prompt isolation and structured JSON output via Ollama.
  5. *Suspicion Score Aggregator:* Weighted score fusion ($w_1 S_{\text{rule}} + w_2 S_{\text{ml}} + w_3 S_{\text{llm}}$) with 3-way decision thresholds (`SAFE`, `MALICIOUS`, `UNCERTAIN`).
* **Active Prevention Layer**: Intercepts and blocks malicious prompts before they reach the response generator (**Zero prompt tokens forwarded to LLM on block**).
* **SQLite Relational Audit Logging**: Persistent relational logging across `requests`, `detections`, `actions`, and `responses` tables with latency telemetry.
* **Live Security Sandbox**: Interactive tool to inspect how any prompt traverses each security stage.
* **Scientific Benchmark Suite**: Automated evaluation reporting Accuracy, Precision, Recall, F1-Score, FPR, FNR, and Confusion Matrix.

---

## 📁 Directory Structure

```
Secure-AI-Chatbot/
├── config.py                     # Global constants, thresholds, weights, model settings
├── database.py                   # SQLite schema, request logging, audit metrics
├── ollama_client.py              # Ollama REST API client (Generator & Detector)
├── app.py                        # Streamlit web application (Chat UI, Sandbox, Logs, Benchmark)
├── requirements.txt              # Project dependencies
├── dataset/
│   ├── generate_dataset.py       # Dataset generator script
│   ├── prompt_injection_dataset.json   # 220-sample benchmark dataset (Train/Val/Test)
│   └── prompt_injection_dataset.csv    # CSV dataset representation
├── preprocessing/
│   ├── __init__.py
│   └── processor.py              # Text normalization, unicode sanitization, obfuscation decoder
├── detection/
│   ├── __init__.py
│   ├── rule_detector.py          # Category-based Regex & heuristic detectors
│   ├── ml_detector.py            # TF-IDF + Logistic Regression inference model
│   ├── llama_detector.py         # LLM Semantic Analyzer with isolated instructions
│   ├── scorer.py                 # Risk score aggregator & 3-way decision engine
│   ├── train_model.py            # ML pipeline training and validation script
│   └── model/                    # Serialized ML model weights (.joblib)
├── prevention/
│   ├── __init__.py
│   ├── blocker.py                # Security gateway (enforces block boundary)
│   ├── policy.py                 # Policy engine for rate limits & input envelopes
│   └── warning.py                # Explainable security alert generator
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Benchmark evaluator (Accuracy, Precision, Recall, F1)
│   └── results.json              # Evaluated metrics storage
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py     # Unit tests for text normalization & encoding checks
│   ├── test_rules.py             # Unit tests for rule detector
│   ├── test_ml_detector.py       # Unit tests for ML pipeline
│   ├── test_prevention.py        # Unit tests verifying prompt blocking
│   ├── test_end_to_end.py        # Comprehensive test suite (9 test categories)
│   └── run_all_tests.py          # Standalone test runner script
├── docs/
│   ├── ARCHITECTURE.md           # Technical design & security architecture
│   ├── DFD.md                    # Data Flow Diagrams (Level 0, Level 1, Level 2)
└── README.md                     # Project documentation and run guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
* **Python 3.11.9** (or Python 3.10+)
* **Ollama** installed locally ([ollama.com](https://ollama.com))

### 2. Pull the Llama Model in Ollama
Open your terminal and pull Meta's Llama 3.2 3B model:
```bash
ollama pull llama3.2:3b
ollama serve
```

### 3. Install Dependencies
Navigate to the project directory and install requirements:
```bash
pip install -r requirements.txt
```

### 4. Train the ML Security Model
Train the Scikit-Learn baseline on the training split:
```bash
python detection/train_model.py
```

### 5. Run Automated Tests
Execute the comprehensive test suite verifying all 9 project test categories:
```bash
python tests/run_all_tests.py
```

### 6. Run Benchmark Evaluation
Run the scientific benchmarking script on the held-out test split:
```bash
python evaluation/evaluate.py
```

### 7. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔍 How Each Module Works

### 1. Preprocessing (`preprocessing/processor.py`)
Normalizes text with Unicode NFKC, checks for delimiter injections (e.g. `<|im_start|>`), collapses whitespace exploits (e.g. `i g n o r e`), and decodes Base64/Hex/URL hidden payloads into a consolidated representation for analysis.

### 2. Rule Detection (`detection/rule_detector.py`)
Scans against regular expression banks covering 5 major direct injection taxonomies (`instruction_override`, `system_prompt_extraction`, `role_manipulation`, `security_bypass`, `hierarchy_manipulation`). Contextually dampens scores for educational inquiries (e.g. *"What is prompt injection?"*).

### 3. ML Detection (`detection/ml_detector.py`)
Transforms sanitized text using sublinear TF-IDF word (1-3) and character n-grams, passing features to a calibrated Logistic Regression classifier to calculate probability of malicious intent.

### 4. Semantic Llama Evaluator (`detection/llama_detector.py`)
Wraps the untrusted prompt in an isolated `<untrusted_user_input>` envelope and queries the local Ollama instance in zero-shot classifier mode with strict JSON output formatting.

### 5. Suspicion Score Aggregation (`detection/scorer.py`)
Computes the weighted risk score:
$$\text{Final Risk Score} = (0.35 \times S_{\text{rule}}) + (0.35 \times S_{\text{ml}}) + (0.30 \times S_{\text{llm}})$$
Applies 3-way classification thresholds:
- $\text{Score} < 0.35 \rightarrow \mathbf{SAFE}$
- $0.35 \le \text{Score} \le 0.70 \rightarrow \mathbf{UNCERTAIN}$
- $\text{Score} > 0.70 \rightarrow \mathbf{MALICIOUS}$

### 6. Prevention Blocker (`prevention/blocker.py`)
Enforces the security boundary: If classified as `MALICIOUS`, the execution is halted immediately and an explainable warning is returned. **The prompt is never sent to the generation model**.

### 7. SQLite Database (`database.py`)
Maintains relational tables for requests, detections, actions, and model responses in `chatbot_security.db`.
