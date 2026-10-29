# System Architecture & Technical Specifications

## 1. System Overview

The **Prompt Injection Prevention & Detection Framework** establishes a robust, defense-in-depth security gateway positioned directly between untrusted user inputs and a local Large Language Model (**Meta Llama 3.2:3B** running on **Ollama**).

### High-Level Architectural Diagram

```
                              ┌────────────────────────┐
                              │  Untrusted User Prompt │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  Input Preprocessing   │
                              │  - Unicode Normalizer  │
                              │  - Obfuscation Decoder │
                              │  - Delimiter Scanner   │
                              └───────────┬────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      ┌─────────────────────────┐                     ┌─────────────────────────┐
      │  Layer 1: Rule Detector │                     │  Layer 2: ML Classifier │
      │  - Regex & Heuristics   │                     │  - TF-IDF Word N-Grams  │
      │  - Category Matchers    │                     │  - Calibrated Logistic  │
      │  - False-Positive Guard │                     │    Regression Model     │
      └───────────┬─────────────┘                     └───────────┬─────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Layer 3: Llama Guard   │
                              │ - Isolated Evaluation  │
                              │ - Zero-Shot Classifier │
                              │ - Structured JSON Out  │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Suspicion Score        │
                              │ Aggregator & Decision  │
                              │ Score = Σ (w_i * S_i)  │
                              └───────────┬────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            [Score > 0.70: MALICIOUS]                  [Score < 0.35: SAFE]
                     │                                         │
                     ▼                                         ▼
      ┌───────────────────────────────┐        ┌───────────────────────────────┐
      │     PREVENTION BLOCKER        │        │   OLLAMA LLAMA 3.2 GENERATOR  │
      │  - Hard Security Boundary     │        │  - System Prompt Isolation    │
      │  - Zero Tokens Reach LLM      │        │  - Generates Natural Response │
      │  - Explainable User Alert     │        └───────────────┬───────────────┘
      └──────────────┬────────────────┘                        │
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  SQLite Audit Logging  │
                              │  - Requests & Latency  │
                              │  - Detections & Scores │
                              │  - Actions & Responses │
                              └────────────────────────┘
```

---

## 2. Core Security Layers

### 2.1 Preprocessing Layer (`preprocessing/processor.py`)
- **Unicode NFKC Normalization:** Neutralizes homoglyph substitution attacks (e.g. Cyrillic or fullwidth Unicode lookalike characters).
- **Whitespace Normalization:** Collapses inter-character spacing tactics (e.g. `i g n o r e`).
- **Obfuscation De-anonymizer:** Decodes Base64, Hexadecimal, and URL-encoded segments to evaluate the concealed payload.
- **Delimiter Injection Scanner:** Detects boundary token injection (e.g. `<|im_start|>`, `### System:`, `[INST]`).

### 2.2 Rule-Based Detection Layer (`detection/rule_detector.py`)
- **Taxonomy Categories:**
  1. `instruction_override`: Tokens attempting to supersede system rules.
  2. `system_prompt_extraction`: Tokens attempting to dump hidden developer prompts.
  3. `role_manipulation`: Jailbreak personas (e.g. DAN, Developer Mode, Root Admin).
  4. `security_bypass`: Commands requesting moderation deactivation.
  5. `hierarchy_manipulation`: Claims of superior instruction authority.
- **False-Positive Mitigation:** Recognizes inquiry verbs and academic framing (e.g., *"What is prompt injection?"*), dampening penalties to avoid blocking benign educational questions.

### 2.3 Machine Learning Layer (`detection/ml_detector.py`)
- **Feature Extraction:** Sublinear TF-IDF representation combining unigrams, bigrams, trigrams (1-3) and character n-grams.
- **Model:** Calibrated Logistic Regression trained specifically on direct prompt injection vectors.
- **Generalization:** Catches paraphrased attacks and synonyms that escape exact regular expressions.

### 2.4 Semantic Llama Evaluator (`detection/llama_detector.py`)
- **Prompt Isolation Envelope:** Encloses the untrusted text inside `<untrusted_user_input>` delimiters, preventing the evaluating model from executing the payload.
- **Strict Output Schema:** Instructs Ollama to output valid JSON with explicit `risk_score` (0.0–1.0) and `attack_category`.

### 2.5 Suspicion Score Aggregation (`detection/scorer.py`)
- Weighted linear formula:
  $$\text{Final Risk Score} = w_{\text{rule}} \cdot S_{\text{rule}} + w_{\text{ml}} \cdot S_{\text{ml}} + w_{\text{llm}} \cdot S_{\text{llm}}$$
  where $w_{\text{rule}} = 0.35, w_{\text{ml}} = 0.35, w_{\text{llm}} = 0.30$.
- **3-Way Decision Thresholds:**
  - $\text{Score} < 0.35 \rightarrow \mathbf{SAFE}$
  - $0.35 \le \text{Score} \le 0.70 \rightarrow \mathbf{UNCERTAIN}$ (Subject to strict policy)
  - $\text{Score} > 0.70 \rightarrow \mathbf{MALICIOUS}$ (Blocked immediately)

### 2.6 Prevention Engine (`prevention/blocker.py`)
- **The Absolute Security Boundary:** If marked `MALICIOUS`, the request is halted immediately. **Zero prompt tokens are transmitted to the response generation LLM**.
- Generates an explainable security alert with attack category, risk score, and justification.

---

## 3. Database Schema (SQLite)

The local SQLite database (`chatbot_security.db`) enforces relational integrity across four tables:

```sql
-- 1. Incoming prompt requests
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    original_prompt TEXT NOT NULL,
    normalized_prompt TEXT NOT NULL,
    length INTEGER NOT NULL
);

-- 2. Multi-layer detection telemetry
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    rule_score REAL NOT NULL,
    ml_score REAL NOT NULL,
    llm_score REAL NOT NULL,
    final_risk_score REAL NOT NULL,
    classification TEXT NOT NULL,
    attack_category TEXT NOT NULL,
    reason TEXT,
    latency_ms REAL NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
);

-- 3. Security decisions and actions
CREATE TABLE actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    decision TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    reason TEXT,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
);

-- 4. Final model generation outputs
CREATE TABLE responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    response_text TEXT,
    generation_latency_ms REAL,
    total_latency_ms REAL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE
);
```
