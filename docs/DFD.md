# Data Flow Diagrams (DFD)

This document provides formal **Level 0 (Context Level)**, **Level 1 (Top Level)**, and **Level 2 (Detailed Security Pipeline)** Data Flow Diagrams for the **Prompt Injection Prevention & Detection Framework**.

---

## 1. Level 0 DFD: Context Diagram

The Context Diagram defines the external entities (User and Local LLM Runtime) and the high-level boundary of the Prompt Injection Security System.

```mermaid
flowchart LR
    User([🧑 User / Client])
    System[[🛡️ Prompt Injection Prevention & Detection System]]
    Ollama([🤖 Ollama Local LLM Runtime])
    DB[(🗄️ SQLite Security Database)]

    User -->|1. Raw Prompt Query| System
    System -->|2. Safe Audited Prompt| Ollama
    Ollama -->|3. Generated AI Response| System
    System -->|4. Safe Response OR Security Block Warning| User
    System <-->|5. Audit Logging & Telemetry Queries| DB
```

---

## 2. Level 1 DFD: Top-Level Functional Flow

Level 1 breaks down the system into its primary subsystems: Preprocessing, Detection, Prevention Decision, LLM Generation, and Logging.

```mermaid
flowchart TD
    User([🧑 User])
    DB[(🗄️ SQLite Audit Database)]
    Ollama([🤖 Local Ollama / Llama 3.2])

    P1[1.0 Input Normalization & Preprocessing]
    P2[2.0 Multi-Layer Detection Engine]
    P3[3.0 Suspicion Score Aggregation]
    P4[4.0 Prevention & Policy Enforcement]
    P5[5.0 LLM Response Generation]
    P6[6.0 Relational Audit Logging]

    User -->|Raw Text| P1
    P1 -->|Cleaned Text + Obfuscation Flags| P2
    P1 -->|Log Request Event| P6
    
    P2 -->|Rule, ML & Semantic LLM Scores| P3
    P3 -->|Aggregated Risk Score & Category| P4
    P3 -->|Log Detection Metrics| P6

    P4 -->|If MALICIOUS: Intercept & Block| User
    P4 -->|Log Security Action| P6

    P4 -->|If SAFE: Forward Prompt| P5
    P5 -->|Query Llama 3.2| Ollama
    Ollama -->|AI Text Output| P5
    P5 -->|Safe Answer| User
    P5 -->|Log Generated Response| P6

    P6 <-->|Read / Write Relational Records| DB
```

---

## 3. Level 2 DFD: Deep-Dive Security Pipeline

Level 2 details the internal mechanics of the Multi-Layer Detection Engine (2.0) and Prevention Decision (4.0).

```mermaid
flowchart TD
    In[Normalized Input + Decoded Tokens]

    subgraph Detection Layer
        D1[2.1 Rule Detector: Category Matchers & Heuristics]
        D2[2.2 ML Classifier: TF-IDF + Logistic Regression]
        D3[2.3 Semantic Llama Detector: Isolated Prompt]
        
        In --> D1
        In --> D2
        In --> D3

        D1 -->|Rule Score [0.0 - 1.0]| Agg[2.4 Suspicion Score Aggregator]
        D2 -->|ML Probability [0.0 - 1.0]| Agg
        D3 -->|Llama Risk [0.0 - 1.0]| Agg
    end

    subgraph Decision & Prevention Layer
        Agg -->|Fused Score: Σ w_i * S_i| Dec{3.0 Risk Threshold Evaluator}
        
        Dec -->|Score > 0.70: MALICIOUS| Block[4.1 Security Blocker]
        Dec -->|0.35 <= Score <= 0.70: UNCERTAIN| Policy[4.2 Policy Enforcer]
        Dec -->|Score < 0.35: SAFE| Safe[4.3 Safe Dispatcher]

        Block -->|Generate Explainable Alert| OutBlock[Send Block Notice to User & Halt]
        Policy -->|Apply Strict Action| OutBlock
        Safe -->|Send to Generator| LLM[5.0 Ollama Llama 3.2 Generator]
    end
```
