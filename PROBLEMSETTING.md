# Real-Time Named Entity Recognition with Selective Inference

This project addresses **Named Entity Recognition (NER)** in a **streaming / real-time** setting. The input sequence $\mathbf{X}_{0:t} = \langle x_0, x_1, \dots, x_t \rangle$
arrives token by token. At each time step $t$, the system only sees the partial input and must decide:

- Should we **invoke NER now**?
- Or should we **wait** for more context?

This simulates real-world applications like:
- **Emergency call transcription**
- **Voice assistant pipelines**
- **Low-latency document analysis**

In these scenarios, identifying entities (like locations, names) **early** can be as critical as being accurate — with constraints on **latency** and **computation**.

---

## Dataset & Streaming Setup

- **Dataset:** [OntoNotes 5.0](https://huggingface.co/datasets/conll2012_ontonotesv5) (`english_v12` portion via Hugging Face)
- **Splits:**
  - `Train`: 80%
  - `Validation`: 10%
  - `Test`: 10%
- **Streaming Simulation:**
  - Input revealed **token by token**
  - At each time step $t$, models only see prefix $\mathbf{X}_{0:t}$.
- **Validation used for hyperparameter tuning**, **test split** for final evaluation.

---

## Selective Inference Approaches (Non-RL)

### 1. Confidence-Based Invocation

- Train a **lightweight binary classifier** over BERT [CLS] token embeddings.
- Task: predict whether a complete entity **ends** at current token.
- At inference:
  - If confidence > threshold $\tau$: **RUN NER**
  - Else: **WAIT**
- Labels generated from OntoNotes BIO tagging scheme.
- Tunable threshold $\tau$ selected on validation set.

---

### 2. Window-Based Trigger Classifier

- Slide a **fixed-length window** (e.g. 6 tokens) over the stream.
- Label windows as `1` if they contain **at least one complete entity**, `0` otherwise.
- Embedding types explored:
  - **Static Word Embeddings:** Word2Vec (via `staticvectors`)
    - Average Pooling
    - Max Pooling
  - **Contextual Embeddings:** BERT CLS token
- Train binary MLP classifier on these embeddings to predict whether NER should be invoked on a window.

---

## Training Pipeline

- Extract streaming prefixes or sliding windows from OntoNotes with entity-end or entity-span labels.
- Generate embeddings:
  - **Word2Vec**: fast, local context
  - **BERT**: contextual, slower
- Train classifiers using PyTorch (`nn.Linear`-based MLPs), with:
  - **Weighted loss** for imbalanced labels
  - **Validation-based threshold tuning**

---

## Evaluation Metrics

### 1. Entity Detection Quality

| Metric                         | Description                                                |
|-------------------------------|------------------------------------------------------------|
| **Time-to-First Detection (TTFD)** | Avg. tokens until an entity is correctly predicted         |
| **Entity Recall @ t**         | Percent of entities detected by time step $t$         |
| **Entity Completion Accuracy**| Accuracy of classifier in detecting entity ends/spans     |

---

### 2. Computational Efficiency

| Metric                       | Description                                               |
|-----------------------------|-----------------------------------------------------------|
| **BERT Calls (Classifier vs Always-On)** | Number of actual NER calls saved |
| **Reduction %**             | Relative savings in BERT invocations                     |
| **Inference Overhead**      | Time/cost of running the classifier compared to full NER |

---

### 3. Classifier Performance

| Metric      | Description                     |
|-------------|---------------------------------|
| **Precision** | Positive prediction quality     |
| **Recall**    | True entity detection coverage  |
| **F1 Score**  | Harmonic mean of P/R            |

---

## Baselines for Comparison

| Baseline         | Description                                                   |
|------------------|---------------------------------------------------------------|
| **Always-on**     | Run NER at every time step (upper-bound on accuracy + cost)   |
| **Post-hoc**      | Run NER only at the final token (minimal compute, high delay) |
| **Fixed-interval**| Run NER every $k$ tokens regardless of context            |
| **Oracle**        | Best possible timing for NER invocation (upper-bound)         |

---

## In short

This project explores **efficient NER under streaming constraints**, aiming to invoke NER *only when needed*. By using lightweight classifiers trained on partial input or windows, we:
- **Reduce unnecessary NER calls**
- **Maintain high entity detection quality**
- **Enable low-latency real-time applications**


