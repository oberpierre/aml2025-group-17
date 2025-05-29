We want to address the problem of Named Entity Recognition (NER) in a (near-)real-time setting, where the input sequence $\mathbf{X}_{0:t} = \langle x_0, x_1, \dots, x_t \rangle$ arrives incrementally over time. At each time step $t$, only the first $t + 1$ tokens are available. Our goal is to identify named entities (e.g., persons, locations) from this partial input as early as possible, rather than waiting for the full sequence to arrive.

This setting simulates real-world scenarios such as transcribed emergency calls, where timely identification of critical information — like the caller's name, location, or the nature of the incident — can greatly assist the operator. In this context, latency and computation budget are as important as accuracy.

We propose two *non-RL* selective‐inference strategies. At each step $t$, given the prefix $\mathbf{X}_{0:t}$, the system decides whether to:

* **WAIT** — continue accumulating input; or  
* **RUN_NER** — invoke the NER module and extract entities from $\mathbf{X}_{0:t}$.

---

## Dataset & Data Splitting

* **Dataset:** OntoNotes 5.0 (english_v12 portion).  
* ⁠We are using the dataset as provided by hugging face, which already incorporates a **Document-level split** of:
  * **Training:** 80%
  * **Validation:** 10%  
  * **Test:** 10%  
* **Streaming simulation:** within each split, reveal tokens one by one; any decision at time $t$ uses only $\mathbf{X}_{0:t}$.
* ⁠**Hyperparameter tuning** (e.g. confidence threshold $\tau$, $\varepsilon$) is performed on *Validation*; final results reported on *Test*.

---

## Non-RL Alternatives

### 1. Confidence-Thresholding  
* Train a lightweight *scorer* $s(\mathbf{X}_{0:t})$ (e.g. a small feed-forward head over current token embeddings) that predicts the model’s confidence in its tags.  
* *Rule:* if $s(\mathbf{X}_{0:t}) < \tau$, **RUN_NER**; otherwise **WAIT**.  
* *Hyperparameter:* confidence threshold $\tau$ selected on Validation to balance early invocation against tagging accuracy.

### 2. Supervised “When-to-Stop” Classifier  
* *Label construction:* for each prefix $\mathbf{X}_{0:t}$ in the training set, compute its NER F1 versus the full-sequence F1; assign label *1* if within $\varepsilon$, else *0*.  
* ⁠Train a binary classifier $f(\mathbf{X}_{0:t})\in\{0,1\}$ to predict “good time to invoke NER.”  
* *Inference:* at time $t$, if $f(\mathbf{X}_{0:t})=1$, **RUN_NER**; otherwise **WAIT**.  
* ⁠*Hyperparameter:* $\varepsilon$ (and any classifier decision threshold) tuned on Validation.

---

### Formal Objective

Let $y_t$ be the ground-truth NER tags and $\hat{y}_t$ the predicted tags when NER is invoked at time $t$. Define $\mathcal{I}_f\subseteq\{0,1,\dots,T\}$ be the set of invocation times determined by the invocation function $f$ (e.g., confidence thresholding or a binary classifier). We seek to minimize

$$
\mathcal{L}(f) = \sum_{t \in \mathcal{I}_f} \ell_{\mathrm{NER}}(\hat{y}_t, y_t) + \lambda\, C(\mathcal{I}_f),
$$

where  
* $\ell_{\mathrm{NER}}$ is the token-level cross-entropy loss,  
* $⁠C(\mathcal{I})$ penalizes late or frequent invocations (e.g. number of calls, cumulative delay),  
* $⁠\lambda\ge0$ balances accuracy vs. efficiency.

---

## Evaluation

### 1. NER Accuracy  
* *FPR @ FNR ≤ ε:* false positive rate under varying false negative rate thresholds (e.g., FNR ≤ 1% and 0.1%)
* *Token-level precision, recall*  
* *Micro and macro averages*, across entity types

### 2. Timeliness (Latency-Aware Metrics)  
* *Time-To-First-Detection (TTFD):* average number of tokens between entity mention start and its first correct prediction  
* *Coverage @ $t$:* percentage of entities correctly predicted by time $t$

### 3. Computational Efficiency  
* ⁠*NER invocations per sequence*  
* ⁠*Total runtime* or *FLOPs* per document  
* *Inference overhead* relative to an always-on baseline

### 4. Baselines for Comparison  
* *Always-on:* run NER at every token  
* ⁠*Post-hoc:* run NER only at end of sequence $t=T$
* *Fixed-interval:* run every $k$
* ⁠*Oracle:* best possible timing (upper bound)

All methods are evaluated on the *test* split of OntoNotes 5.0, simulating streaming by revealing input token-by-token and measuring how early and accurately each system extracts entities.
