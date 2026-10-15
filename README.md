# When is fine-tuning worth it? Text-to-SQL with Llama 3.2 3B (LoRA vs QLoRA)

A controlled study on a free Colab T4 (16 GB): how well can a small LLM turn a table schema plus an English question into correct SQL, and when does fine-tuning beat prompting?

**Headline results (test = 500 unseen-schema questions, out-of-distribution = 60 hand-written questions on a 4-table database)**

- Fine-tuning raised execution accuracy from **61%** (best prompting baseline) to **86 to 87%** (LoRA and QLoRA), p < 0.001.
- LoRA and QLoRA were **statistically indistinguishable** in quality. QLoRA used **31% less GPU memory** (5.1 vs 7.4 GB) and trained about **30% slower** (8.6 vs 6.6 min).
- Fine-tuning **hurt out-of-distribution accuracy**: from **80%** (few-shot base model) to about **44 to 45%**. More training data made it worse. The model largely learned the dataset's habits, not general SQL.

## 1. Problem and data

Task: given `CREATE TABLE ...` statements and a question, output one SQLite query. Data: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (78,577 rows, mostly single-table WikiSQL-style questions plus some Spider).

**Evaluation is execution-based.** The dataset has no table rows, so I generate a small SQLite database per example with random rows, seeded with the literals found in the gold query so filters actually match something. A prediction counts as correct if it returns the same rows as the gold query (column order ignored; row order only if the gold has `ORDER BY`). Reported metrics: exact match, valid-SQL rate, execution accuracy.

**Data audit** (what was removed and why):

| Step | Count |
|---|---|
| Raw rows | 78,577 |
| Exact duplicates | 0 |
| Rows with conflicting labels (same question and schema, different SQL) | 2 |
| Candidates whose gold SQL fails on its own schema (bad schema or error) | 148 of 7,400 sampled |
| Candidates whose gold returns no rows on the synthetic data | 524 of 7,400 sampled (dropped from val/test only) |
| Val/test examples sharing an exact schema with training | removed (24 val, 11 test) |
| Val/test examples near-duplicate of a training question (TF-IDF cosine at or above 0.9) | 0 val, 1 test |
| **Final splits** | **train 4,897, val 500, test 500** |

Val and test therefore measure **unseen schemas**, not memorized tables.

**Out-of-distribution set:** 60 hand-written questions on a 4-table shop database (customers, products, orders, order_items) with real rows. It needs multi-table JOINs, GROUP BY/HAVING, subqueries and NOT IN, which the training data rarely contains.

*Evaluation bug found and fixed during the project:* my first version of the data generator only seeded single-quoted values, but this dataset writes most values in double quotes. That wrongly dropped about 60% of candidates as "empty" and skewed the first test set toward easier queries. After the fix I re-scored all saved adapters (no retraining). Numbers below are the corrected ones; the first run had overstated the fine-tuning gain.

## 2. Setup

Llama 3.2 3B Instruct, fp16 (T4 has no bf16). LoRA: 16-bit base weights. QLoRA: 4-bit NF4 with double quantization. Both: rank 16, alpha 32, dropout 0.05, all linear layers (0.75% trainable, 24.3M parameters), learning rate 2e-4, 1 epoch, effective batch 16, paged 8-bit AdamW, loss on the SQL answer only. Same data, seeds and hyperparameters for both methods. Main comparison: 2,000 training examples, 3 seeds. Ablations and the learning curve: single seed.

## 3. Results

### Table 1: quality (test set; mean over 3 seeds for fine-tuned models)

| Model | Exact match | Valid SQL | Execution acc. | OOD exec. acc. (60 q) |
|---|---|---|---|---|
| Base, zero-shot | 0.468 | 0.842 | 0.422 (95% CI 0.378 to 0.464) | 0.683 |
| Base, 3-shot | 0.596 | 0.940 | 0.618 (0.576 to 0.660) | 0.817 |
| Base, 5-shot + strict system prompt | 0.598 | 0.936 | 0.610 (0.568 to 0.654) | 0.800 |
| **LoRA** | 0.799 | 0.999 | **0.869** +/- 0.009 (seed 0: 0.874, CI 0.844 to 0.902) | 0.450 |
| **QLoRA** | 0.792 | 0.998 | **0.855** +/- 0.009 (seed 0: 0.866, CI 0.834 to 0.894) | 0.439 |

CIs are bootstrap 95% intervals over test examples (seed 0). Paired McNemar exact tests (seed 0):
- Best prompting baseline vs QLoRA: 8 questions only the baseline got right, 136 only QLoRA, **p < 0.0001**.
- LoRA vs QLoRA: 11 vs 7, **p = 0.48**, no detectable difference.

### Table 2: resources (LoRA vs QLoRA)

| | Peak GPU memory | Train time (2,000 examples) | Trainable params |
|---|---|---|---|
| LoRA (16-bit) | 7.4 GB | 6.6 min | 24.3M (0.75%) |
| QLoRA (4-bit) | **5.1 GB** | 8.6 min | 24.3M (0.75%) |

Memory is `torch.cuda.max_memory_allocated` during training, measured in a clean run (500 examples, batch 4, short sequences). Peak memory scales with batch size and sequence length, so treat these as relative numbers for this setup. Training time is the mean over 3 seeds.

### Figure: how much data do you need? (QLoRA, seed 0)

| Training examples | 250 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|
| Test execution accuracy | 0.690 | 0.726 | 0.782 | 0.866 |
| OOD execution accuracy | 0.717 | 0.633 | 0.567 | 0.500 |

![learning curve](learning_curve.png)

The curve has **not plateaued** at 2,000 examples, so the answer to "how much data" is "more than 2,000 for this task". Meanwhile OOD accuracy falls steadily as in-distribution accuracy rises. Even 250 examples already scores below the few-shot base model on OOD (0.717 vs 0.80).

### Table 3: ablations (QLoRA, 1,000 training examples, validation set, seed 0)

| Rank | Attention layers only | All linear layers |
|---|---|---|
| 8 | 0.748 (0.14% trainable) | 0.808 (0.38%) |
| 16 | 0.796 (0.28%) | 0.822 (0.75%) |
| 64 | 0.858 (1.13%) | 0.870 (2.94%) |

Higher rank helped in this range, and adapting all linear layers helped most at low rank. With about 500 validation examples and one seed, differences under roughly 3 to 4 points are within noise. Note that the main experiments used rank 16, chosen before the ablation; the ablation suggests rank 64 would do better, which I did not re-run.

### Forgetting check (300 MMLU validation questions)

Base 0.583, QLoRA 0.553, LoRA 0.593. With n = 300 the standard error is about 2.8 points, so no measurable forgetting of general knowledge.

### Error analysis (share of the 500 test questions, seed 0)

| Failure type | Base (few-shot+) | QLoRA | LoRA |
|---|---|---|---|
| Wrong aggregation | 12.8% | 7.2% | 6.6% |
| Wrong columns / logic | 14.4% | 3.2% | 2.6% |
| Wrong literal / value | 4.4% | 2.6% | 3.2% |
| Hallucinated column | 6.4% | 0.0% | 0.0% |
| Join mismatch | 0.8% | 0.2% | 0.2% |
| Ordering / limit | 0.2% | 0.2% | 0.0% |

![error categories](error_categories.png)

Fine-tuning eliminated hallucinated columns and most structural errors. What remains is dominated by cases that look like **label noise or arbitrary dataset conventions**, from reading ten failures (not a systematic audit):
- The gold SQL for "more than 245 runs" is `runs = "245"`, while the model's `runs > 245` is arguably more correct.
- "Name the silver for baseball" has gold `MIN(silver)`; WikiSQL assigns aggregates somewhat arbitrarily.
- Several "wrong" answers differ only in the capitalization of a literal (`"USISL D-3 Pro league"` vs `"... League"`), which the question text cannot determine.

So the achievable ceiling on this test set is below 100%. On the OOD set (QLoRA, seed 0: 30 of 60 correct) the failures were wrong columns or logic (10), hallucinated columns (9), join mismatches (5), wrong aggregation (4) and wrong literal (2).

## 4. When should you fine-tune?

Based on these results:
- **Fine-tune** if your queries look like your training data: the gain over the best prompt was about 25 points, and LoRA or QLoRA both deliver it. QLoRA is the better choice when memory is the constraint.
- **Do not rely on the fine-tuned model for new schemas or query shapes** (multi-table joins). Here a few-shot prompt on the untouched model was far better (0.80 vs 0.44) and needed no training.
- **Likely explanation (not proven):** the training data is dominated by single-table WikiSQL-style questions with their own labeling conventions; more of it pulls the model toward those conventions. A natural next step is training on a mix that includes multi-table examples, or training for fewer steps.

## 5. Limitations (read before trusting the numbers)

- 500 test questions give roughly +/- 3 point intervals; the OOD set has 60 questions, so roughly +/- 12 points, and OOD accuracy varied from 0.38 to 0.55 across seeds. Treat the OOD result as a strong signal about direction, not a precise number.
- Synthetic databases approximate real data; execution accuracy can still count a wrong query as right if it returns the same rows by coincidence, or a right one as wrong (string comparison is case-sensitive).
- Gold labels in this dataset are noisy (see error analysis), so absolute accuracy understates model quality.
- One model size, one dataset, one task, one epoch. Seeds were only repeated for the main comparison. Ablations and the learning curve use one seed. The learning curve stops at 2,000 examples.
- The MMLU subset is a coarse probe of forgetting.

## 6. Reproduce

Open `sql_finetune_llama32_v2.ipynb` in Colab (T4), request access to `meta-llama/Llama-3.2-3B-Instruct`, install the packages, restart the session, and run the cells in order. Every step has a check cell; training runs are cached to Google Drive so a disconnect only loses the current run.


