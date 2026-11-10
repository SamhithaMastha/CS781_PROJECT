# CS781 Capstone: In-House Legal Intelligence System

**Samhitha Mastha** · Roll No. 240634 · samhitham24@iitk.ac.in · IIT Kanpur
Codabench username: `s_240634`

A compact, self-contained legal NLP pipeline built across seven competition phases. A small legal foundation model was **pretrained from scratch** (custom tokenizer, no pretrained LMs, no external legal corpora), four different architectures were compared, and the resulting checkpoints were adapted to six downstream legal tasks.

The full write-up is in [`report/CS781-CP-Samhitha-Mastha-240634-report.pdf`](report/CS781-CP-Samhitha-Mastha-240634-report.pdf).

---

## Results (official Codabench scores, 9 Nov 2026)

| Phase | Task | Metric | Best score | Rank | Best system |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore | **0.8287** | 95/142 | Model A (decoder-only) |
| 2 | Legal NER (strict span) | Macro-F1 | **0.4251** | 98/133 | Ensemble of Model A checkpoints |
| 3 | Rhetorical role prediction | Macro-F1 | **0.4752** | 83/131 | E1: Model A + LoRA |
| 4 | Court judgment prediction | Macro-F1 | **0.6246** | 114/131 | BEST B: Model B, full FT, attention pooling |
| 5 | Statute identification | Macro-F1 | **0.3584** | 87/133 | exp4: Model D distilled from A |
| 6 | Abstractive summarization | BERTScore | **0.8318** | 91/131 | E4: Model D distilled from A (LoRA teacher) |
| 7 | Prior case retrieval | Micro-F1@7 | **0.3904** | 66/131 | BM25 + dense fusion (ties BM25) |

39 valid submissions in total, at least 4 per phase.

---

## Repository layout

```
CS781_PROJECT/
├── README.md
├── requirements.txt
├── report/
│   └── CS781-CP-Samhitha-Mastha-240634-report.pdf
└── code/
    ├── phase1/phase1.ipynb   # tokenizer + pretraining of 4 architectures + generation eval
    ├── phase2/phase2.ipynb   # Legal NER
    ├── phase3/phase3.ipynb   # Rhetorical roles (CL + IT)
    ├── phase4/phase4.ipynb   # Court judgment prediction
    ├── phase5/phase5.ipynb   # Statute identification
    ├── phase6/phase6.ipynb   # Abstractive summarization
    └── phase7/phase7.ipynb   # Prior case retrieval
```

All code is notebook-based and was developed and run on **Kaggle GPU notebooks (NVIDIA Tesla T4, 16 GB)**.

---

## Architecture

### Phase 1 foundation models (all within the 50M–300M parameter limit)

| Model | Type | Params | Key design |
|---|---|---|---|
| **A** | Decoder-only Transformer | 109.5M | 12 layers, hidden 768, 12 heads |
| **B** | Encoder–decoder | 123.7M | 6 encoder + 6 decoder layers |
| **C** | Hybrid | 114.3M | Gated linear recurrence, attention every 3rd layer |
| **D** | Efficient | 85.9M | Sliding-window attention (256), global attention every 4th layer, GQA with 4 KV heads |

- **Tokenizer:** custom 32k byte-pair encoding, saved as `bpe32k.json`.
- **Training:** AdamW, peak LR 6e-4, cosine decay, 200 warmup steps, block length 1024, fp16, 1,000 steps × 32,768 tokens (~32.8M tokens consumed).
- **Corpus:** Acts, High Courts, Supreme Court and Tribunals (170M training words after exact-dedup, minimum-length filtering and source sampling; High Courts and Tribunals kept at 50%).
- **Decoding for evaluation:** temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.1.

### Downstream transfer (Phases 2–7)

Each phase loads the Phase 1 checkpoints and `bpe32k.json`, adds a task head, and compares adaptation methods across the four backbones.

| Phase | Methods compared | Task-specific design |
|---|---|---|
| 2 NER | Full FT (A), LoRMA (B), LoRA (C), KD (D) | 512-token windows, 128 overlap, char offsets preserved for strict span matching |
| 3 Rhetorical roles | LoRA (A), LoRMA (B), IA3 (C), KD (D), full A as teacher | Document-level context chunks with sentence positions |
| 4 CJPE | Full FT + attention pooling (B), BitFit (C), full/KD (D), LoRMA (A) | Multi-chunk long-document handling (head / tail / full), tuned decision threshold |
| 5 Statutes | Full FT (A), LoRMA (B), QLoRA (C), KD (D) | 100-label multi-label head, per-label thresholds |
| 6 Summarization | LoRA (A), Houlsby and Pfeiffer adapters (B), LoRMA (C), full FT (D), KD (D) | Generation-length study, BERTScore and ROUGE |
| 7 PCR | BM25, partial FT, LoRMA, IA3, KD student, ensemble | Contrastive dense retrieval fused with BM25 (fusion weight tuned on dev) |

---

## Phase-by-phase description and individual submissions

Every notebook contains the full experiment sweep for its phase, and each experiment produced one Codabench submission (`prediction.json` inside a zip). The tables below list **every submission** (39 in total) with its backbone, adaptation method, local development score and official score. The row marked ★ is the best submission for the phase.

### Phase 1: Legal LM pretraining (`code/phase1/phase1.ipynb`, 4 submissions)

**Methodology.** Exact-duplicate removal, minimum-length filtering, per-source sampling, 32k BPE tokenizer trained from scratch, then pretraining of four architectures from random initialisation (1,000 steps × 32,768 tokens each). Evaluation generates continuations for held-out prompts and scores them with BERTScore, ROUGE-L and BLEU.

**Implementation.** Corpus cleaning and statistics, tokenizer training, token binarisation, a shared training loop (AdamW, cosine schedule, fp16, resumable checkpoints), a decoding sweep (sampling vs greedy with no-repeat 3-grams) and zip packaging.

| Submission | Model | Params | Val PPL | Local BERT / ROUGE-L / BLEU | Official BERT / ROUGE-L / BLEU |
|---|---|---|---|---|---|
| ★ A | Decoder-only | 109.5M | 29.1 | 0.8304 / 0.2076 / 0.0819 | **0.8287** / 0.1838 / 0.0290 |
| C | Hybrid (gated recurrence + attention) | 114.3M | 29.6 | 0.8295 / 0.2046 / 0.0803 | 0.8278 / 0.1824 / 0.0280 |
| D | Efficient (SWA + GQA) | 85.9M | 29.6 | 0.8284 / 0.2061 / 0.0807 | 0.8272 / 0.1812 / 0.0266 |
| B | Encoder–decoder | 123.7M | 55.2 | 0.8131 / 0.1733 / 0.0327 | 0.8166 / 0.1739 / 0.0180 |

### Phase 2: Legal NER (`code/phase2/phase2.ipynb`, 7 submissions)

**Methodology.** 12 entity labels, strict character-span matching. Documents are cut into 512-token windows with 128-token overlap (windows without entities kept with probability 0.3), with original character offsets preserved so spans can be reconstructed exactly. A token-classification head sits on the Phase 1 backbone (A is run bidirectionally).

**Implementation.** Window builder, offset reconstruction, strict span-F1 scorer, per-experiment training with best-epoch checkpointing, KD student training, and a checkpoint ensemble.

| Submission | Backbone and method | Dev macro-F1 | Official macro-F1 |
|---|---|---|---|
| ★ Ensemble of A checkpoints | A, 4 members (`exp1_full_A`, `abl_A_512_lr5e-5`, `abl_A_1024_lr1e-4`, `abl_A_1024_lr5e-5`) | 0.3551 | **0.4251** |
| exp1 | A, full fine-tuning | 0.3519 | 0.4177 |
| exp4 | D, knowledge distillation (teacher = exp1) | 0.3162 | 0.3745 |
| exp3b | C, LoRA (wide) | 0.2934 | 0.3458 |
| exp3 | C, LoRA | 0.2708 | 0.2992 |
| exp2d | B, LoRMA (long schedule) | 0.2125 | 0.2418 |
| exp2 | B, LoRMA | 0.1061 | 0.1092 |

### Phase 3: Rhetorical role prediction (`code/phase3/phase3.ipynb`, 4 submissions)

**Methodology.** Sentence-level role classification over two domains (Corporate Law and Information Technology), 13 roles in total. Compares sentence-only context with document-level chunks (about 960 tokens) plus sentence-position information. Controlled ablations E6–E8 and a no-distillation control E5 are included as development runs.

**Implementation.** Domain-aware data normalisation (`CL::id` / `IT::id` keys, merged into one `prediction.json` at the end), LoRA / LoRMA / IA3 injection, distillation from the full-A teacher E0, and measurement of trainable parameters and peak memory (e.g. B-LoRMA 2.39M trainable, 1.55 GB; C-IA3 0.87M trainable, 3.83 GB).

| Submission | Backbone and method | Dev macro-F1 (all) | Official macro-F1 |
|---|---|---|---|
| ★ E1 | A, LoRA, document context + position | 0.4588 | **0.4752** |
| E4 | D, distillation from E0 (A full teacher) | 0.4759 | 0.4326 |
| E3 | C, IA3 | 0.4319 | 0.4065 |
| E2 | B, LoRMA | 0.3010 | 0.2820 |

Development-only runs (not submitted): E0 A full teacher 0.5175, E5 D full without distillation 0.4514, E6 sentence context only 0.3246, E7 document context without position 0.4464, E8 rsLoRA 0.4693.

### Phase 4: Court judgment prediction (`code/phase4/phase4.ipynb`, 7 submissions)

**Methodology.** Binary judgment prediction on long cases (median 2,546 tokens). Cases are split into up to 6 × 512-token chunks, encoded, and combined by attention pooling. Head, tail and full-document contexts were compared, and the decision threshold was tuned on development data.

**Implementation.** Chunked encoder, attention-pooling classifier, technique registry (`apply_technique`: full, BitFit, LoRMA, KD), resumable state in `state.json`, and a `SMOKE` flag for quick sanity runs. The final BEST B model trains on all 37,387 labelled cases for 3 epochs with threshold 0.35.

| Submission | Backbone and method | Dev macro-F1 | Official macro-F1 |
|---|---|---|---|
| ★ BEST B | B, full FT, all data, 3 epochs | 0.6513* | **0.6246** |
| B full | B, full FT, attention pooling | 0.5912 | 0.5874 |
| D kd | D, distillation | 0.5845 | 0.5451 |
| Ensemble | Five runs | 0.5825 | 0.5420 |
| C bitfit | C, BitFit | 0.5703 | 0.5385 |
| D full | D, full FT | 0.5734 | 0.5364 |
| A lorma | A, LoRMA | 0.5672 | 0.5306 |

\*Tuned threshold, so an optimistic estimate; at threshold 0.5 the dev score is 0.5955.

### Phase 5: Legal statute identification (`code/phase5/phase5.ipynb`, 5 submissions)

**Methodology.** Multi-label classification over 100 imbalanced statute labels (3.71 labels per document on average), with global vs per-label thresholds (exp1 dev macro-F1 0.3545 → 0.3632). Documents are encoded as 2 × 512-token chunks.

**Implementation.** Multi-label head with BCE, per-label threshold search, QLoRA through `bitsandbytes`, saving of teacher logits and embeddings (`*_train_out.pt`) for distillation.

| Submission | Backbone and method | Dev macro-F1 | Official macro-F1 |
|---|---|---|---|
| ★ exp4 | D, distillation from exp1 | 0.3677 | **0.3584** |
| exp3 | C, QLoRA | 0.3630 | 0.3557 |
| exp1 | A, full fine-tuning | 0.3632 | 0.3555 |
| Ablation | A, LoRMA, head mode | 0.3449 | 0.3382 |
| exp2 | B, LoRMA | 0.2360 | 0.2344 |

Best run also reaches micro-F1 0.4552 and subset accuracy 0.0326.

### Phase 6: Abstractive summarization (`code/phase6/phase6.ipynb`, 6 submissions)

**Methodology.** Long legal judgments (median 3,128 words) summarised to about 638 words. Input length 1280 plus 256 generated tokens. The reference summaries are highly extractive, so generation settings were also checked for their effect on length and overlap (minimum new tokens 60 / 100 / 150).

**Implementation.** Seq2seq / decoder generation wrappers for all four backbones, LoRA, Houlsby and Pfeiffer adapters, LoRMA, full fine-tuning, a KD student distilled from the E1 teacher, and BERTScore + ROUGE evaluation.

| Submission | Backbone and method | Local BERTScore | Official BERTScore |
|---|---|---|---|
| ★ E4 | D, distillation (teacher E1) | 0.8349 | **0.8318** |
| E3 | C, LoRMA | 0.8305 | 0.8283 |
| E6 | D, full FT (no teacher) | 0.8297 | 0.8278 |
| E1 | A, LoRA | 0.8313 | 0.8271 |
| E2 | B, Houlsby adapters | 0.8216 | 0.8183 |
| E5 | B, Pfeiffer adapters | 0.8195 | 0.8162 |

### Phase 7: Prior case retrieval (`code/phase7/phase7.ipynb`, 6 submissions)

**Methodology.** Given a query case, retrieve the top 7 relevant candidates. Dense encoders (Phase 1 model A, 4 × 512 chunked inputs, contrastive training) are fused with a BM25 score; the fusion weight is tuned on the development split. Zero-shot transfer and pooling / chunking ablations come first.

**Implementation.** Single-pass tokenisation of queries and candidates, chunked encoding, in-batch contrastive training, BM25 implementation, fusion and top-k selection, failure analysis (`failure_analysis_dev.csv`), and a final zip validator.

| Submission file | System | Dense-only dev F1@7 | Fused dev F1@7 | Fusion weight | Official F1@7 |
|---|---|---|---|---|---|
| `sub1_bm25.zip` | BM25 baseline | n/a | 0.3780 | n/a | 0.3904 |
| `sub2_partial_ft.zip` | Partial / layer-wise fine-tuning | 0.2902 | 0.3991 | 0.45 | 0.3904 |
| `sub3_lorma.zip` | LoRMA, rank 8 | 0.2946 | 0.3924 | 0.40 | 0.3881 |
| `sub4_ia3.zip` | IA3 | 0.2446 | 0.3880 | 0.30 | 0.3904 |
| `sub5_kd_student.zip` | Distillation student | 0.2535 | 0.3924 | 0.55 | 0.3816 |
| `sub6_ensemble.zip` | Ensemble of all | 0.2946 | 0.3980 | 0.40 | 0.3904 |

Four submissions tie with BM25 at the best official score (0.3904), so the sparse baseline is the main driver of the result. The tied systems are `sub1_bm25`, `sub2_partial_ft`, `sub4_ia3` and `sub6_ensemble`.

### Compute and hyperparameter summary

| Hyperparameter | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|
| Representative params (M) | 109.5 | 109.6 | 112.9 | 67.1 | 55.3 | 85.9 | 109.5 |
| Context / block length | 1024 | 512 + overlap | 1024 | 6 × 512 | 2 × 512 | 1280 + 256 | 4 × 512 |
| Peak LR | 6e-4 | 1e-4 / 1e-3 | 4e-4 / 1e-3 | 1e-4 / 1e-3 | 1e-4 / 1e-3 | 3e-4 | 5e-5 |
| Epochs / steps | 1000 steps | best of 5 ep | 4 | 3 | 3 | 600 steps | 600 steps |
| Main technique | pretrain | FT / PEFT / KD | LoRA | pooling / KD | multi-label / KD | generation / KD | dense + BM25 |

Hardware: Kaggle NVIDIA Tesla T4 (16 GB). Where the report notes that a hyperparameter was not consistently recorded, it is not listed here.

---

## Setup and execution

### Requirements
- Kaggle (or Colab) notebook with a GPU (T4 16 GB was used) and Internet enabled for the first run.
- A Hugging Face access token with access to the course dataset `Exploration-Lab/CS-781-Capstone`, stored as a Kaggle secret named **`HF_TOKEN`**. No tokens are stored in this repository.
- Python 3.10+. Dependencies are listed in `requirements.txt`; Kaggle images already include most of them.

```bash
pip install -r requirements.txt
```

### Running the phases in order

The phases depend on each other: **Phase 1 must be run first**, because every later phase needs its outputs.

1. **Phase 1.** Upload `code/phase1/phase1.ipynb` to Kaggle, add the `HF_TOKEN` secret, and *Run All*. It downloads the data, cleans and samples the corpus, trains the 32k BPE tokenizer, pretrains models A–D and evaluates generation. Outputs to `/kaggle/working`: `ckpt/{A_decoder,B_encdec,C_hybrid,D_efficient}.pt`, `bpe32k.json`, and `submissions/*.zip`.
2. **Save the Phase 1 outputs as a Kaggle Dataset.** This must include the four `*.pt` checkpoints and `bpe32k.json`.
3. **Phases 2–7.** For each notebook, attach the Phase 1 dataset as an *Input*, add `HF_TOKEN`, and *Run All*. Notebooks locate checkpoints and the tokenizer automatically by searching `/kaggle/input`. The task data is downloaded from the Hugging Face dataset on the first run.
4. Each phase writes Codabench-ready `*.zip` files, each containing only `prediction.json` at the archive root, to `/kaggle/working` (see the `submissions` folders). Every notebook validates the zip format before finishing.

Phase-specific notes:
- **Phase 4:** set `SMOKE = True` for a roughly 5-minute sanity run, then `SMOKE = False` for the real run.
- **Phase 7:** optional environment variables `P7_DATA_DIR`, `P7_TOK`, `P7_INPUT_ROOTS`, `P7_WORK` override data, tokenizer and working paths; it falls back to downloading the data from Hugging Face.
- **Phases 3 and 4:** the notebooks cache finished runs (`result.json` / `state.json`), so re-running resumes rather than retrains.

### Reproducibility notes
- Paths are Kaggle-style (`/kaggle/input`, `/kaggle/working`, `/kaggle/temp`). To run elsewhere, change the path constants at the top of each notebook.
- Exact scores can vary slightly with GPU, library versions and seeds. Reported numbers are the official Codabench results.
- Local validation and hidden-test rankings did not always agree (notably Phases 2 and 3), and the Phase 2 and Phase 3 dev splits are small. See the report, sections 5.2 and 7.

---

## Key findings

- The decoder-only Model A was the best Phase 1 model and a strong base for NER and rhetorical roles.
- Knowledge distillation into the compact Model D was competitive or best in Phases 5 and 6.
- Document-level context mattered much more than positional information for rhetorical roles (dev macro-F1 0.3246 → 0.4464 → 0.4588).
- BM25 was hard to beat for prior case retrieval: four systems tied with it at 0.3904.
- Strict-span NER is dominated by boundary errors and false positives.

---

## LLM usage declaration

ChatGPT was used for concept clarification, brainstorming, small debugging questions, and report wording and LaTeX formatting. Core training and inference code, experiments, hyperparameter tuning and submissions were done by the author. Full declaration is in the report.

## References

1. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
2. C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," 2004.
3. T. Zhang et al., "BERTScore: Evaluating Text Generation with BERT," ICLR, 2020.
