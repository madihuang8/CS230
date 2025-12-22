# Identifying Workslop: Evaluating Useability of AI-Generated Content

**Authors:** Kaitlyn Kwan, Sophie Li, Madison Huang

## Overview

Workslop is an emerging term that refers to AI-generated content that appears highly polished but lacks substantive value. As organizations increasingly turn to AI tools to boost productivity, studies reveal a paradoxical effect: rather than reducing workload, AI-generated output often shifts the burden to other workers who must parse through verbose, low-quality content. Harvard Business Review estimates that this cognitive offloading costs organizations up to $9 million annually in lost productivity.

This project tackles the challenge of distinguishing genuinely useful content from content that merely *appears* useful by focusing on **information density** through three approaches:

1. **Redundancy Detection** – Semantic similarity between sentences
2. **Specificity Scoring** – Improving existing specificity prediction methods
3. **Binary Classification** – Training on Wikipedia editor flags

## Methodology

### Method 1: Semantic Similarity (Redundancy Detection)

Sentences with high semantic similarity to surrounding text likely repeat content instead of contributing new information—a key marker of filler.

- **Dataset:** Semantic Textual Similarity Benchmark (STSB)
- **Model:** Fine-tuned `all-mpnet-base-v2` from sentence-transformers
- **Loss:** CosineSimilarityLoss
- **Results:** Test Pearson Cosine: 0.875, Test Spearman Cosine: 0.872

### Method 2: Specificity Scoring

Specific statements ("deployed on 3 A100s with batch size 32") are more useful than vague ones ("used appropriate hardware").

- **Baseline:** BiLSTM with hand-crafted features from "Domain Agnostic Real-Valued Specificity Prediction" (2019)
- **Experiments:** Replacing BiLSTM with pretrained transformers (DistilRoBERTa, MiniLM), better augmentation methods, Beta distribution for output space

### Method 3: Information Density Classification

A hybrid architecture combining pre-trained BERT embeddings with hand-crafted information density features.

- **Dataset:** WikiSQE (50,000 sentences, 80/10/10 split)
- **Custom Features:** Lexical diversity, specificity (proper nouns + numbers), normalized sentence length
- **Labels:** Wikipedia editor flags mapped to workslop vs. substantive-but-incomplete
- **Results:** Test Accuracy: 62.2%, F1 Score: 0.64

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| Substantive | 0.64 | 0.55 | 0.60 | 5022 |
| Workslop | 0.61 | 0.69 | 0.64 | 4978 |

## Future Work

- Validate label mapping with deeper WikiSQE dataset analysis
- Extend training beyond 3 epochs with better regularization
- Explore multitask learning combining all three approaches: `L = λ1*L_redundancy + λ2*L_Beta + λ3*L_wiki`
- Contrastive triplet training for better separation of meaningful vs. filler content

## Tech Stack

- PyTorch
- Hugging Face Transformers
- Sentence-Transformers
- BERT / DistilRoBERTa / MiniLM

## Getting Started
```bash
git clone https://github.com/madihuang8/CS230.git
cd CS230
pip install -r requirements.txt
```

## References

1. Cer et al. "SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation." (2017)
2. Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks." (2019)
3. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." (2019)
4. McCarthy & Jarvis. "MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment." (2010)
5. Ando et al. "WikiSQE: A Large-Scale Dataset for Sentence Quality Estimation in Wikipedia." AAAI (2024)

## Contributions

- **Kaitlyn** – Method 1: Redundancy from semantic similarity
- **Sophie** – Method 2: Real-valued specificity prediction
- **Madi** – Method 3: Information density from Wikipedia editors

---

*Built for CS230 at Stanford University*
