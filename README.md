# REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://technion-cs-nlp.github.io/REVS/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2406.09325v3)

Welcome to the official repository for REVS, a groundbreaking method for surgically removing sensitive information from Large Language Models while preserving their capabilities and defending against extraction attacks.

**🚨 The Problem**: Language models memorize and leak sensitive information (emails, SSNs, URLs) from their training data, creating serious privacy and compliance risks.

**✅ Our Solution**: REVS identifies and modifies specific neurons that promote sensitive tokens, demoting them in the vocabulary space without breaking the model's general knowledge.


## REVS Main Method

(1) Project neuron to vocabulary space →
(2) Demote sensitive token rank →
(3) Project back to update neuron

## 🎯 Key Features

- **🛡️ Robust Against Attacks**: First unlearning method that resists extraction attacks (Logit-Lens, Delta, Perturbation)
- **🎪 Surgical Precision**: Targets specific neurons without degrading general capabilities
- **📊 Real-World Tested**: Evaluated on naturally memorized emails, URLs, and synthetic SSNs
- **⚡ Efficient**: Non-gradient-based approach that's faster than retraining

## 📈 Performance Highlights

| Dataset | REVS Unlearning Score | Best Baseline
|---------|----------------------|---------------
| SSN (Synthetic) | **89.58%** | 36.98%
| Emails (Natural) | **62.37%** | 50.30%
| URLs (Natural) | **44.25%** | 28.03%

*Results on Llama-3-8B. REVS consistently outperforms all baselines while maintaining model integrity.*

## 🚀 Quick Start

### Installation

Set up your environment using our provided conda configuration:

```bash
# Clone the repository
git clone https://github.com/technion-cs-nlp/REVS.git
cd REVS

# Create conda environment with all dependencies
./setup_conda_env.sh

# Activate the environment
conda activate revsenv
```

## 📚 Demo

Explore our interactive Jupyter notebook:

```bash
jupyter notebook notebooks/revs_demo.ipynb
```

This demo showcases unlearning sample of naturally memorized email addresses

### Supported Models

- **Llama-3-8B** (meta-llama/Llama-3-8b)
- **GPT-J-6B** (EleutherAI/gpt-j-6b)

*Support for additional models coming soon!*

## 📊 Datasets

We provide three carefully curated datasets:

1. **📧 Emails**: 205 real email addresses naturally memorized by Llama-3-8B
2. **🔗 URLs**: 203 real URLs naturally memorized by GPT-J-6B
3. **🆔 SSNs**: 200 synthetic social security numbers for controlled testing

*Note: All sensitive data has been anonymized for research purposes.*

## 🔬 Method Overview

REVS operates through three key phases:

### 1. 🎯 Localization
- **Layer Selection**: Identify layers where target tokens rank highly
- **Neuron Selection**: Find neurons with high activation + strong token association

### 2. ✂️ Editing
- **Vocabulary Projection**: Project neurons to vocabulary logit space
- **Rank Demotion**: Iteratively reduce target token ranks
- **Back Projection**: Update neuron values in hidden space

### 3. 🛡️ Verification
- **Effectiveness**: Measure unlearning success with capped rank scores
- **Integrity**: Ensure general capabilities remain intact (MMLU, GSM8K)
- **Robustness**: Test against extraction attacks

## 📖 Paper Citation

If you use REVS in your research, please cite:

```bibtex
@inproceedings{ashuach2025revs,
  title={REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space},
  author={Ashuach, Tomer and Tutek, Martin and Belinkov, Yonatan},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```
