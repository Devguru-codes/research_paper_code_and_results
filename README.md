# Annotation Budget Matters: A Systematic Evaluation of Deep Learning and Gradient Boosting for Hyperspectral Soil Classification

This repository contains the source code, experimental notebooks, and resulting metrics for the paper:
**"Annotation Budget Matters: A Systematic Evaluation of Deep Learning and Gradient Boosting for Hyperspectral Soil Classification"**

## Overview
This study challenges the assumption that complex deep learning architectures are universally optimal for hyperspectral imaging (HSI) classification, particularly under severe annotation starvation. We pit modern deep sequence models (Vision Transformers, SpectralFormers, GCNs, 3D-CNNs) against robust traditional gradient-boosting ensembles (LightGBM).

**Key Findings:**
1. **The Crossover Point:** Under extreme few-shot regimes (e.g., $<5\%$ training data), classical LightGBM remains robust and outperforms attention-based mechanisms. As the annotation budget increases, deep models with local inductive biases (HybridSN) rapidly overtake LightGBM.
2. **Asymmetric Cross-Domain Transfer:** Pre-training on abundant macro-agriculture data successfully rescues deep models in data-starved micro-soil scenarios. However, the reverse yields negative transfer due to severe domain shift.
3. **Physical Interpretability:** Utilizing SHAP (SHapley Additive exPlanations), we verified that models ground their decisions in physically meaningful absorption bands (e.g., water absorption at 1400nm, clay content).

## Repository Structure

```
.
├── notebooks/                # Jupyter notebooks for model training, transfer learning, and ablations
├── results_from_notebook/    # Output metrics, classification maps, and SHAP analyses
├── *.png                     # High-resolution figures generated for the manuscript
└── springer_manuscript.tex   # The LaTeX source file for the final manuscript
```

## Abstract
The integration of deep learning architectures in hyperspectral imaging (HSI) has yielded exceptional results, yet a significant disconnect remains between standard benchmarking environments and the operational realities of remote sensing, where ground-truth labeling is extremely scarce. In this paper, we challenge the assumption that complex deep learning models are optimal for all scenarios by evaluating modern architectures against traditional gradient-boosting ensembles across strict annotation budgets. Our empirical analysis reveals a distinct crossover point: while standard deep sequence models exhibit severe degradation in extreme few-shot regimes ($<5\%$ training data), classical LightGBM models remain robust, outperforming domain-specific transformers. We then introduce an Asymmetric Cross-Domain Transfer Learning framework, demonstrating that pre-training on an abundant macro-agriculture source domain successfully rescues deep models in data-starved micro-soil target environments, whereas the reverse domain shift yields negative transfer. Ultimately, we provide an actionable framework for agricultural deployment, recommending traditional ensembles for off-the-shelf, low-resource drone applications, and reserving deep models for scenarios where hierarchical pre-training is feasible.

## Models Evaluated
* **LightGBM** (Gradient Boosting Decision Trees)
* **HybridSN** (3D-CNN with strong local inductive biases)
* **Vanilla ViT** (Standard Vision Transformer)
* **SpectralFormer-inspired** (Domain-specific Vision Transformer)
* **1D-CNN** & **GCN** (Baselines)

## Getting Started
The notebooks can be executed independently. Ensure that you have the appropriate Python environment with PyTorch (for deep models) and LightGBM installed. Wait for the dataset paths to be properly configured before running the notebooks locally.
