# genCell: Diffusion-Based Synthetic Single-Cell Data Generation

`genCell` is a deep generative framework for synthesizing high-fidelity single-cell RNA-seq (scRNA-seq) profiles using **stable diffusion models**.  
This project explores how the **choice of prior distribution**—**Gaussian**, **Log-normal**, and **Student’s t**—affects model training dynamics, data quality, and biological realism in generated cellular gene expression data.  
The work focuses on generating realistic synthetic cells that preserve biological structure and variability, enabling applications in **data augmentation**, **benchmarking**, and **computational biology research**.  

---

## Key Features

- **Diffusion Model** adapted for scRNA-seq (tabular data)
- Support for **multiple prior distributions**:
  - Standard Gaussian
  - Log-normal (biologically motivated)
  - Student’s t (heavy-tailed robustness)
- Benchmarking with **baseline generative models**:
  - VAE (Variational Autoencoder)
  - GAN (Generative Adversarial Network)
- **Quantitative and qualitative evaluation**, including:
  - **Energy Distance**
  - **Maximum Mean Discrepancy (MMD)**
  - **Spearman Rank Correlation**
  - **KNN-AUC**
  - **UMAP visualization**

---

## Dataset

This project uses a subset of the **Tabula Muris** dataset (subset with 10 diverse cell types across multiple organs).

| Cell Type | Sample Count |
|---------|-------------|
| B cell | 5918 |
| Basal cell (epidermis) | 3964 |
| T cell | 3643 |
| Keratinocyte | 3536 |
| Mesenchymal progenitor cell | 1676 |
| Fibroblast | 1666 |
| Chondrocyte | 1630 |
| Mesenchymal stem cell | 1468 |
| Endothelial cell | 1418 |
| Macrophage | 1290 |

- Format: `AnnData (.h5ad)`
- Input dimension: 128 genes per cell
- Values: log-normalized expression matrix

---

## Project Structure

genCell/

├── data/ # (placeholder - real dataset not included)

├── models/ # Diffusion, VAE, and GAN model definitions

├── training/ # Training scripts for each model

├── generate/ # Synthetic data generation scripts

├── evaluation/ # Metrics + UMAP + KNN-AUC evaluation

└── README.md

---
## Results Overview

Gaussian prior: stable convergence, good baseline quality

Student’s t prior: improved robustness to outlier gene expression

Log-normal prior: heavy tailed distribution, caused structure collapse

The Gaussian and Student's t prior most closely matched real scRNA-seq distributions, supporting its biological relevance.

---
## Installation

### Requirements
```
pip install -r requirements.txt

