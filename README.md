# Batch Bayesian Optimisation for protein sequence desgin
This repository contains a PyTorch implementation of the paper Protein Sequence Design with Batch Bayesian Optimisation, which is a variant of directed evolution, which prioritizes the search for low-order mutants. Based on the local search mechanism that prioritizes the search for low-order mutants, a 1D CNN (Convolutional Neural Network) is utilized to specialize in exploring the local fitness landscape surrounding the wild type within this variant of directed evolution.

# Installation

```
conda create -n bbo python=3.8 -y
conda activate bbo
conda install pytorch=1.10.2 -c pytorch -y
conda install numpy=1.19 pandas=1.3 -y
conda install -c conda-forge tape_proteins=0.5 -y
pip install sequence-models==1.2.0
```

# Usage
Run the following commands to reproduce the main results shown in the paper. There are 12 fitness landscapes to support a diverse evaluation on black-box protein sequence design.
```
bash run.sh
```
