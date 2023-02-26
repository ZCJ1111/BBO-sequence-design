# Proximal Exploration (PEX)

This repository contains a PyTorch implementation of our paper [Proximal Exploration for Model-guided Protein Sequence Design](https://www.biorxiv.org/content/10.1101/2022.04.12.487986) published at ICML 2022.
Proximal Exploration (PEX) is a variant of directed evolution, which prioritizes the search for low-order mutants.
Based this local-search mechanism, a model architecture called Mutation Factorization Network (MuFacNet) is developed to specialize in the local fitness landscape around the wild type.

## Installation

The dependencies can be set up using the following commands:

```bash
conda create -n pex python=3.8 -y
conda activate pex
conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y
conda install numpy=1.19 pandas=1.3 -y
conda install -c conda-forge tape_proteins=0.5 -y
pip install sequence-models==1.2.0
pip install flexs
pip install gpytorch
```


## Usage

Run the following commands to search for CDR3 sequences with lowest dG in the given dataset:

```bash
python run.py \
  --device 'cuda:0' \
  --landscape custom \
  --alg pex \
  --name 'esm-pex-1ADQ' \
  --num_rounds 40 \
  --net esm1b \
  --ensemble_size 1 \
  --out-dir /path/to/output \
  --fitness-data /path/to/dataset \
  --sequence-column 'CDR3' \
  --fitness-column 'Energy' \
  --invert-score

```
