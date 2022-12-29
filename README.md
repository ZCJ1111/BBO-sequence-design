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
```

Clone this repository and download the oracle landscape models by the following commands:

```bash
git clone https://github.com/HeliXonProtein/proximal-exploration.git
cd proximal-exploration
bash download_landscape.sh
```

install: https://github.com/samsinai/FLEXS/

## Usage

Run the following commands to reproduce our main results shown in section 5.1. There are eight fitness landscapes to support a diverse evaluation on black-box protein sequence design.

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

In the default configuration, the protein fitness landscape is simulated by a TAPE-based oracle model. By adding the argument `--oracle_model=esm1b`, the landscape simulator is switched to an oracle model based on ESM-1b.
