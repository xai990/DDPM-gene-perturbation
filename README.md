# DDPM for mRNA gene expression perturbation

This repository contains the code of a diffusion model and a neural network model, specifically for breast cancer. 
## Installation

DDPM is a collection of Python scripts. Recommand that run diffusion model on [Palmetto](https://www.palmetto.clemson.edu/palmetto/) -- a Clemson university research cluster. To use the Python scripts directly, clone this repository.  All of the Python dependencies can be installed in an Anaconda environment:
```bash
# load Anaconda module if needed 
module load anaconda3/2022.05-gcc/9.5.0 

# clone repository
git clone https://github.com/xai990/DDPM-gene-perturbation.git

cd DDPM-gene-perturbation
# create conda environment called "DDPM"
conda create env -f environment.yml

# run DDPM on breast cancer data  -- make sure the scripts are running on a computed node (an interactive job includes at least one gpu)
python scripts/train.py
