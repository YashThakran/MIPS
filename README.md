# Off-Policy Evaluation for Large Action Spaces via Embeddings

This repository contains the implementation of ICML 2022 paper "[Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf)" .

## Requirements and Setup

The Python environment is built using [poetry](https://github.com/python-poetry/poetry). You can build the same environment as in our experiments by cloning the repository and running `poetry install` directly under the folder (if you have not installed `poetry` yet, please run `pip install poetry` first).

```bash
# clone the repository
git clone https://github.com/usaito/icml2022-mips.git
cd src

# install poetry
pip install poetry

# build the environment with poetry
poetry install
```

The versions of Python and necessary packages are specified as follows (from [pyproject.toml](./pyproject.toml)).

```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "0.5.5"
scikit-learn = "1.0.2"
pandas = "1.3.5"
scipy = "1.7.3"
numpy = "^1.22.4"
matplotlib = "^3.5.2"
seaborn = "^0.11.2"
hydra-core = "1.0.7"
```

## Running the Code

The experimental workflow is implemented using [Hydra](https://github.com/facebookresearch/hydra). The commands needed to reproduce the experiments are summarized below. Please move under the `src` directly first and then run the commands. The experimental results (including the corresponding figures) will be stored in the `logs/` directory.

### Section 4.1: Synthetic Data

```bash
cd src


# How does MIPS perform with varying number of actions?
poetry run python synthetic/main_n_actions.py setting.beta=-1,0,1 setting.eps=0.05,0.8 -m


# How does MIPS perform with varying sample sizes?
poetry run python synthetic/main_n_val.py setting.beta=-1,0,1 setting.eps=0.05,0.8 -m


# How does MIPS perform with varying numbers of deficient actions?
poetry run python synthetic/main_n_def_actions.py setting.beta=-1,0,1 setting.eps=0.05,0.8 -m


# How does MIPS perform with varying number of unobserved embedding dimensions?
poetry run python synthetic/main_n_unobs_cat_dim.py setting.n_cat_per_dim=2 setting.n_cat_dim=20 setting.beta=-1,0,1 setting.eps=0.05,0.8 -m


# How data-driven embedding selection affects the performance of MIPS?
poetry run python synthetic/main_n_val.py setting.n_cat_per_dim=2 setting.n_cat_dim=20 setting.embed_selection=True


# How does MIPS perform with varying logging policies? (in the Appendix)
poetry run python synthetic/main_beta.py


# How does MIPS perform with varying evaluation policies? (in the Appendix)
poetry run python synthetic/main_eps.py


# How does MIPS perform with varying noise levels? (in the Appendix)
poetry run python synthetic/main_noise.py
```

### Section 4.2: Real-World Data

To run the real-world experiment, please download the [Open Bandit Dataset](https://research.zozo.com/data.html) (about 11GB) and place it as `./src/real/open_bandit_dataset/`. Then, run the following command. It may take a few days.


```bash
cd src

poetry run python real/main.py setting.sample_size=10000,50000,100000,500000 -m
```
