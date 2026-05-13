# LayerToFair
Official implementation of the paper "LayerToFair: An Efficient Post-processing Framework for Layer-aware Fairness Repair in Deep Neural Networks".
## Framework of LayerToFair
<img width="1859" height="624" alt="image" src="https://github.com/user-attachments/assets/41e3783e-21f3-4344-a866-72076a559ad2" />

## Environment Setup

We recommend using a conda environment:

```bash
conda create -n layertofair python=3.9
conda activate layertofair
pip install torch==2.3.1
pip install numpy==2.0.1
pip install scikit-learn==1.6.1
```

## Repository Structure

```
LayerToFair/
├── GRPO/                  # LayerToFair scripts
├── NeuFair/               # NeuFair scripts
├── care/                  # CARE scripts
├── saved models/          # pretrained models 
└── README.md
```

## Usage
To run LayerToFair on a specific dataset with a specific sensitive attribute, simply run the corresponding script under the `GRPO/` folder:
```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```
## Hyperparameter Configuration
Two key hyperparameters can be configured directly in each run script (`GRPO/grpo_{dataset}_{sensitive_attribute}.py`):

| Parameter | Variable in Script | Default | Description |
|-----------|-----------|---------|-------------|
| Importance threshold | `threshold` | `0.01` | Feature importance threshold for key neuron identification. Neurons with importance scores above this threshold are selected as key neurons. |
| Scale range | `scale_bounds` | `[0, 2]` | Search range for the scaling factors applied to key neuron outputs. |

## Reproducing Multi-Seed Results
All experiments are run with 10 random seeds to account for randomness in the repair process. The pretrained models corresponding to all 10 seeds are integrated in the `model_paths` list in each run script, and all seeds are executed in a single run by default.

To reproduce the full multi-seed results reported in the paper, simply run:
```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```
If you wish to run a specific seed only, comment out the other entries in the `model_paths` list in the corresponding script before running.

