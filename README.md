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
