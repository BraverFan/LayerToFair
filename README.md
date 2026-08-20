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
| Importance threshold | `threshold` |  `0.01` | Feature importance threshold for key neuron identification. Neurons with importance scores above this threshold are selected as key neurons. |
| Scale range | `scale_bounds` | `[0,2]` | Search range for the scaling factors applied to key neuron outputs. |

## Reproducing Multi-Seed Results
All experiments are run with 10 random seeds to account for randomness in the repair process. The pretrained models corresponding to all 10 seeds are integrated in the `model_paths` list in each run script, and all seeds are executed in a single run by default.

To reproduce the full multi-seed results reported in the paper, simply run:
```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```
If you wish to run a specific seed only, comment out the other entries in the `model_paths` list in the corresponding script before running.

# Reproducing Experimental Results

This document provides instructions for reproducing the experimental results corresponding to each research question (RQ) in our paper.

For all commands below:

- `{dataset}` represents the dataset name (e.g., `adult`, `bank`, `compas`, `default`, `meps16`).
- `{sensitive_attribute}` represents the sensitive attribute used in the experiment (e.g., `sex`, `race`).

Users can reproduce the reported results by replacing these placeholders with the corresponding dataset and sensitive attribute.

---

## RQ1: Comparison with Existing Fairness Repair Methods

RQ1 evaluates the effectiveness of LayerToFair compared with existing fairness repair methods.

### LayerToFair (Our Method)

To reproduce the results of LayerToFair, run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

The optimization time can be controlled by modifying the following parameter in the corresponding script:

```python
max_time_minutes
```

---

### NeuFair

To reproduce the results of NeuFair, run:

```bash
python NeuFair/final_{dataset}_{sensitive_attribute}.py
```

---

### CARE

To reproduce the results of CARE, run:

```bash
python care/care_{dataset}_{sensitive_attribute}.py
```

---

### FairFLRep

The implementation of FairFLRep is available at:

https://github.com/openjamoses/FairFLRep

Run:

```bash
python fairflrep.py
```

to reproduce the corresponding results.

---

# RQ2: Effect of Different Layer Repair Strategies

RQ2 studies the impact of different layer repair strategies, including:

- Key Layers
- All Layers
- Last Layer

---

## Key Layers

To reproduce the Key Layers strategy, run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

The number of repaired layers can be controlled by modifying:

```python
key_layers_num
```

in the corresponding script.

For example:

```python
key_layers_num = 4
```

means that four key layers are repaired.

---

## All Layers

To reproduce the All Layers strategy, run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}_all_layers.py
```

Alternatively, modify the following line in:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

as:

```python
layer_name_keys = list(key_neurons.keys())
```

to repair all hidden layers.

---

## Last Layer

To reproduce the Last Layer strategy, run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

and modify:

```python
layer_name_keys = list(key_neurons.keys())[-1:]
```

to repair only the last hidden layer.

---

# RQ3: Effect of Different Neuron Identification Strategies

RQ3 investigates the influence of different neuron identification strategies.

---

## LayerToFair (Our Method)

Run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

---

## Random Neuron Identification Strategy

Run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}_random_neurons.py
```

---

## Gradient-based Neuron Identification Strategy

Run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}_grad_neurons.py
```

---

## Causality-based Neuron Identification Strategy

Run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}_causal_neurons.py
```

---

# RQ4: Effect of Different Optimization Algorithms

RQ4 compares different optimization algorithms for fairness repair.

The compared optimization methods include:

- GRPO
- PPO
- PSO

---

## GRPO

Run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

---

## PPO

Run:

```bash
python GRPO/ppo_{dataset}_{sensitive_attribute}.py
```

---

## PSO

Run:

```bash
python care/care_{dataset}_{sensitive_attribute}.py
```

For a fair comparison, the neuron localization module in CARE should be replaced with the neuron identification module implemented in:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

while keeping the PSO optimization procedure unchanged.

---

# RQ5: Effect of Feature Importance Threshold

RQ5 evaluates the impact of different feature importance thresholds.

To reproduce the results, run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

The feature importance threshold can be adjusted by modifying:

```python
threshold
```

in the corresponding script.

Different values of `threshold` generate results under different feature importance selection criteria.

---

# RQ6 Flexible Fairness Metric Adaptation

## EOD Metric

By default, LayerToFair optimizes the EOD metric. To reproduce the results under 
the EOD metric, directly run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```
---

## SPD Metric

To reproduce the results under the SPD metric, modify 
`GRPO/Environment.py`.

Find:

```python
new_fairness, _, _, new_performance, _ = compute_metrics(
    self.model, self.X_val, self.y_val,
    self.sens_val, self.sens_classes, self.dataset
)
```

Replace with:

```python
_, new_fairness, _, new_performance, _ = compute_metrics(
    self.model, self.X_val, self.y_val,
    self.sens_val, self.sens_classes, self.dataset
)
```

Then run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```

---

## DI Metric

To reproduce the results under the DI metric, modify 
`GRPO/Environment.py`.

Find:

```python
new_fairness, _, _, new_performance, _ = compute_metrics(
    self.model, self.X_val, self.y_val,
    self.sens_val, self.sens_classes, self.dataset
)
```

Replace with:

```python
_, _, new_fairness, new_performance, _ = compute_metrics(
    self.model, self.X_val, self.y_val,
    self.sens_val, self.sens_classes, self.dataset
)
```

Then run:

```bash
python GRPO/grpo_{dataset}_{sensitive_attribute}.py
```
