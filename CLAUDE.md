# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural network assignment comparing three implementations of feedforward networks:
- **`Custom_nn.py`** — from-scratch NumPy implementation (`NeuralNetwork`), variable depth, supports sigmoid/relu/tanh, binary cross-entropy loss, optional class weights
- **`LLM_nn.py`** — LLM-generated NumPy implementation (`LLMNeuralNetwork`), multiclass with softmax output, categorical cross-entropy loss
- **`PyTorch_nn.py`** — PyTorch `nn.Module` wrapper (`TorchModel`), fixed 2-hidden-layer architecture, `CrossEntropyLoss`
- **`NNutils.py`** — shared helpers: `count_learnable_params()` (works across all three), `estimate_vram()`

Notebooks (`main.ipynb`, `Breast_Cancer_Diagnostic.ipynb`) drive experiments on `data/Iris.csv` and breast cancer data.

## Key Architectural Differences

| | `NeuralNetwork` | `LLMNeuralNetwork` | `TorchModel` |
|---|---|---|---|
| Data layout | `(features, samples)` | `(samples, features)` | tensors `(samples, features)` |
| Output | sigmoid (binary) | softmax (multiclass) | logits → CrossEntropyLoss |
| Weights | `self.W`, `self.b` lists | `self.params` dict `W1/b1...` | `nn.Linear` layers |
| Train entry | `.train(X, Y)` | `.fit(X, y, epochs)` | manual loop calling `.backward()` |

## Running Code

No build step. Run notebooks with Jupyter; run scripts directly:

```bash
python Custom_nn.py
jupyter notebook main.ipynb
```

Dependencies: `numpy`, `torch`, `scikit-learn`, `pandas`, `matplotlib`. Install via:

```bash
uv add numpy torch scikit-learn pandas matplotlib
```
