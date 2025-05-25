import numpy as np
import torch

def count_learnable_params(model) -> int:
    """
    Works with:
      * your scratch NeuralNetwork (has .parameters())
      * the LLM model (has .parameters())
      * any torch.nn.Module
      * sklearn MLPClassifier after .fit()
    """
    # PyTorch or scratch variants
    if hasattr(model, "parameters"):
        return sum(
            (p.size if isinstance(p, np.ndarray) else p.numel())
            for p in model.parameters()
        )
    # scikit-learn MLPClassifier
    elif hasattr(model, "coefs_"):
        return sum(w.size for w in model.coefs_) + \
               sum(b.size for b in model.intercepts_)
    else:
        raise TypeError("Object exposes no learnable parameters")

def estimate_vram(param_count: int, dtype_bytes: int = 4, multiplier: int = 4) -> float:
    """
    Rough VRAM in **MB** for training with Adam (weights + grads + m + v).
    For inference only, set multiplier=1.
    """
    return param_count * dtype_bytes * multiplier / (1024 ** 2)