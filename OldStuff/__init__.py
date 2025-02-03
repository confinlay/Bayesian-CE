import torch

if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")