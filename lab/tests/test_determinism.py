"""Determinism probe for MPS backend."""
import torch
from transformer_lens import HookedTransformer

print("Running MPS determinism probe...")
torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("On CPU, exiting probe.")
    exit()

m = HookedTransformer.from_pretrained("gpt2-small", device=device, dtype=torch.float32)
toks = m.to_tokens(["hello world"])

# Run 1
torch.manual_seed(42)
out1 = m(toks).detach()

# Run 2
torch.manual_seed(42)
out2 = m(toks).detach()

diff = (out1 - out2).abs().max().item()
print(f"Max absolute diff between two seeded runs: {diff}")

if diff > 1e-5:
    print("WARNING: Determinism drift detected.")
else:
    print("SUCCESS: Output appears deterministic.")
