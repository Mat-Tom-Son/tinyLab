
import torch
import torch.nn.functional as F
import math

def compute_entropy_naive(probs):
    """Naive H = -sum(p * log(p))"""
    # Clip to avoid log(0)
    p = probs + 1e-16
    return -(p * torch.log(p)).sum(dim=-1)

def compute_entropy_stable(logits):
    """Stable H using log_softmax"""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)

def test_entropy_stability():
    print("=== Entropy Numerics Audit ===")
    
    # CASE 1: Uniform distribution
    logits = torch.zeros(10)
    ent_naive = compute_entropy_naive(F.softmax(logits, dim=-1))
    ent_stable = compute_entropy_stable(logits)
    print(f"Uniform (10): Naive={ent_naive.item():.8f}, Stable={ent_stable.item():.8f}, Diff={abs(ent_naive-ent_stable).item():.8e}")

    # CASE 2: Sharp distribution (underflow risk)
    logits = torch.tensor([100.0, -100.0])
    ent_naive = compute_entropy_naive(F.softmax(logits, dim=-1))
    ent_stable = compute_entropy_stable(logits)
    print(f"Sharp [100, -100]: Naive={ent_naive.item():.8f}, Stable={ent_stable.item():.8f}, Diff={abs(ent_naive-ent_stable).item():.8e}")
    
    # CASE 3: Extremely sharp (one huge logit)
    logits = torch.tensor([1000.0, 0.0, 0.0])
    ent_naive = compute_entropy_naive(F.softmax(logits, dim=-1))
    ent_stable = compute_entropy_stable(logits)
    print(f"Extreme [1000, 0, 0]: Naive={ent_naive.item():.8f}, Stable={ent_stable.item():.8f}, Diff={abs(ent_naive-ent_stable).item():.8e}")
    # Expected: ~0.0

    # CASE 4: Precision (float32 vs float64)
    logits_32 = torch.randn(10, dtype=torch.float32)
    logits_64 = logits_32.double()
    
    ent_32 = compute_entropy_stable(logits_32)
    ent_64 = compute_entropy_stable(logits_64)
    
    print(f"Random (10): Float32={ent_32.item():.8f}, Float64={ent_64.item():.8f}, Diff={abs(ent_32-ent_64).item():.8e}")

    # CASE 5: Tiny Lab specific case (attention weights)
    # Simulating a "crystallized" head with 1.0 on one token and 0 elsewhere
    # Naive method on "probs" often relies on 0 * log(0) handling
    probs_hard = torch.tensor([1.0, 0.0, 0.0])
    ent_naive_hard = compute_entropy_naive(probs_hard)
    print(f"Hard Probs [1,0,0]: Naive={ent_naive_hard.item():.8f} (Should be 0.0)")

if __name__ == "__main__":
    test_entropy_stability()
