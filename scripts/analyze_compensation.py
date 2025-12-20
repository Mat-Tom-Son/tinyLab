
import json
import statistics

def load_metrics(path):
    steps = []
    accs = []
    edis = {} # head -> list of values
    
    try:
        with open(path, 'r') as f:
            for line in f:
                d = json.loads(line)
                steps.append(d['step'])
                accs.append(d['test_acc'])
                
                for k, v in d.items():
                    if "EDI" in k:
                        if k not in edis: edis[k] = []
                        edis[k].append(v)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None, None, None
        
    return steps, accs, edis

def analyze():
    # Paths
    prod_path = "reports/parity/train/parity_head0_omega1.0_seed1_production_run/metrics.jsonl"
    map_path = "reports/parity/train/parity_head0_omega1.0_seed1_compensation_map/metrics.jsonl"
    frozen_path = "reports/parity/train/parity_head0_omega1.0_seed1_frozen_control/metrics.jsonl"
    
    # Load
    p_steps, p_accs, _ = load_metrics(prod_path)
    m_steps, m_accs, m_edis = load_metrics(map_path)
    f_steps, f_accs, _ = load_metrics(frozen_path)
    
    if not p_steps or not m_steps or not f_steps:
        return

    # 1. Compare Recovery (Frozen vs Production)
    print("\n--- Recovery Comparison (Frozen vs Production) ---")
    pulse_indices = [i for i, s in enumerate(p_steps) if 45000 <= s <= 46000]
    
    print(f"{'Step':<8} | {'Prod Acc':<10} | {'Frozen Acc':<10}")
    print("-" * 35)
    for i in pulse_indices:
        step = p_steps[i]
        p_val = p_accs[i]
        
        # Find matching step in frozen
        try:
            f_idx = f_steps.index(step)
            f_val = f_accs[f_idx]
            print(f"{step:<8} | {p_val:.3f}      | {f_val:.3f}")
        except ValueError:
            pass

    # 2. Compensation Map (Did Heads 1-3 move?)
    print("\n--- Compensation Map (Heads 1-3 EDI) ---")
    print(f"{'Step':<8} | {'H0 (Target)':<12} | {'H1':<8} | {'H2':<8} | {'H3':<8}")
    print("-" * 55)
    
    for i in pulse_indices:
        step = m_steps[i]
        
        # Get EDI for all heads
        row_str = f"{step:<8} | "
        for h in range(4):
            key = f"L0H{h}_EDI"
            val = m_edis.get(key, [0.0]*len(m_steps))[i]
            row_str += f"{val:.4f}   "
            if h == 0: row_str += "  | "
            else: row_str += "| "
            
        print(row_str)

if __name__ == "__main__":
    analyze()
