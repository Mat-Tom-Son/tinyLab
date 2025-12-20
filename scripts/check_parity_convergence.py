
import json
import glob
import os
import numpy as np

def check_parity_convergence():
    # Target directories: Parity Omega 1.0 Baseline seeds (0, 1, 2)
    # Looking for 'sweep_summary.json' or 'metrics.json'
    
    # We found many directories. Let's look for "omega1.0_seed*_medium" which seems to be the main experiment
    patterns = [
        "reports/parity/train/parity_head0_omega1.0_seed*/metrics.jsonl"
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
        
    print(f"Found {len(files)} potential log files.")
    
    results = []
    
    for f in files:
        try:
            # Read last line of jsonl
            last_line = None
            with open(f, 'r') as fp:
                for line in fp:
                    if line.strip():
                        last_line = line
            
            if last_line:
                data = json.loads(last_line)
                # Look for 'edi' or 'entropy' keys
                # Parity code might log 'attn_entropy' or similar
                edi = data.get('edi') 
                if edi is None:
                    # Fallback: check if 'metrics' key exists
                    if 'metrics' in data and 'edi' in data['metrics']:
                        edi = data['metrics']['edi']
                        
                # Extract seed from filename
                parts = f.split('_seed')
                if len(parts) > 1:
                    seed = parts[1].split('/')[0].split('_')[0] # handle _medium or others
                else:
                    seed = "unknown"
                    
                step = data.get('step', 'unknown')
            
                if edi is not None:
                    results.append({'file': f, 'seed': seed, 'edi': edi, 'step': step})
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"{'Seed':<10} | {'Step':<10} | {'Final EDI (Parity)':<20}")
    print("-" * 50)
    
    values = []
    for r in results:
        print(f"{r['seed']:<10} | {r['step']:<10} | {r['edi']:<20}")
        if isinstance(r['edi'], (int, float)):
            values.append(r['edi'])
            
    if values:
        mean = np.mean(values)
        std = np.std(values)
        print("-" * 50)
        print(f"Mean EDI: {mean:.12f}")
        print(f"Std Dev:  {std:.12f}")
        
if __name__ == "__main__":
    check_parity_convergence()
