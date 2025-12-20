
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_pulse():
    # Load data
    prod_path = "reports/parity/train/parity_head0_omega1.0_seed1_production_run/metrics.jsonl"
    steps, accs = [], []
    
    with open(prod_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            steps.append(d['step'])
            accs.append(d['test_acc'])
            
    # Filter for pulse window (44k to 46k)
    zoom_steps, zoom_accs = [], []
    for s, a in zip(steps, accs):
        if 44000 <= s <= 46000:
            zoom_steps.append(s)
            zoom_accs.append(a)
            
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(zoom_steps, zoom_accs, 'b-', linewidth=2, label='Test Accuracy')
    
    # Pulse region shaded
    plt.axvspan(45000, 45500, color='red', alpha=0.1, label='Pulse ($\omega=0.5$)')
    
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Homeostatic Recovery from Transient Pulse')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.6, 1.02)
    plt.tight_layout()
    
    plt.savefig('paper/unified_homeostasis/figures/pulse_recovery.pdf')
    print("Saved figure to paper/unified_homeostasis/figures/pulse_recovery.pdf")

if __name__ == "__main__":
    plot_pulse()
