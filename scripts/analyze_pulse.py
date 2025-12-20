
import json

def analyze_pulse(metrics_file):
    steps = []
    edi = []
    
    pulse_start = 45000
    pulse_end = 45500
    
    print(f"{'Step':<6} | {'EDI':<8} | {'Note'}")
    print("-" * 30)
    
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            step = data['step']
            
            # Check for EDI keys
            edi_val = None
            for k, v in data.items():
                if "EDI" in k:
                    edi_val = v
                    break
            
            if edi_val is not None:
                steps.append(step)
                edi.append(edi_val)
                
                # Print around pulse
                if 44000 <= step <= 46000 and step % 100 == 0:
                     note = ""
                     if pulse_start <= step <= pulse_end:
                         note = "* PULSE *"
                     print(f"{step:<6} | {edi_val:.4f}   | {note}")

    if not steps:
        print("No EDI metrics found in log.")
        return

    # Simple text plot
    print("\nText Plot:")
    for s, e in zip(steps, edi):
        if s % 100 == 0 and 44000 <= s <= 46000:
             bar = "#" * int(e * 20)
             print(f"{s:4d}: {bar} ({e:.2f})")

if __name__ == "__main__":
    analyze_pulse("reports/parity/train/parity_head0_omega1.0_seed1_production_run/metrics.jsonl")
