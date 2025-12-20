
import time
import re
import sys

def monitor(files):
    print(f"Monitoring {files}...")
    best_acc = 0.0
    
    # Track file positions
    positions = {f: 0 for f in files}
    
    while True:
        changed = False
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    f.seek(positions[fpath])
                    lines = f.readlines()
                    positions[fpath] = f.tell()
                    
                    for line in lines:
                        if "test_acc" in line:
                            # Extract acc
                            m = re.search(r"test_acc=([0-9.]+)", line)
                            if m:
                                acc = float(m.group(1))
                                if acc > best_acc:
                                    best_acc = acc
                                    print(f"New best: {acc:.3f} in {fpath}")
                                
                                if acc > 0.95:
                                    print(f"GROKKING DETECTED! {fpath} reached {acc}")
                                    return fpath
                                    
                        if "Training complete" in line:
                            print(f"Finished: {fpath}")
            except FileNotFoundError:
                pass
                
        time.sleep(5)

if __name__ == "__main__":
    monitor(sys.argv[1:])
