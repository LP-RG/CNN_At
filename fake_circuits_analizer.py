import sys
import os
import time
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python circuit_analizer.py <input_verilog> <bitwidth> <output_npy>")
        sys.exit(1)

    input_path = sys.argv[1]
    bitwidth = sys.argv[2]
    output_path = sys.argv[3]

    print(f"[FAKE ANALYZER] Processing: {os.path.basename(input_path)}")
    print(f"                Bitwidth: {bitwidth}")
    print(f"                Target: {output_path}")

    if not os.path.exists(input_path):
        print(f"[FAKE ANALYZER] Error: Input file '{input_path}' not found!")
        sys.exit(1)

    time.sleep(0.5)


    fake_data = np.random.rand(256, 256).astype(np.float32)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.save(output_path, fake_data)
        print(f"[FAKE ANALYZER] Success: Created {output_path}")
        
    except Exception as e:
        print(f"[FAKE ANALYZER] Crash during save: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()