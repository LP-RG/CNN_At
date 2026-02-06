import sys
import os
import argparse
import time
from vpadanalyzer.synthesis import Synthesis


# file(baseId_stepsize_stepfactor),area,power,delay,pda,mean_ae,mean_ae_cnn,accuracy


CSV_HEADER = "file,area,power,delay,pda,mean_ae,mean_ae_cnn,max_ae\n"

CSV_FILE = "results.csv"

try:
    import sub_xpat_circuits_generator
    import sub_x_pat_simulator
except ImportError:
    sys.path.append(os.getcwd())
    import sub_xpat_circuits_generator
    import sub_x_pat_simulator


def circuits_analizer(input_path):
    """
    Analizza un circuito e restituisce la sua area, potenza e ritardo.
    """
    area = Synthesis.area(input_path)
    power = Synthesis.power(input_path)
    delay = Synthesis.delay(input_path)
    print("Synth completed")
    return {"file": os.path.basename(input_path), "area": area, "power": power, "delay": delay}

def generate_npy_for_single_file(input_verilog, bitwidth, output_npy_path, stepsize=None, stepfactor=None):
    """
    1. Converte il Verilog in funzione Python.
    2. Simula e crea il .npy.
    3. Appende le metriche in un file CSV.
    """
    
    filename = os.path.basename(input_verilog)

    #append stepsize and stepfactor to file name
    if stepsize is not None and stepfactor is not None:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_ss{stepsize}_sf{stepfactor}{ext}"
        output_npy_path = os.path.join(os.path.dirname(output_npy_path), f"{name}_ss{stepsize}_sf{stepfactor}.npy")
    
    print(f"[NPY GEN] Processing: {filename} -> {output_npy_path}")
    print(f"[NPY GEN] Stepsize: {stepsize}, Stepfactor: {stepfactor}, Bitwidth: {bitwidth}")

    sub_xpat_circuits_generator.generate_approx_mult_function(input_verilog, bitwidth)

    try:
        mean_ae, mean_ae_cnn, max_error = sub_x_pat_simulator.execute_save(
            bitwidth, 
            output_npy_path
        )

        data = circuits_analizer(input_verilog)

        print(f"[NPY GEN] Success. MAE: {mean_ae}, MaxErr: {max_error}")
        print(f"[NPY GEN] Synth Metrics: Area: {data['area']}, Power: {data['power']}, Delay: {data['delay']}")

        with open(CSV_FILE, "a") as f:
            pda = data['area'] * data['power'] * data['delay']
            csv_line = f"{filename},{data['area']},{data['power']},{data['delay']},{pda},{mean_ae},{mean_ae_cnn},{max_error}\n"
            f.write(csv_line)

    except Exception as e:
        print(f"[NPY GEN] Error during simulation: {e}")
        if os.path.exists("sub_x_pat_multiplier.py"):
            os.remove("sub_x_pat_multiplier.py")
        sys.exit(1)

    if os.path.exists("sub_x_pat_multiplier.py"):
        os.remove("sub_x_pat_multiplier.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NPY from Verilog for Orchestrator")
    parser.add_argument("input_verilog", type=str, help="Path to input .v file")
    parser.add_argument("bitwidth", type=int, help="Bitwidth (e.g., 8 or 16)")
    parser.add_argument("output_npy", type=str, help="Path to output .npy file")

    parser.add_argument("--stepsize", type=int, help="Stepsize for SubXpat (default: 10)")
    parser.add_argument("--stepfactor", type=int, help="Stepfactor for SubXpat (default: 2)")

    args = parser.parse_args()

    if not os.path.exists(args.input_verilog):
        print(f"Error: Input file {args.input_verilog} not found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)

    generate_npy_for_single_file(args.input_verilog, args.bitwidth, args.output_npy, args.stepsize, args.stepfactor)