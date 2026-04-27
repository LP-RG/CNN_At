import sys
import os
import argparse
import csv
import time
from vpadanalyzer.synthesis import Synthesis

CSV_HEADER = "file,area,power,delay,pda,mean_ae,mean_ae_cnn,max_ae,accuracy\n"
CSV_FILE = "results.csv"

try:
    import sub_xpat_circuits_generator
    import sub_x_pat_simulator
except ImportError:
    sys.path.append(os.getcwd())
    import sub_xpat_circuits_generator
    import sub_x_pat_simulator

def find_matching_stats(current_stats):
    if not os.path.exists(CSV_FILE):
        return None
    try:
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = True
                for key in current_stats:
                    if str(row.get(key)) != str(current_stats[key]):
                        match = False
                        break
                if match:
                    acc_val = row.get("accuracy", "").strip()
                    if acc_val != "":
                        try:
                            if float(acc_val) != 0:
                                return acc_val
                        except ValueError: continue
    except Exception:
        return None
    return None

def circuits_analizer(input_path, power_sim_input):
    area = Synthesis.area(input_path)
    start_time = time.time()
    power = Synthesis.power(input_path, input_vectors_file=power_sim_input)
    end_time = time.time()
    print(f"Synth completed in {end_time - start_time:.2f} seconds")
    delay = Synthesis.delay(input_path)
    return {"file": os.path.basename(input_path), "area": area, "power": power, "delay": delay}

def generate_npy_for_single_file(input_verilog, bitwidth, output_npy_path, experiment_name, power_sim_input, csv_file, calc_error):
    base_filename = os.path.basename(input_verilog)
    name, ext = os.path.splitext(base_filename)
    
    # Identificativo unico usato come chiave nel CSV
    filename_key = f"{name}_{experiment_name}"

    if calc_error:
        sub_xpat_circuits_generator.generate_approx_mult_function(input_verilog, bitwidth)

    try:
        if calc_error:
            mean_ae, mean_ae_cnn, max_error = sub_x_pat_simulator.execute_save(bitwidth, output_npy_path)
        else:
            mean_ae, mean_ae_cnn, max_error = -1.0, -1.0, -1.0  # Placeholder values
        data = circuits_analizer(input_verilog, power_sim_input)
        pda = data['area'] * data['power'] * data['delay']

        current_metrics = {
            "area": data['area'], "power": data['power'], "delay": data['delay'],
            "pda": pda, "mean_ae": mean_ae, "mean_ae_cnn": mean_ae_cnn, "max_ae": max_error
        }

        found_accuracy = find_matching_stats(current_metrics)
        
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a") as f:
            if not file_exists:
                f.write(CSV_HEADER)
            acc_str = found_accuracy if found_accuracy else ""
            csv_line = f"{filename_key},{data['area']},{data['power']},{data['delay']},{pda},{mean_ae},{mean_ae_cnn},{max_error},{acc_str}\n"
            f.write(csv_line)
        
        if found_accuracy:
            print(f"[NPY GEN] Match found! Accuracy {found_accuracy} reused.")

    except Exception as e:
        print(f"[NPY GEN] Error: {e}")
        sys.exit(1)
    finally:
        if os.path.exists("sub_x_pat_multiplier.py"):
            os.remove("sub_x_pat_multiplier.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_verilog")
    parser.add_argument("bitwidth", type=int)
    parser.add_argument("output_npy")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--power-sim-input")
    parser.add_argument("--csv-file", default=CSV_FILE)
    parser.add_argument("--calc-error", action="store_true", default=False)

    args = parser.parse_args()

    if os.path.isdir(args.input_verilog):
        verilog_files = [f for f in os.listdir(args.input_verilog) if f.endswith(".v")]
        if not verilog_files:
            print(f"Error: No .v files found in directory {args.input_verilog}.")
            sys.exit(1)

        os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
        
        for verilog_file in verilog_files:
            input_path = os.path.join(args.input_verilog, verilog_file)
            output_npy_path = os.path.join(os.path.dirname(args.output_npy), f"{os.path.splitext(verilog_file)[0]}.npy")
            generate_npy_for_single_file(input_path, args.bitwidth, output_npy_path, args.experiment_name, args.power_sim_input, args.csv_file, args.calc_error)
    else:
        if not os.path.exists(args.input_verilog):
            print(f"Error: Input file {args.input_verilog} not found.")
            sys.exit(1)

        os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
        generate_npy_for_single_file(args.input_verilog, args.bitwidth, args.output_npy, args.experiment_name, args.power_sim_input, args.csv_file, args.calc_error)