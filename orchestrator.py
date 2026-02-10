import subprocess
import os
import time
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# PREREQUISITES:
# 1. subxpat repo in the same parent directory of this repo
# 2. Python 3.8+ installed
# 3. updated Z3Log folder in subxpat/.venv/lib/python3.8/site-packages

# 1a. Generate verilog files using /subxpat/main.py
# 1b. For each generated verilog
    # 2. Pass generated verilog to analyzer (npy_generator.py) -> save result on results.csv + generate .npy
    # 3. callback training when analyzer is done (res_net_training.py)
# 3. end execution

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.dirname(CURR_DIR)
SUBXPAT_DIR = os.path.join(SUBDIR, "subxpat")
SUBXPAT_OUTPUT_DIR = os.path.join(SUBXPAT_DIR, "output", "ver")
SCRIPT_XPAT = os.path.join(SUBXPAT_DIR, "main.py")

SCRIPT_ANALIZER = os.path.join(CURR_DIR, "npy_generator.py")

ANALIZER_OUTPUT_DIR = os.path.join(CURR_DIR, "npy_outputs")

SCRIPT_TRAINING = os.path.join(CURR_DIR, "res_net_training.py")

def is_file_ready(file_path):
    try:
        subprocess.check_output(["lsof", "-w", file_path])
        return False
    except subprocess.CalledProcessError:
        return True
    except FileNotFoundError:
        return False


def run_analizer(file_path, bitwidth, output_dir, stepsize, stepfactor):
    filename = os.path.basename(file_path)
    output_npy_name = os.path.splitext(filename)[0] + f"_ss{stepsize}_sf{stepfactor}.npy"
    output_npy_path = os.path.join(output_dir, output_npy_name)
    
    cmd = [
        sys.executable, SCRIPT_ANALIZER,
        file_path,          
        str(bitwidth),      
        output_npy_path,
        "--stepsize", str(stepsize),
        "--stepfactor", str(stepfactor)
    ]
    
    try:
        with open("log/analizer_out.log", "a") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"[ANALIZER] Done: {filename}")
        return output_npy_path 
    except subprocess.CalledProcessError as e:
        print(f"[ANALIZER] Error on {filename}: {e}")
        raise

def run_training(input_npy_path, conv_type, model_name, exact_acc_val, bitwidth):
    filename = os.path.basename(input_npy_path)

    print(f"[TRAINING] Starting: {filename} | Conv Type: {conv_type} | Model: {model_name} | Exact Acc: {exact_acc_val} | Bitwidth: {bitwidth}")
    
    cmd = [
        sys.executable, SCRIPT_TRAINING,
        "--conv_type", str(conv_type),
        "--model_name", str(model_name),
        "--input_path", input_npy_path,
        "--bit_width", str(bitwidth)
    ]

    if exact_acc_val is not None:
         cmd.extend(["--exact_accuracy", str(exact_acc_val)])
    
    
    try:
        with open("log/training_out.log", "a") as log_file:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"[TRAINING] Done: {filename}")
        return input_npy_path
    except subprocess.CalledProcessError as e:
        print(f"[TRAINING] Error on {filename}: {e}")
        raise

def orchestrator(args):
    
    start_timestamp = time.time()
    print(f"--- ORCHESTRATOR STARTED ---")
    

    if not os.path.exists(ANALIZER_OUTPUT_DIR):
        os.makedirs(ANALIZER_OUTPUT_DIR)

    #setup venv if non existant
    venv_python = os.path.join(SUBXPAT_DIR, ".venv", "bin", "python")
    subprocess.run([venv_python, "-m", "pip", "install", "jinja2"], cwd=SUBXPAT_DIR, check=True)


    if not os.path.exists(venv_python):
        print(f"Virtual environment not found at {venv_python}")
        print("Attempting to run 'make setup' in subxpat directory...")
        try:
            subprocess.run(["make", "setup"], cwd=SUBXPAT_DIR, check=True)
            print("Setup completed successfully.")
        except subprocess.CalledProcessError:
            print("'make setup' failed. Please fix the subxpat repo manually.")
            return
        except FileNotFoundError:
            print("'make' command not found. Cannot run setup.")
            return
        
        if not os.path.exists(venv_python):
            print("Error: 'make setup' finished but python executable is still missing.")
            return

    # --- 1a. START SUBXPAT ---
    if os.path.exists(SCRIPT_XPAT):
        subxpat_args = [
            venv_python, SCRIPT_XPAT,
            args.exact_benchmark, 
            "--subxpat", 
            "--template", "nonshared", 
            "--extraction-mode", str(args.extraction_mode), 
            "--min-labeling", 
            "--encoding", "z3bvec", 
            "--max-error", str(args.max_error), 
            "--imax", "4", 
            "--omax", "2", 
            "--max-lpp", "4", 
            "--max-ppo", "4", 
            "--baseet", "45", 
            "--stepsize", str(args.stepsize), 
            "--stepfactor", str(args.stepfactor), 
            "--metric", "wre", 
            "--timeout", str(args.timeout)
        ]

        """
        subxpat_args = [
            venv_python, SCRIPT_XPAT,
            "adder_i8_o5", "--max-lpp=8", "--max-ppo=32", "--max-error=16"
        ]
        """

        # extract bitwidth from benchmark name
        try:
            bitwidth = int(int(args.exact_benchmark.split("_i")[1].split("_")[0]) / 2)
            print(f"Extracted bitwidth: {bitwidth}")
        except IndexError:
            print("Errore: Il nome del benchmark non segue il formato standard (es. mul_i8_o5)")
            sys.exit(1)

        print(f"Launching SUBXPAT in background...")
        my_env = os.environ.copy()
        
        my_env["PYTHONPATH"] = SUBXPAT_DIR + os.pathsep + my_env.get("PYTHONPATH", "")

        venv_bin = os.path.dirname(venv_python)
        my_env["PATH"] = venv_bin + os.pathsep + my_env.get("PATH", "")

        proc_gen = subprocess.Popen(subxpat_args, cwd=SUBXPAT_DIR, env=my_env)
    else:
        print(f"ERROR: {SCRIPT_XPAT} not found.")
        return

    processed_files = set()
    training_futures = []

    gen_is_running = True

    # 1b. for each new file created by subxpat launch analizer + training
    with ProcessPoolExecutor() as analizer_executor, ThreadPoolExecutor(max_workers=4) as training_executor:

        pending_locked_files = False

        while gen_is_running or len(training_futures) > 0 or pending_locked_files:

            #print(f"Checking for new files... (Gen Running: {gen_is_running} | Pending Locked Files: {pending_locked_files} | Training Queue: {len(training_futures)})")

            pending_locked_files = False

            training_futures = [f for f in training_futures if not f.done()]

            gen_is_running = proc_gen.poll() is None
            files_to_process = []
            
            if os.path.exists(SUBXPAT_OUTPUT_DIR):
                try:
                    for f in os.listdir(SUBXPAT_OUTPUT_DIR):
                        full_path = os.path.join(SUBXPAT_OUTPUT_DIR, f)
                        if os.path.isfile(full_path) and full_path not in processed_files:
                            try:
                                # Check if file is created by current run
                                if os.path.getmtime(full_path) > start_timestamp:
                                    files_to_process.append(full_path)
                            except FileNotFoundError:
                                continue
                except FileNotFoundError:
                    pass

            for file_path in files_to_process:

                if not is_file_ready(file_path):
                    pending_locked_files = True
                    continue

                processed_files.add(file_path)
                
                # 2. Submit ANALYZER (run circuits_analizer.py)
                future_ana = analizer_executor.submit(
                    run_analizer, 
                    file_path, 
                    bitwidth, 
                    ANALIZER_OUTPUT_DIR,
                    str(args.stepsize),
                    str(args.stepfactor)
                )
                
                # 3. Callback TRAINING (run res_net_training.py) when ANALYZER is done
                def schedule_training(f_ana):
                    try:
                        npy_path = f_ana.result()
                        t_fut = training_executor.submit(
                            run_training, 
                            npy_path, 
                            args.conv_type, 
                            args.model_name,
                            args.exact_accuracy,
                            bitwidth 
                        )
                        training_futures.append(t_fut)
                    except Exception as e:
                        print(f"Skip Training: Analysis Error -> {e}")

                future_ana.add_done_callback(schedule_training)

            time.sleep(1)

    print("--- COMPLETED ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for Subxpat -> Analyzer -> Training")
    
    # Subxpat arguments
    parser.add_argument(metavar='exact-benchmark', dest='exact_benchmark', type=str, help='Circuit to approximate (Verilog file in `input/ver/`)')
    parser.add_argument("--extraction-mode", type=str, default="55", help="Value for --extraction-mode in subxpat (Default: 55)")
    parser.add_argument("--max-error", type=str, default="100", help="Value for --max-error in subxpat (Default: 100)")
    parser.add_argument('--stepsize', type=int, required=False, default=10, help='Value for --stepsize in subxpat (Default: 10)')
    parser.add_argument('--stepfactor', type=int, required=False, default=2, help='Value for --stepfactor in subxpat (Default: 2)')
    parser.add_argument("--timeout", type=str, default="10800", help="Value for --timeout in subxpat (Default: 10800 seconds)")
    
    
    # Training arguments
    parser.add_argument("--conv-type", type=str, default="3", help="Conv type for training (Default: 3)")
    parser.add_argument("--model-name", type=str, default="resnet", help="Model name for training (Default: resnet)")
    parser.add_argument("--exact-accuracy", type=int, default=None, help="Integer value for --exact-accuracy in training")

    args = parser.parse_args()
    
    orchestrator(args)