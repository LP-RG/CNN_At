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

if SUBXPAT_DIR not in sys.path:
    sys.path.insert(0, SUBXPAT_DIR)
try:
    from sxpat.specifications import *
except ImportError as e:
    print(f"Error importing sxpat specifications: {e}")
    raise

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


def run_analizer(file_path, bitwidth, output_dir, beta, alpha):
    filename = os.path.basename(file_path)
    output_npy_name = os.path.splitext(filename)[0] + f"_beta{beta}_alpha{alpha}.npy"
    output_npy_path = os.path.join(output_dir, output_npy_name)
    
    cmd = [
        sys.executable, SCRIPT_ANALIZER,
        file_path,          
        str(bitwidth),      
        output_npy_path,
        "--beta", str(beta),
        "--alpha", str(alpha)
    ]
    
    try:
        with open("log/analizer_out.log", "a") as log_file:
            log_file.write(f"\n--- ANALYZER STARTED: {filename} | beta: {beta}, alpha: {alpha} | date: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
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
            log_file.write(f"\n--- TRAINING STARTED: {filename} | Conv Type: {conv_type} | Model: {model_name} | Exact Acc: {exact_acc_val} | Bitwidth: {bitwidth} | date: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
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
        
        def enum_to_str(val):
            if hasattr(val, 'value'):
                return str(val.value)
            return str(val)

        def add_arg_if_not_none(args_list, arg_name, arg_value):
            if arg_value is not None:
                args_list.extend([f"--{arg_name}", enum_to_str(arg_value)])
        
        subxpat_args = [
            venv_python, SCRIPT_XPAT,
            args.exact_benchmark, 
        ]
        
        if args.min_labeling:
            subxpat_args.append("--min-labeling")
        
        if args.subxpat:
            subxpat_args.append("--subxpat")

        add_arg_if_not_none(subxpat_args, "template", args.template)
        add_arg_if_not_none(subxpat_args, "extraction-mode", args.extraction_mode)
        add_arg_if_not_none(subxpat_args, "encoding", args.encoding)
        add_arg_if_not_none(subxpat_args, "max-error", args.max_error)
        add_arg_if_not_none(subxpat_args, "imax", args.imax)
        add_arg_if_not_none(subxpat_args, "omax", args.omax)
        add_arg_if_not_none(subxpat_args, "max-lpp", args.max_lpp)
        add_arg_if_not_none(subxpat_args, "max-ppo", args.max_ppo)
        add_arg_if_not_none(subxpat_args, "baseet", args.baseet)
        add_arg_if_not_none(subxpat_args, "beta", args.beta)
        add_arg_if_not_none(subxpat_args, "alpha", args.alpha)
        add_arg_if_not_none(subxpat_args, "c-constant", args.c_constant)
        add_arg_if_not_none(subxpat_args, "cnn-constraint", args.cnn_constraint)
        add_arg_if_not_none(subxpat_args, "metric", args.metric)
        add_arg_if_not_none(subxpat_args, "timeout", args.timeout)

        """
        subxpat_args = [
            venv_python, SCRIPT_XPAT,
            "adder_i8_o5", "--max-lpp=8", "--max-ppo=32", "--max-error=16"
        ]
        """

        # extract bitwidth from benchmark name
        try:
            bitwidth = int(int(args.exact_benchmark.split("_i")[1].split("_")[0]) // 2)
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
                
                # 2. Submit ANALYZER (run npy_generator.py)
                future_ana = analizer_executor.submit(
                    run_analizer, 
                    file_path, 
                    bitwidth, 
                    ANALIZER_OUTPUT_DIR,
                    str(args.beta),
                    str(args.alpha)
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
    parser = argparse.ArgumentParser(description="Orchestrator for Subxpat (see Subxpat help for infos) -> Analyzer -> Training")
    
    
    # Training arguments
    parser.add_argument("--conv-type", type=str, default="3", help="Conv type for training (Default: 3)")
    parser.add_argument("--model-name", type=str, default="resnet", help="Model name for training (Default: resnet)")
    parser.add_argument("--exact-accuracy", type=int, default=None, help="Integer value for --exact-accuracy in training")

    args, subxpat_argv = parser.parse_known_args()

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + subxpat_argv
    
    specs = Specifications.parse_args()
    
    sys.argv = original_argv
    vars(args).update(vars(specs))
    
    orchestrator(args)