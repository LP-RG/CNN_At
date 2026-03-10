import subprocess
import os
import time
import sys
import argparse
import shutil
import csv
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIR = os.path.dirname(CURR_DIR)
SUBXPAT_DIR = os.path.join(SUBDIR, "A_A_T/subxpat")

if SUBXPAT_DIR not in sys.path:
    sys.path.insert(0, SUBXPAT_DIR)

try:
    from subxpat.sxpat.specifications import *
except ImportError as e:
    print(f"Error importing sxpat specifications: {e}")
    raise

SCRIPT_ANALIZER = os.path.join(CURR_DIR, "npy_generator.py")
ANALIZER_OUTPUT_DIR = os.path.join(CURR_DIR, "npy_outputs")
SCRIPT_TRAINING = os.path.join(CURR_DIR, "res_net_training.py")

CSV_FILE = "results.csv"
csv_lock = Lock()

def get_log_paths(exp_name):
    log_dir = os.path.join(CURR_DIR, "log", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    return {
        "subxpat": os.path.join(log_dir, "subxpat.log"),
        "analyzer": os.path.join(log_dir, "analyzer.log"),
        "training": os.path.join(log_dir, "training.log")
    }

def check_if_accuracy_exists_in_csv(npy_path):
    filename_key = os.path.splitext(os.path.basename(npy_path))[0]
    if not os.path.exists(CSV_FILE):
        return None
    with csv_lock:
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["file"] == filename_key:
                    acc = row.get("accuracy", "").strip()
                    if acc != "" and float(acc) != 0:
                        return acc
    return None

def update_accuracy_in_csv(npy_path, accuracy):
    filename_key = os.path.splitext(os.path.basename(npy_path))[0]
    if not os.path.exists(CSV_FILE):
        return
    with csv_lock:
        rows = []
        updated = False
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["file"] == filename_key:
                    row["accuracy"] = str(accuracy)
                    updated = True
                rows.append(row)
        if updated:
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

def copy_project_to_exp(base: str, destination: str):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(base, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))

def is_file_ready(file_path):
    try:
        subprocess.check_output(["lsof", "-w", file_path])
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True

def run_analizer(file_path, bitwidth, output_dir, exp_name, log_path):
    filename = os.path.basename(file_path)
    output_npy_name = os.path.splitext(filename)[0] + f"_{exp_name}.npy"
    output_npy_path = os.path.join(output_dir, output_npy_name)
    cmd = [sys.executable, SCRIPT_ANALIZER, file_path, str(bitwidth), output_npy_path, "--experiment-name", exp_name]
    with open(log_path, "a") as lfile:
        lfile.write(f"\n--- ANALYZING: {filename} ---\n")
        subprocess.run(cmd, check=True, stdout=lfile, stderr=lfile)
    return output_npy_path

def run_training(input_npy_path, conv_type, model_name, exact_acc_val, bitwidth, log_path):
    filename_key = os.path.splitext(os.path.basename(input_npy_path))[0]
    existing_acc = check_if_accuracy_exists_in_csv(input_npy_path)
    if existing_acc:
        return input_npy_path
    cmd = [sys.executable, SCRIPT_TRAINING, "--conv_type", str(conv_type), "--model_name", str(model_name), "--input_path", input_npy_path, "--bit_width", str(bitwidth)]
    if exact_acc_val is not None:
        cmd.extend(["--exact_accuracy", str(exact_acc_val)])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    with open(log_path, "a") as lfile:
        lfile.write(f"\n--- TRAINING: {filename_key} ---\n")
        lfile.write(result.stdout)
    final_accuracy = None
    for line in result.stdout.splitlines():
        if line.startswith("FINAL_ACCURACY:"):
            try:
                final_accuracy = float(line.split(":")[1].strip())
            except ValueError: pass
            break
    if final_accuracy is not None:
        update_accuracy_in_csv(input_npy_path, final_accuracy)
    return input_npy_path

def orchestrator(args, subxpat_argv):
    start_timestamp = time.time()
    exp_name = args.experiment_name
    logs = get_log_paths(exp_name)
    print(f"--- ORCHESTRATOR STARTED ({exp_name}) ---")
    EXP_SUBXPAT_DIR = os.path.join(CURR_DIR, exp_name)
    copy_project_to_exp(SUBXPAT_DIR, EXP_SUBXPAT_DIR)
    
    VENV_ACTIVATE = os.path.join(EXP_SUBXPAT_DIR, ".venv/bin/activate")
    if not os.path.exists(VENV_ACTIVATE):
        subprocess.run(["make", "setup"], cwd=EXP_SUBXPAT_DIR, check=True)

    # Costruzione comando SubXpat pulita
    subxpat_cmd_list = ["python3", "main.py"]
    # Aggiunge gli argomenti originali passati da riga di comando all'orchestrator
    subxpat_cmd_list.extend(subxpat_argv)

    full_shell_command = f"source {VENV_ACTIVATE} && {' '.join(subxpat_cmd_list)}"

    try:
        bitwidth = int(int(args.exact_benchmark.split("_i")[1].split("_")[0]) // 2)
    except:
        sys.exit("Benchmark format error")

    subxpat_log_file = open(logs['subxpat'], "w")
    proc_gen = subprocess.Popen(["/bin/bash", "-c", full_shell_command], cwd=EXP_SUBXPAT_DIR, stdout=subxpat_log_file, stderr=subxpat_log_file)

    processed_files = set()
    training_futures = []
    if not os.path.exists(ANALIZER_OUTPUT_DIR): os.makedirs(ANALIZER_OUTPUT_DIR)

    with ProcessPoolExecutor() as analizer_executor, ThreadPoolExecutor(max_workers=2) as training_executor:
        while proc_gen.poll() is None or len(training_futures) > 0:
            training_futures = [f for f in training_futures if not f.done()]
            ver_dir = os.path.join(EXP_SUBXPAT_DIR, "output", "ver")
            if os.path.exists(ver_dir):
                for f in os.listdir(ver_dir):
                    full_path = os.path.join(ver_dir, f)
                    if not os.path.isfile(full_path) or full_path in processed_files: continue
                    if os.path.getmtime(full_path) <= start_timestamp: continue
                    if not is_file_ready(full_path): continue
                    processed_files.add(full_path)
                    future_ana = analizer_executor.submit(run_analizer, full_path, bitwidth, ANALIZER_OUTPUT_DIR, exp_name, logs['analyzer'])
                    def schedule_training(f_ana):
                        try:
                            npy_path = f_ana.result()
                            t_fut = training_executor.submit(run_training, npy_path, args.conv_type, args.model_name, args.exact_accuracy, bitwidth, logs['training'])
                            training_futures.append(t_fut)
                        except Exception as e: print(f"Error: {e}")
                    future_ana.add_done_callback(schedule_training)
            time.sleep(1)

    subxpat_log_file.close()
    print("--- COMPLETED ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-type", default="3")
    parser.add_argument("--model-name", default="resnet")
    parser.add_argument("--exact-accuracy", type=int, default=None)
    parser.add_argument("--experiment-name", required=True)
    args, subxpat_argv = parser.parse_known_args()
    
    # Carichiamo le specifiche solo per validazione e per estrarre exact_benchmark
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + subxpat_argv
    specs = Specifications.parse_args()
    sys.argv = original_argv
    vars(args).update(vars(specs))
    
    orchestrator(args, subxpat_argv)