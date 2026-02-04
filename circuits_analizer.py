from vpadanalyzer.synthesis import Synthesis
import os
import sys
import sub_xpat_circuits_generator
import time
import multiplier_outputs_plotting
import sub_x_pat_simulator
import csv

def circuits_analizer(input_path):
    """
    Analizza un circuito e restituisce la sua area, potenza e ritardo.
    """
    area = Synthesis.area(input_path)
    power = Synthesis.power(input_path)
    delay = Synthesis.delay(input_path)
    print("Synth completed")
    return {"file": os.path.basename(input_path), "area": area, "power": power, "delay": delay}

def create_matrices(multipliers_folder, bitwidth, output_plot_path, results):
    mean_ae_datas = []
    if not os.path.isdir(multipliers_folder):
        print(f"Errore: La cartella '{multipliers_folder}' non esiste.")
        return []
    os.makedirs(output_plot_path, exist_ok=True)
    # Raccoglie i dati di tutti i file
    for filename in sorted(os.listdir(multipliers_folder)):  # Ordina i file per nome
        print(f"simulating mult:{filename}")
        input_path = os.path.join(multipliers_folder, filename)
        if os.path.isfile(input_path):
            name = filename.split(".")[0]
            sub_xpat_circuits_generator.generate_approx_mult_function(input_path, bitwidth)
            try:
                mean_ae, mean_ae_cnn, max_error = sub_x_pat_simulator.execute_save(bitwidth,os.path.join(output_plot_path,name + ".npy"))
                mean_ae_datas.append({"file": os.path.basename(input_path), "mean_ae": mean_ae, "mean_ae_cnn": mean_ae_cnn, "max_ae": max_error})
                #multiplier_outputs_plotting.plots(name,os.path.join(output_plot_path,name + ".npy"),output_plot_path)
            except:
                print(f"skipping mult {filename}")
    return mean_ae_datas    
            
def analyze_multipliers(multipliers_folder):
    if not os.path.isdir(multipliers_folder):
        print(f"Errore: La cartella '{multipliers_folder}' non esiste.")
        return []
    # Raccoglie i dati di tutti i file
    multipliers_data = []
    for filename in sorted(os.listdir(multipliers_folder)):  # Ordina i file per nome
        input_path = os.path.join(multipliers_folder, filename)
        if os.path.isfile(input_path):
            data = circuits_analizer(input_path)
            data["pda"] = data["area"] * data["power"] * data["delay"]
            multipliers_data.append(data)
    return multipliers_data




def merge_results(results, mean_ae_results):
    """
    Combina i dati di results e mean_ae_results sulla base del campo 'file'.
    Restituisce una lista di dizionari con tutti i campi unificati.
    """
    merged = []

    # Creiamo un dizionario per accesso rapido ai mean_ae_results
    mean_ae_dict = {item["file"]: item for item in mean_ae_results}

    for r in results:
        file_name = r["file"]
        combined = dict(r)  # copia dei dati base (area, power, delay)
        if file_name in mean_ae_dict:
            combined.update(mean_ae_dict[file_name])  # aggiunge mean_ae e mean_ae_cnn
        else:
            print(f"Warning: file {file_name} not found.")
        merged.append(combined)

    return merged


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise NotImplementedError("Usage: script.py <multipliers_folder> <bitwidth> <output_path>")

    multipliers_folder = sys.argv[1]
    bitwidth = int(sys.argv[2])
    output_path = sys.argv[3]


    results = analyze_multipliers(multipliers_folder)
    mean_ae_results = create_matrices(multipliers_folder, bitwidth, output_path, results)

    os.remove("sub_x_pat_multiplier.py")

    if results and mean_ae_results:
        for data in results:
            print(data)
        print(mean_ae_results)

        merged_results = merge_results(results, mean_ae_results)

        
        fieldnames = ["file", "area", "power", "delay", "pda", "mean_ae", "mean_ae_cnn","max_ae"]

    output_filename = "circuits_area_power.csv"
    output_full_path_file = os.path.join(output_path, output_filename)

    with open(output_full_path_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in merged_results:
            writer.writerow(data)

    

    
    

