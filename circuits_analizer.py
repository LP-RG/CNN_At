from vpadanalyzer.synthesis import Synthesis
import sub_xpat_circuits_generator
import os

multipliers_folder = "./mult_folder"


def circuits_analizer(input_path):
    area = Synthesis.area(input_path)
    power = Synthesis.power(input_path)
    delay = Synthesis.delay(input_path)
    print(f"File: {input_path}, Area = {area}, Power = {power}, Delay = {delay}")
    return area

def analyze_multipliers(multipliers_folder):
    min_area = float('inf')  
    file_with_min_area = None

    if not os.path.isdir(multipliers_folder):
        print(f"Errore: La cartella '{multipliers_folder}' non esiste.")
        return
    
    for filename in os.listdir(multipliers_folder):
        input_path = os.path.join(multipliers_folder, filename)
        if os.path.isfile(input_path):
            current_area = circuits_analizer(input_path)
            if current_area < min_area:
                min_area = current_area
                file_with_min_area_full_path  = input_path

    if file_with_min_area_full_path:
        # Costruisci il nuovo percorso per il file rinominato
        new_name = "mul_best.v"
        new_path = os.path.join(multipliers_folder, new_name)

        # Rinominare il file
        try:
            os.rename(file_with_min_area_full_path, new_path)
            print(f"\n--- Analisi Completata ---")
            print(f"Il file con l'area minore era: **{os.path.basename(file_with_min_area_full_path)}**")
            print(f"È stato rinominato in: **{new_name}** con un'area di: **{min_area}**")
        except OSError as e:
            print(f"Errore durante la ridenominazione del file {os.path.basename(file_with_min_area_full_path)}: {e}")
    else:
        print(f"\nNessun file processato nella cartella: {multipliers_folder}")
    return new_path


if __name__ == "__main__":
    mult_path = analyze_multipliers(multipliers_folder)
    sub_xpat_circuits_generator.generate_approx_mult_function(mult_path,4)

