import os
import random
import numpy as np
import multiplier_outputs_plotting

multiplier_folder = "./random_multipliers"

if not os.path.exists(multiplier_folder):
    os.mkdir(multiplier_folder)

def generate_random_multiplier(bit_width,iter_number):
    #generate exact multiplier
    exact_multiplier = np.outer(np.arange(bit_width**2), np.arange(bit_width**2))
    random_multiplier = np.zeros((bit_width**2,bit_width**2))
    #for each cell generate a random number and a random sign (do not generate negative numbers)
    random_re_threshold = random.randint(0, 100)
    for i in range(0,bit_width ** 2):
        for j in range(0,bit_width ** 2):
            exact_output = exact_multiplier[i][j]
            random_number = random.randint(0,np.round((random_re_threshold * exact_output) / 100))
            
            random_sign = random.randint(0,1)
            if(random_sign == 1):
                random_multiplier[i][j] = exact_output + random_number
            else:
                random_multiplier[i][j] = exact_output - random_number
    mult_number =  (str(iter_number) + "_" + str(random_re_threshold))
    np.save(multiplier_folder + "/" + mult_number + ".npy", random_multiplier)
    # multiplier_outputs_plotting.mult_output_plotting(mult_number, random_multiplier, multiplier_folder + "/" + multiplier_folder + "_plot/")
    #criteria to generate multipliers >>


def generate_random_multipliers(bit_width, total_number):
    for iteration in range(total_number):
        generate_random_multiplier(bit_width,iteration)

#TODO itero sulla cartella dei moltiplicatori random 
# per ogni moltiplicatore creo il file res.npy nella cartella principale di AAT
# creo un dizionario con nome multiplier: best_accuracy

if __name__ == "__main__":
    generate_random_multipliers(4, 10)
