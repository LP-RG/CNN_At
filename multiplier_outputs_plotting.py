import sys
import numpy as np
import matplotlib.pyplot as plt

output_plots_path = "./outputs_plot/"
def mult_output_plotting(mult_name, matrix, output_plots_path):
    plt.rcParams.update({'font.size': 20})
    plt.imshow(matrix, cmap='Reds', origin='upper')# Modifica per avere i ticks da -128 a 128
    plt.xticks(np.arange(matrix.shape[1], step=np.round(matrix.shape[1] / 5)))
    plt.yticks(np.arange(matrix.shape[0], step=np.round(matrix.shape[1] / 5)))
    plt.xlabel("Weights")
    plt.ylabel("Inputs") 
    plt.colorbar()  
    plt.savefig(output_plots_path + mult_name + ".png", bbox_inches='tight')
    plt.close()

def mult_AE_plotting(mult_name, matrix):
    n_inputs = matrix.shape[0]
    exact_outputs = np.outer(np.arange(n_inputs), np.arange(n_inputs))
    differences = np.abs(matrix - exact_outputs)
    plt.rcParams.update({'font.size': 20})
    plt.imshow(differences, cmap='Reds', origin='upper', vmax=(np.max(differences)))
    plt.xticks(np.arange(differences.shape[1], step=np.round(matrix.shape[1] / 5)))
    plt.yticks(np.arange(differences.shape[0], step=np.round(matrix.shape[1] / 5)))
    plt.xlabel("Weights")
    plt.ylabel("Inputs") 
    plt.colorbar() 
    plt.savefig(output_plots_path + mult_name + "_AE.png", bbox_inches='tight')
    plt.close()

def mult_RE_plotting(mult_name, matrix):
    n_inputs = matrix.shape[0]
    exact_outputs = np.outer(np.arange(n_inputs), np.arange(n_inputs))
    differences = np.abs(matrix - exact_outputs) * 100 /  np.where(exact_outputs == 0, 1, exact_outputs)
    plt.rcParams.update({'font.size': 20})
    plt.imshow(differences, cmap='Greens', origin='upper', vmax=(np.max(differences)))
    plt.xticks(np.arange(differences.shape[1], step=np.round(matrix.shape[1] / 5)))
    plt.yticks(np.arange(differences.shape[0], step=np.round(matrix.shape[1] / 5)))
    plt.xlabel("Weights")
    plt.ylabel("Inputs") 
    plt.colorbar() 
    plt.savefig(output_plots_path + mult_name + "_RE.png", bbox_inches='tight')
    plt.close()

def plots(mult_name,file_path):
    outputs = np.load(file_path)
    mult_output_plotting(mult_name=mult_name, matrix=outputs, output_plots_path=output_plots_path)
    mult_AE_plotting(mult_name=mult_name, matrix=outputs)
    mult_RE_plotting(mult_name=mult_name, matrix=outputs)

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        raise NotImplementedError
    plots(sys.argv[1],sys.argv[2])
