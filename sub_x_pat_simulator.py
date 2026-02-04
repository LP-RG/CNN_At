import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import os
heat_maps_path = "heat_maps/npy_matrix/8_resnet_20"
def get_prob_matrix():
    heat_map_files = sorted([f for f in os.listdir(heat_maps_path) if f.endswith(".npy")])
    cumulative_heatmap = None
    for f in heat_map_files:
        matrix = np.load(os.path.join(heat_maps_path, f))
        matrix =matrix.sum(axis = 0)
        if cumulative_heatmap is None:
            cumulative_heatmap = matrix.copy()
        else:
            cumulative_heatmap += matrix

    # opzionale: normalizza per renderla una distribuzione di probabilità
    prob_matrix = cumulative_heatmap / np.sum(cumulative_heatmap)
    return prob_matrix

def multiplier_test(bit_width, filename,sub_x_pat_multiplier):
    scrumbled_res = np.zeros(((2**bit_width),(2**bit_width)))
    for i in range(0,(2**bit_width)):
        for y in range(0,(2**bit_width)):
            scrumbled_res[i][y] = sub_x_pat_multiplier.approx_mult(i,y)
    np.save(filename,scrumbled_res)


def get_mult_caracteristics(bit_width, filename,sub_x_pat_multiplier):
    #mean_re = 0
    #max_re = 0
    max_error = 0
    mean_ae = 0
    mean_ae_cnn = 0
    output_exact_array = []
    #relative_error_array = []
    max_diff = -np.inf
    #min_diff = np.inf
    prob_matrix = get_prob_matrix()
    for i in range(0,2**bit_width):
        for y in range(0,2**bit_width):
            scrumbled = sub_x_pat_multiplier.approx_mult(i,y)
            exact = i * y
            diff_no_abs = (scrumbled - exact)
            #if(diff_no_abs > max_diff):
            #    max_diff = diff_no_abs
            #if(diff_no_abs < min_diff):
            #    min_diff = diff_no_abs
            diff = abs(diff_no_abs)
            #re = diff / max(1,abs(exact))
            output_exact_array.append(exact)
            #relative_error_array.append(re * 100)
            #mean_re += re
            mean_ae += diff
            mean_ae_cnn += diff * prob_matrix[i][y]
            """if(re > max_re):
                max_re = re"""
            if(diff > max_error):
                max_error = diff
    """print(f"mult : {filename}")
    print(f"mean absolute error: {(mean_ae / ((2**bit_width)*(2**bit_width)))}")
    print(mean_ae_cnn)"""
    return (mean_ae / ((2**bit_width)*(2**bit_width))), mean_ae_cnn, max_error
    """mean_re = (mean_re / ((2**bit_width)*(2**bit_width)))
    print(f"mean relative error: {mean_re}")
    print(f"max relative error: {max_re}")
    print(f"max_error: {max_diff}")
    print(f"min_error: {min_diff}")"""
def execute_save(bit_width, filename):
    if 'sub_x_pat_multiplier' in sys.modules:
        sub_x_pat_multiplier = importlib.reload(sys.modules['sub_x_pat_multiplier'])
    else:
        import sub_x_pat_multiplier
    mean_ae, mean_ae_cn, max_error = get_mult_caracteristics(bit_width,filename,sub_x_pat_multiplier)
    multiplier_test(bit_width,filename,sub_x_pat_multiplier)
    return mean_ae, mean_ae_cn,max_error