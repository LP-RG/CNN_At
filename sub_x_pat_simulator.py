import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib

def multiplier_test(bit_width, filename,sub_x_pat_multiplier):
    scrumbled_res = np.zeros(((2**bit_width),(2**bit_width)))
    for i in range(0,(2**bit_width)):
        for y in range(0,(2**bit_width)):
            scrumbled_res[i][y] = sub_x_pat_multiplier.approx_mult(i,y)
    np.save(filename,scrumbled_res)


def get_mult_caracteristics(bit_width, filename,sub_x_pat_multiplier):
    mean_re = 0
    max_re = 0
    max_error = 0
    mean_ae = 0
    output_exact_array = []
    relative_error_array = []
    max_diff = -np.inf
    min_diff = np.inf
    for i in range(0,2**bit_width):
        for y in range(0,2**bit_width):
            scrumbled = sub_x_pat_multiplier.approx_mult(i,y)
            exact = i * y
            diff_no_abs = (scrumbled - exact)
            if(diff_no_abs > max_diff):
                max_diff = diff_no_abs
            if(diff_no_abs < min_diff):
                min_diff = diff_no_abs
            diff = abs(diff_no_abs)
            """if(i < 50 and diff > 200):
               print(f"error for inputs :{i} * {y}, error = {diff}")
            if(i < 100 and diff > 400):
               print(f"error for inputs :{i} * {y}, error = {diff}")
            if(i < 150 and diff > 600):
               print(f"error for inputs :{i} * {y}, error = {diff}")
            if(i < 200 and diff > 800):
               print(f"error for inputs :{i} * {y}, error = {diff}")
            if(i < 250 and diff > 1000):
               print(f"error for inputs :{i} * {y}, error = {diff}")"""
            re = diff / max(1,abs(exact))
            output_exact_array.append(exact)
            relative_error_array.append(re * 100)
            mean_re += re
            mean_ae += diff
            if(re > max_re):
                max_re = re
            if(diff > max_error):
                max_error = diff
    print(f"mult : {filename}")
    print(f"mean absolute error: {(mean_ae / ((2**bit_width)*(2**bit_width)))}")
    print(f"max absolute error: {max_error}")
    
    mean_re = (mean_re / ((2**bit_width)*(2**bit_width)))
    print(f"mean relative error: {mean_re}")
    print(f"max relative error: {max_re}")
    print(f"max_error: {max_diff}")
    print(f"min_error: {min_diff}")
def execute_save(bit_width, filename):
    if 'sub_x_pat_multiplier' in sys.modules:
        sub_x_pat_multiplier = importlib.reload(sys.modules['sub_x_pat_multiplier'])
    else:
        import sub_x_pat_multiplier
    get_mult_caracteristics(bit_width,filename,sub_x_pat_multiplier)
    multiplier_test(bit_width,filename,sub_x_pat_multiplier)