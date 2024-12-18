import numpy as np
import matplotlib.pyplot as plt

def mul_i16_o16_wce1024(a: int, b: int) -> int:
	# Converte i numeri in binari a 8 bit
	pi07, pi06, pi05, pi04, pi03, pi02, pi01, pi00 = [int(bit) for bit in bin(a)[2:].zfill(8)]
	pi15, pi14, pi13, pi12, pi11, pi10, pi09, pi08 = [int(bit) for bit in bin(b)[2:].zfill(8)]
	_072_ =  not pi14
	_073_ = pi12  and  pi06
	_074_ = pi05  and  pi13
	_075_ = _074_  and  _073_
	_076_ = _075_  and   not (_072_)
	_077_ = pi05  and  pi14
	_078_ = _072_ if _075_ else _077_
	_079_ = pi04  and  pi15
	_080_ = _079_  and  _078_
	_081_ =  not (_080_  or  _076_)
	_082_ = pi11  and  pi07
	_083_ =  not _082_
	_084_ = _074_ ^ _073_
	_085_ = _084_  and   not (_083_)
	_086_ = pi13  and  pi06
	_087_ = pi12  and  pi07
	_088_ = _087_ ^ _086_
	_089_ = _088_  and  _085_
	_090_ = _088_ ^ _085_
	_091_ = _079_ ^ _078_
	_092_ = _091_  and  _090_
	_093_ =  not (_092_  or  _089_)
	_094_ = pi13  and  pi07
	_095_ = pi05  and  pi15
	_096_ = _087_  and  _086_
	_097_ = pi06  and  pi14
	_098_ = _072_ if _096_ else _097_
	_099_ = _098_ ^ _095_
	_100_ = _099_ ^ _094_
	_101_ =  not (_100_ ^ _093_)
	_102_ =  not (_101_ ^ _081_)
	_103_ =  not (_091_ ^ _090_)
	_104_ = _084_ ^ _083_
	_105_ = pi11  and  pi05
	_106_ = pi09  and  pi07
	_107_ = _106_  and  _105_
	_108_ =  not (_107_  and  pi06)
	_109_ =  not (pi11  and  pi06)
	_110_ = pi06 if _107_ else _109_
	_111_ =  not (pi13  and  pi04)
	_112_ =  not (pi05  and  pi12)
	_113_ = pi10  and  pi07
	_114_ =  not (_113_ ^ _112_)
	_115_ =  not (_114_ ^ _111_)
	_116_ = _115_  and   not (_110_)
	_117_ = _108_  and   not (_116_)
	_118_ = _117_  or  _104_
	_119_ = _113_  and   not (_112_)
	_120_ = _114_  and   not (_111_)
	_121_ = _120_  or  _119_
	_122_ = pi04  and  pi14
	_123_ = _122_ ^ _121_
	_124_ = pi03  and  pi15
	_125_ = _124_ ^ _123_
	_126_ = _117_ ^ _104_
	_127_ = _126_  and  _125_
	_128_ = _118_  and   not (_127_)
	_129_ = _128_  or  _103_
	_130_ = _122_  and  _121_
	_131_ = _124_  and  _123_
	_132_ = _131_  or  _130_
	_133_ = _128_ ^ _103_
	_134_ = _133_  and  _132_
	_135_ = _129_  and   not (_134_)
	_136_ =  not (_135_ ^ _102_)
	_137_ = _133_ ^ _132_
	_138_ =  not (_126_ ^ _125_)
	_139_ = _115_ ^ _110_
	_140_ = _106_ ^ _105_
	_141_ =  not (pi11  and  pi04)
	_142_ =  not (pi08  and  pi06)
	_143_ = _106_  and   not (_142_)
	_144_ = _143_  or   not (_141_)
	_145_ =  not (pi01  and  pi15)
	_146_ = pi02  and  pi14
	_147_ = _145_  and   not (_146_)
	_148_ = _144_  and   not (_147_)
	_000_ =  not (_148_  and  _140_)
	_001_ = _148_  or  _140_
	_002_ =  not (pi13  and  pi03)
	_003_ =  not (pi12  and  pi04)
	_004_ = pi10  and  pi06
	_005_ =  not (_004_ ^ _003_)
	_006_ = _005_ ^ _002_
	_007_ = _001_  and   not (_006_)
	_008_ = _000_  and   not (_007_)
	_009_ = _008_  or  _139_
	_010_ = _004_  and   not (_003_)
	_011_ = _005_  and   not (_002_)
	_012_ = _011_  or  _010_
	_013_ = pi03  and  pi14
	_014_ = _013_ ^ _012_
	_015_ = pi02  and  pi15
	_016_ = _015_ ^ _014_
	_017_ = _008_ ^ _139_
	_018_ = _017_  and  _016_
	_019_ = _009_  and   not (_018_)
	_020_ = _019_  or  _138_
	_021_ = _013_  and  _012_
	_022_ = _015_  and  _014_
	_023_ = _022_  or  _021_
	_024_ = _019_ ^ _138_
	_025_ = _024_  and  _023_
	_026_ = _020_  and   not (_025_)
	_027_ = _026_  and   not (_137_)
	_028_ = _017_ ^ _016_
	_029_ = _146_  and   not (_145_)
	_030_ =  not (pi03  and  pi10)
	_031_ =  not (_030_  or  _112_)
	_032_ = _031_  and   not (_147_)
	_033_ = _032_  or  _029_
	_034_ = _033_  or  _028_
	_035_ =  not (_024_ ^ _023_)
	_036_ = _035_  or   not (_034_)
	_037_ = _137_  and   not (_026_)
	_038_ = _036_  and   not (_037_)
	_039_ = _038_  or  _027_
	po12 =  not (_039_ ^ _136_)
	_040_ = _100_  and   not (_093_)
	_041_ = _101_  and   not (_081_)
	_042_ =  not (_041_  or  _040_)
	_043_ =  not (_096_  and  pi14)
	_044_ =  not (_098_  and  _095_)
	_045_ =  not (_044_  and  _043_)
	_046_ = _099_  and  _094_
	_047_ = pi14  and  pi07
	_048_ = pi15  and  pi06
	_049_ = _048_ ^ _047_
	_050_ = _049_ ^ _046_
	_051_ = _050_ ^ _045_
	_052_ = _051_  and   not (_042_)
	_053_ = _042_  and   not (_051_)
	_054_ = _053_  or  _052_
	_055_ = _102_  and   not (_135_)
	_056_ = _039_  or   not (_136_)
	_057_ = _056_  and   not (_055_)
	po13 = _057_ ^ _054_
	_058_ = _049_  and  _046_
	_059_ = _050_  and  _045_
	_060_ = _059_  or  _058_
	_061_ =  not (pi15  and  pi07)
	_062_ =  not (_061_  or  _097_)
	_063_ = _062_ ^ _060_
	_064_ = _052_  or  _055_
	_065_ = _056_  and   not (_064_)
	_066_ =  not (_065_  or  _053_)
	po14 = _066_ ^ _063_
	_067_ = _066_  and  _063_
	_068_ = _048_  and  _047_
	_069_ = _062_  and  _060_
	_070_ = _069_  or  _068_
	po15 = _070_  or  _067_
	po09 =  not (_033_ ^ _028_)
	po10 =  not (_035_ ^ _034_)
	_071_ = _037_  or  _027_
	po11 = _071_ ^ _036_
	po00 = 0
	po01 = 0
	po02 = 0
	po03 = 0
	po04 = 0
	po05 = 0
	po06 = 0
	po07 = 1
	po08 = 1
	bits = [int(po15), int(po14), int(po13), int(po12), int(po11), int(po10), int(po09),int( po08),int(po07), int(po06), int(po05), int(po04), int(po03), int(po02),int( po01), int(po00)]
	bit_string = ''.join(str(bit) for bit in bits)
	result = int(bit_string, 2)
	return result

def plot_matrix_with_values(matrix, filename):
    plt.figure(figsize=(15, 15))  # Dimensione più ridotta per una matrice 255x255

    # Normalizza la matrice per la mappa dei colori (colore chiaro per zero)
    vmin = np.min(matrix)
    vmax = np.max(matrix)
    
    # Mostra la matrice con la scala dei colori diverging
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

    # Barra dei colori
    plt.colorbar(label='Value')

    # Etichette degli assi
    plt.title('Matrix Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # Salva il grafico in un file
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Salva il grafico come file PNG
    plt.close()
    
def multiplier_test():
    exact_res_matrix = np.zeros((256,256))
    custom_res_matrix = np.zeros((256,256))
    res_diff_matrix = np.zeros((256,256))
    for x in range(0,256):
        for y in range(0,256):
            res_exact = x * y
            res_scrumbled = mul_i16_o16_wce1024(x,y)
            res_diff = res_exact - res_scrumbled
            if(res_diff > 1024 or res_diff < -1024):
                print("error")
                return None
            else:
                exact_res_matrix[x][y]=res_exact
                custom_res_matrix[x][y]=res_scrumbled
                res_diff_matrix[x][y]= res_diff
    return exact_res_matrix,custom_res_matrix,res_diff_matrix

exact_res_matrix,custom_res_matrix,res_diff_matrix = multiplier_test()
np.save('res_diff_matrix.npy', res_diff_matrix)