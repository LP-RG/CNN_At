import numpy as np

def mul_i16_o16_wce8192(a: int, b: int) -> int:
	pi07, pi06, pi05, pi04, pi03, pi02, pi01, pi00 = [int(bit) for bit in bin(a)[2:].zfill(8)]
	pi15, pi14, pi13, pi12, pi11, pi10, pi09, pi08 = [int(bit) for bit in bin(b)[2:].zfill(8)]
	po10 = not pi15
	_00_ = pi13 and pi07
	_01_ = not (pi15 and pi05)
	_02_ = not (pi12 and pi06)
	_03_ = _02_ or not (_00_)
	_04_ = not (pi14 and pi06)
	_05_ = _03_ and not (_04_)
	_06_ = _05_ ^ _01_
	_07_ = not (_06_ ^ _00_)
	po12 = _07_ ^ po10
	_08_ = _07_ or pi15
	_09_ = _00_ and not (_06_)
	_10_ = not (pi14 and pi07)
	_11_ = pi06 and pi15
	_12_ = not (_11_ ^ _10_)
	_13_ = _12_ ^ _09_
	_14_ = _01_ or not (_05_)
	_15_ = pi14 and not (_03_)
	_16_ = _14_ and not (_15_)
	_17_ = not (_16_ ^ _13_)
	po13 = _17_ ^ _08_
	_18_ = _17_ and _08_
	_19_ = _12_ and _09_
	_20_ = _13_ and not (_16_)
	_21_ = _20_ or _19_
	_22_ = not (pi15 and pi07)
	_23_ = _04_ and not (_22_)
	_24_ = _23_ ^ _21_
	po14 = _24_ ^ _18_
	_25_ = _24_ and _18_
	_26_ = _11_ and not (_10_)
	_27_ = _23_ and _21_
	_28_ = _27_ or _26_
	po15 = _28_ or _25_
	po00 = 0
	po01 = 0
	po02 = 0
	po03 = 0
	po04 = 0
	po05 = 0
	po06 = 0
	po07 = 0
	po08 = 0
	po09 = 0
	po11 = po10
	bits = [int(po15), int(po14), int(po13), int(po12), int(po11), int(po10), int(po09),int( po08),int(po07), int(po06), int(po05), int(po04), int(po03), int(po02),int( po01), int(po00)]
	bit_string = ''.join(str(bit) for bit in bits)
	result = int(bit_string, 2)
	return result


def multiplier_test():
    exact_res_matrix = np.zeros((256,256))
    custom_res_matrix = np.zeros((256,256))
    res_diff_matrix = np.zeros((256,256))
    for x in range(0,256):
        for y in range(0,256):
            res_exact = x * y
            res_scrumbled = mul_i16_o16_wce8192(x,y)
            res_diff = res_exact - res_scrumbled
            if(res_diff > 8192 or res_diff < -8192):
                print("error")
                return None
            else:
                exact_res_matrix[x][y]=res_exact
                custom_res_matrix[x][y]=res_scrumbled
                res_diff_matrix[x][y]= res_diff
    return exact_res_matrix,custom_res_matrix,res_diff_matrix

exact_res_matrix,custom_res_matrix,res_diff_matrix = multiplier_test()
np.save('res_diff_matrix8192.npy', res_diff_matrix)