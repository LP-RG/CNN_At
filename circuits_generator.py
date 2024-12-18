import re
destination_file = open("scrumbled_multiplier8192.py", "w")
function_definition = "def mul_i16_o16_wce1024(a: int, b: int) -> int:\n"
parsing_row = ("\tpi07, pi06, pi05, pi04, pi03, pi02, pi01, pi00 = [int(bit) for bit in bin(a)[2:].zfill(8)]\n")
parsing_row += ("\tpi15, pi14, pi13, pi12, pi11, pi10, pi09, pi08 = [int(bit) for bit in bin(b)[2:].zfill(8)]\n")
destination_file.write(function_definition)
destination_file.write(parsing_row)
# Apri il file Verilog in modalità lettura
with open("mul_i16_o16_wce8192.v", "r") as file:
    # Scorri riga per riga
    for line in file:
        # Rimuove eventuali spazi o newline
        if "assign" in line:
            if("po" in line):                
                line_new = "\t" + line[9:].replace("|", "or").replace("&", "and").replace("~", "not ").replace("1'h0", "0").replace("1'h1", "1").replace(";","")
            else:
                line_new = "\t"+ line[9:].replace("|", "or").replace("&", "and").replace("~", "not ").replace("1'h0", "0").replace("1'h1", "1").replace(";","")

            destination_file.write(line_new)
destination_file.write("\tbits = [int(po15), int(po14), int(po13), int(po12), int(po11), int(po10), int(po09),int( po08),int(po07), int(po06), int(po05), int(po04), int(po03), int(po02),int( po01), int(po00)]\n") 
destination_file.write("\tbit_string = ''.join(str(bit) for bit in bits)\n")
destination_file.write("\tresult = int(bit_string, 2)\n")
destination_file.write("\treturn result\n")
