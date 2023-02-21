# -*- coding: iso-8859-15 -*-

import json, socket
from random import randint, randrange

import datetime
import time
import math

""" def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer) """





""" print("input file number")
file_number = raw_input()
print file_number
fname = "./log_status/data"+file_number+".txt"
dec_fname = "./log_status/data"+str(file_number)+"_decoded.txt"
print fname """
max_vala = float('-inf')
min_vala = float('inf')
max_valv = float('-inf')
min_valv = float('inf')
max_vall = float('-inf')
min_vall = float('inf')



while True:
  print("input file number")
  file_number = raw_input()
  print file_number
  fname = "./log_status/data"+file_number+".txt"
  decv_fname = "./log_status/data"+str(file_number)+"_decodedv.txt"
  deca_fname = "./log_status/data"+str(file_number)+"_decodeda.txt"
  decl_fname = "./log_status/data"+str(file_number)+"_decoded_label.txt"

  norma_fname = "./log_status/data"+str(file_number)+"_norma.txt"
  normv_fname = "./log_status/data"+str(file_number)+"_normv.txt"
  norml_fname = "./log_status/data"+str(file_number)+"_norml.txt"
  print fname
  try:
    
    f = open(fname, "r")
    fv_d = open(decv_fname, "r")
    fa_d = open(deca_fname, "r")
    fl_d = open(decl_fname, "r")


    Lines = fa_d.readlines()
    count = 0
    for line in Lines:
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #print line
        line = line[:-1]

        linee = '{"line": ['+line+']}'
        
        read_line = json.loads(linee)
        #print type(read_line["line"][55])
        """ max = max(read_line["line"])
        min = min(read_line["line"]) """
        #print "\n\n"
        a = read_line["line"]
        #print a 
        #print max(a)
        if max(a)>max_vala:
            max_vala = max(a)
        if min(a)<min_vala:
            min_vala = min(a)
        #print min(a)
        #print len(a)
        #print "\n\n"
        #norm = [round((float(i)-min(a))/(max(a)-min(a)),5) for i in a]
        """ for j in range(0,len(a)): 
            print (a[j]-min(a))/(max(a)-min(a)) """

    Lines = fv_d.readlines()
    count = 0
    for line in Lines:
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #print line
        line = line[:-1]

        linee = '{"line": ['+line+']}'
        
        read_line = json.loads(linee)
        #print type(read_line["line"][55])
        """ max = max(read_line["line"])
        min = min(read_line["line"]) """
        #print "\n\n"
        a = read_line["line"]
        #print a 
        print max(a)
        if max(a)>max_valv:
            max_valv = max(a)
        if min(a)<min_valv:
            min_valv = min(a)
        print min(a)
        #print len(a)
        #print "\n\n"
        #norm = [round((float(i)-min(a))/(max(a)-min(a)),5) for i in a]
        """ for j in range(0,len(a)): 
            print (a[j]-min(a))/(max(a)-min(a)) """
        

    
    f.close()
    fv_d.close()
    fa_d.close()
    fl_d.close()


    
      
				
				
    
  except socket.timeout:
    b=1
  try:
    
    f = open(fname, "r")
    fv_d = open(decv_fname, "r")
    fa_d = open(deca_fname, "r")
    fl_d = open(decl_fname, "r")

    fna = open(norma_fname, "w")
    fnv = open(normv_fname, "w")


    Lines = fv_d.readlines()
    count = 0
    for line in Lines:
        #print "\n"+str(max_valv)+" ,"+str(min_valv)
        
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #print line
        line = line[:-1]

        linee = '{"line": ['+line+']}'
        
        read_line = json.loads(linee)        
        a = read_line["line"]
        #print a 
        norm = [round((float(i)-min_valv)/(max_valv-min_valv),5) for i in a]
        #print norm
        td = str(norm)+"\n"
        td = td.replace('[', '')
        td = td.replace(']', '')
        fnv.write(td)
    Lines = fa_d.readlines()
    count = 0
    for line in Lines:
        #print "\n"+str(max_valv)+" ,"+str(min_valv)
        
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #print line
        line = line[:-1]

        linee = '{"line": ['+line+']}'
        
        read_line = json.loads(linee)        
        a = read_line["line"]
        #print a 
        norm = [round((float(i)-min_vala)/(max_vala-min_vala),5) for i in a]
        #print norm
        td = str(norm)+"\n"
        td = td.replace('[', '')
        td = td.replace(']', '')
        fna.write(td)
        
    
    f.close()
    fv_d.close()
    fa_d.close()
    fl_d.close()

    fna.close()
    fnv.close()

    """ time_p = round(data['time_counter']*10)
    print time_p/10 """
    
    ## Generate a new position each 2 [s]
    # a new message is received each 0.1[s]
      
				
				
    
  except socket.timeout:
    b=1


#c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\normalize.py"

#send position of the ball with the indices to monitor and log the data