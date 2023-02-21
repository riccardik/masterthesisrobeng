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
""" max_vala = float('-inf')
min_vala = float('inf')
max_valv = float('-inf')
min_valv = float('inf')
max_vall = float('-inf')
min_vall = float('inf') """



while True:
  print("input file number")
  file_number = raw_input()
  print file_number
  fname = "./log_status/data"+file_number+".txt"
  minmax_fname = "./log_status/data"+file_number+"minmax.txt"
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

    line = Lines[0][:-1]
    linee = '{"line": ['+line+']}'
    #print linee
    
    read_line = json.loads(linee)['line']
    #print str(read_line)
    print len(read_line)
    size = len(read_line)

    max_vala = [float('-inf') for x in range(size)] 
    #print max_vala
    min_vala = [float('inf') for x in range(size)] 
    #print min_vala

    
    Lines = fv_d.readlines()    
    line = Lines[0][:-1]
    linee = '{"line": ['+line+']}'
    #print linee
    
    read_line = json.loads(linee)['line']
    #print str(read_line)
    print len(read_line)
    size = len(read_line)

    max_valv = [float('-inf') for x in range(size)] 
    #print max_valv
    min_valv = [float('inf') for x in range(size)] 
    #print min_valv

    Lines = fl_d.readlines()    
    line = Lines[0][:-1]
    linee = '{"line": ['+line+']}'
    #print linee
    
    read_line = json.loads(linee)['line']
    #print str(read_line)
    print len(read_line)
    size = len(read_line)

    max_vall = [float('-inf') for x in range(size)] 
    #print max_valv
    min_vall = [float('inf') for x in range(size)] 
    
    
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

        for j in range(0,len(a)):
            if a[j]>max_vala[j]:
                max_vala[j] = a[j]
            if a[j]<min_vala[j]:
                min_vala[j] = a[j]
        """ if max(a)>max_vala:
            max_vala = max(a)
        if min(a)<min_vala:
            min_vala = min(a) """
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
        for j in range(0,len(a)):
            if a[j]>max_valv[j]:
                max_valv[j] = a[j]
            if a[j]<min_valv[j]:
                min_valv[j] = a[j]
        """ print max(a)
        if max(a)>max_valv:
            max_valv = max(a)
        if min(a)<min_valv:
            min_valv = min(a)
        print min(a) """
        #print len(a)
        #print "\n\n"
        #norm = [round((float(i)-min(a))/(max(a)-min(a)),5) for i in a]
        """ for j in range(0,len(a)): 
            print (a[j]-min(a))/(max(a)-min(a)) """
    
    
    Lines = fl_d.readlines()
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
        for j in range(0,len(a)):
            if a[j]>max_vall[j]:
                max_vall[j] = a[j]
            if a[j]<min_vall[j]:
                min_vall[j] = a[j]
        """ print max(a)
        if max(a)>max_valv:
            max_valv = max(a)
        if min(a)<min_valv:
            min_valv = min(a)
        print min(a) """
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
    fnl = open(norml_fname, "w")


    Lines = fv_d.readlines()
    count = 0
    #fnv.write("[")
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
        #norm = [round((float(i)-min_valv)/(max_valv-min_valv),5) for i in a]
        norm = [0 for x in range(len(a))] 
        for j in range(0,len(a)):
            norm[j] = round((float(a[j])-min_valv[j])/(max_valv[j]-min_valv[j]),5)
            
        #print norm
        td = str(norm)+"\n"
        td = td.replace('[', '')
        td = td.replace(']', '')
        fnv.write(td)
    #fnv.write("]")
    Lines = fa_d.readlines()
    count = 0
    #fna.write("[")
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
        #norm = [round((float(i)-min_vala)/(max_vala-min_vala),5) for i in a]
        norm = [0 for x in range(len(a))] 
        for j in range(0,len(a)):
            norm[j] = round((float(a[j])-min_vala[j])/(max_vala[j]-min_vala[j]),5)
        #print norm
        td = str(norm)+"\n"
        td = td.replace('[', '')
        td = td.replace(']', '')
        fna.write(td)
    #fna.write("]")

    #fnl.write("[")
    Lines = fl_d.readlines()
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
        """ print a 
        print min_vall
        print max_vall 
        print "\n" """
        #norm = [round((float(i)-min_vala)/(max_vala-min_vala),5) for i in a]
        norm = [0 for x in range(len(a))] 
        for j in range(0,len(a)):
            #print a[j]
            if min_vall[j] == max_vall[j] :
                norm[j] = 0
            else:
                norm[j] = round((float(a[j])-min_vall[j])/(max_vall[j]-min_vall[j]),5)
            #print norm[j]
        #print norm
        td = str(norm)+"\n"
        td = td.replace('[', '')
        td = td.replace(']', '')
        fnl.write(td)
    #fnl.write("]")
        
    
    f.close()
    fv_d.close()
    fa_d.close()
    fl_d.close()

    fna.close()
    fnv.close()
    """ print("min_vala = "+str(min_vala)+"\n")
    print("min_valv = "+str(min_valv)+"\n")
    print("min_vall = "+str(min_vall)+"\n")
    print("max_vala = "+str(max_vala)+"\n")
    print("max_valv = "+str(max_valv)+"\n")
    print("max_vall = "+str(max_vall)+"\n") """
    variancea = [math.sqrt(math.pow(a + b,2)) for a, b in zip(min_vala, max_vala)]
    variancev = [math.sqrt(math.pow(a + b,2)) for a, b in zip(min_valv, max_valv)]
    variancel = [math.sqrt(math.pow(a + b,2)) for a, b in zip(min_vall, max_vall)]

    towrite = {
        "min_vala":min_vala,
        "min_valv":min_valv,
        "min_vall":min_vall,
        "max_vala":max_vala,
        "max_valv":max_valv,
        "max_vall":max_vall,
        "variancea":variancea,
        "variancev":variancev,
        "variancel":variancel,
        
            }

    data_dump = json.dumps(towrite)
    #print self.fname
    fnminmax = open(minmax_fname, "w")
    #print data_dump
    #data_dump = "ciao"+"\n"
    fnminmax.write(data_dump) 
    fnminmax.close()




    """ time_p = round(data['time_counter']*10)
    print time_p/10 """
    
    ## Generate a new position each 2 [s]
    # a new message is received each 0.1[s]
      
				
				
    
  except socket.timeout:
    b=1


#c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\normalize_bycol.py"

#send position of the ball with the indices to monitor and log the data