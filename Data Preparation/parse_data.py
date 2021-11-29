# -*- coding: iso-8859-15 -*-

import json, socket
from random import randint, randrange

import datetime
import time
import math

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)





""" print("input file number")
file_number = raw_input()
print file_number
fname = "./log_status/data"+file_number+".txt"
dec_fname = "./log_status/data"+str(file_number)+"_decoded.txt"
print fname """



while True:
  print("input file number")
  file_number = raw_input()
  print file_number
  print("(p)osition, (pf)pos/Force, (pfv)pos/force/vel")
  to_parse = raw_input()

  fname = "./log_status/data"+file_number+".txt"
  decv_fname = "./log_status/data"+str(file_number)+"_decodedv.txt"
  deca_fname = "./log_status/data"+str(file_number)+"_decodeda.txt"
  decl_fname = "./log_status/data"+str(file_number)+"_decoded_label.txt"
  print fname
 

  try:
    
    f = open(fname, "r")
    fv_d = open(decv_fname, "w")
    fa_d = open(deca_fname, "w")
    fl_d = open(decl_fname, "w")


    Lines = f.readlines()
    count = 0
    for line in Lines:
        count += 1
        #print("Line{}: {}".format(count, line.strip()))
        #print line
        read_line = json.loads(line)
        #print read_line["rotation_kidney"][0]
        if (to_parse=='p'):
          to_dumpv = str(read_line["arr_pos_v"]) +"\n"
          to_dumpa = str(read_line["arr_pos_a"]) +"\n"
        elif (to_parse=='pf'):
          to_dumpv = str(read_line["arr_force_v"]) +","+ str(read_line["arr_pos_v"]) +"\n"
          to_dumpa = str(read_line["arr_force_a"]) +","+ str(read_line["arr_pos_a"]) +"\n"
        elif (to_parse=='pfv'):
          to_dumpv = str(read_line["arr_force_v"]) +","+ str(read_line["arr_pos_v"]) +"\n"
          to_dumpa = str(read_line["arr_force_a"]) +","+ str(read_line["arr_pos_a"]) +"\n"
        #to_dumpv = str(read_line["arr_pos_v"]) +"\n"
        #to_dumpa = str(read_line["arr_force_a"]) +","+ str(read_line["arr_pos_a"]) +"\n"
        #to_dumpa = str(read_line["arr_pos_a"]) +"\n"
        posxyz = "[["+str(read_line["position_kidney"][0][0])+","+str(read_line["position_kidney"][0][1])+","+str(read_line["position_kidney"][0][2])+"]]"
        to_dumpl = posxyz +","+ str(read_line["rotation_kidney"][0]) +"\n"
        #print to_dumpv
        tdv = ""
        tda = ""
        tdl = ""
        #print "\n\n"

        to_dumpa = to_dumpa.replace('[', '')
        to_dumpa = to_dumpa.replace(']', '')
        to_dumpa = to_dumpa.replace(' ', '') 
        to_dumpv = to_dumpv.replace('[', '')
        to_dumpv = to_dumpv.replace(']', '')
        to_dumpv = to_dumpv.replace(' ', '') 
        to_dumpl = to_dumpl.replace('[', '')
        to_dumpl = to_dumpl.replace(']', '')
        to_dumpl = to_dumpl.replace(' ', '') 

        for i in to_dumpv.split(','):
          
          tdv = tdv+"{:.5f}".format(float(i))+","
        tdv = tdv[:-1]
        tdv = tdv+"\n"
        for i in to_dumpa.split(','):
          tda = tda+"{:.5f}".format(float(i))+","
        tda = tda[:-1]
        tda = tda+"\n"
        for i in to_dumpl.split(','):
          tdl = tdl+"{:.5f}".format(float(i))+","
        tdl = tdl[:-1]
        tdl = tdl+"\n"
        #to_dumpv = "["+to_dumpv+"]"
        #print tdv
        #print to_dumpv
        #print to_dumpv[1]
        #print [round(float(i), 6) for i in to_dumpv.split(',')]
        #print to_dumpv
        #input_raw()
        #print to_dump
        fv_d.write(tdv)
        fa_d.write(tda)
        fl_d.write(tdl)


    #print(data['position_ball'])
    #print(data['indices_nearest'])
    #print(data['indices_position'])
    """ time_p = round(data['time_counter'])
    to_dump = {
        "position_ball": data["position_ball"],
        "force": data1["force"],
        "time": data["time_counter"],
        "indices_nearest": data["indices_nearest"],
        "indices_position": data["indices_position"],
        "indices_velocity": data["indices_velocity"],
        "initial_ball_position": data1["position"]
    }

    #save on a .txt file the state of the current sample
    data_dump = json.dumps(to_dump)+"\n" """
    
    f.close()
    fv_d.close()
    fa_d.close()
    fl_d.close()

    """ time_p = round(data['time_counter']*10)
    print time_p/10 """
    
    ## Generate a new position each 2 [s]
    # a new message is received each 0.1[s]
      
				
				
    
  except socket.timeout:
    b=1


# c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\parse_data.py"

#send position of the ball with the indices to monitor and log the data