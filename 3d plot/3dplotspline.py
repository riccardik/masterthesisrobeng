# -*- coding: iso-8859-15 -*-

import json, socket
from random import randint, randrange
import random
import datetime
import time
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
 
def Ry(theta):
  return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
 
def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])



print("input file number")
#file_number = raw_input()
file_number = "0927221920"
# file_number = "0906191003"
# file_number = "0929110030"
#fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"minmax.txt"
# fname = "./log_status/data"+file_number+".txt"

# fname2 = "./log_status/data"+file_number+"predictedpos.txt"

fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+".txt"
fname2 = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposveinregr.txt"
fname2a = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposartregr.txt"
fname3 = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposveinlstm.txt"
fname3a = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposartlstm.txt"


f = open(fname, "r")
f2 = open(fname2, "r")
f2a = open(fname2a, "r")
f3 = open(fname3, "r")
f3a = open(fname3a, "r")
Lines = f.readlines()
Lines2 = f2.readlines()
Lines2a = f2a.readlines()
Lines3 = f3.readlines()
Lines3a = f3a.readlines()
count = 0
j=0
arr = []
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax.set_axis_off()
ax2.set_axis_off()

fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(111, projection='3d')
for j in range(0, len(Lines), 5):
        
    ax.cla()
    ax2.cla()
    ax3.cla()
    read_line = json.loads(Lines[j])
    read_line2 = json.loads(Lines2[j])
    read_line2a = json.loads(Lines2a[j])
    read_line3 = json.loads(Lines3[j])
    read_line3a = json.loads(Lines3a[j])
    posv = read_line['arr_pos_v']
    posa = read_line['arr_pos_a']
    posa2 = read_line2a['arr_pos_v_pred']
    posa3 = read_line3a['arr_pos_v_pred']
    print(read_line['time_counter'])

    posv2 = read_line2['arr_pos_v_pred']
    posv3 = read_line3['arr_pos_v_pred']

    pos_ = read_line['position_kidney']
    rot_ = read_line['rotation_kidney']


    origin=np.asarray([[pos_[0][0]],[pos_[0][1]],[pos_[0][2]]])
    tail1=np.asarray([[1],[0],[0]])
    tail2=np.asarray([[0],[-1],[0]])
    tail3=np.asarray([[0],[0],[1]])

    
    

    # phi = math.pi/2+index12
    phi =  (rot_[0][0]*math.pi)/180
    theta =(rot_[0][1]*math.pi)/180
    psi = (rot_[0][2]*math.pi)/180
    # print("phi =", phi)
    # print("theta  =", theta)
    # print("psi =", psi)


    R = Rz(psi) * Ry(theta) * Rx(phi)
    # R = Rx(phi)
    origin = R*origin
    tail1 = R*tail1
    tail2 = R*tail2
    tail3 = R*tail3
    print(origin)
    print(tail1)
    print(tail2)
    print(tail3)
    # input()

    q1 = ax.quiver(origin[0], origin[1], origin[2], tail1[0], tail1[1], tail1[2], color='red',linestyle='solid')
    q2=ax.quiver(origin[0], origin[1], origin[2], tail2[0], tail2[1], tail2[2], color='blue',linestyle='solid')
    q3 = ax.quiver(origin[0], origin[1], origin[2], tail3[0], tail3[1], tail3[2], color='green',linestyle='solid')
    q1 = ax2.quiver(origin[0], origin[1], origin[2], tail1[0], tail1[1], tail1[2], color='red',linestyle='solid')
    q2=ax2.quiver(origin[0], origin[1], origin[2], tail2[0], tail2[1], tail2[2], color='blue',linestyle='solid')
    q3 = ax2.quiver(origin[0], origin[1], origin[2], tail3[0], tail3[1], tail3[2], color='green',linestyle='solid')
    q1 = ax3.quiver(origin[0], origin[1], origin[2], tail1[0], tail1[1], tail1[2], color='red',linestyle='solid')
    q2=ax3.quiver(origin[0], origin[1], origin[2], tail2[0], tail2[1], tail2[2], color='blue',linestyle='solid')
    q3 = ax3.quiver(origin[0], origin[1], origin[2], tail3[0], tail3[1], tail3[2], color='green',linestyle='solid')

    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = []    
    for i in range(len(posv)):
        # print(i)
        ax.scatter(posv[i][0], posv[i][1], posv[i][2], c="blue", s=20, label=str(i))
        # ax.text(posv[i][0], posv[i][1], posv[i][2],  '%s' % (str(i)), size=10, zorder=1,      color='k') 
        if (i>=16 and i<=24) :
            x1.append(posv[i][0])
            y1.append(posv[i][1])
            z1.append(posv[i][2])
        if (i>=26 and i<41)  :
            x3.append(posv[i][0])
            y3.append(posv[i][1])
            z3.append(posv[i][2])
        if (i>41) or i==4:
            x2.append(posv[i][0])
            y2.append(posv[i][1])
            z2.append(posv[i][2])
        # if i>4:
        #     raw_input()
    """ for i in range(len(posa)):
        ax.scatter(posa[i][0], posa[i][1], posa[i][2], c="red") """
    pos_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for jj in range(len(pos_1)):
        # print(jj)
        x4.append(posv[jj][0])
        y4.append(posv[jj][1])
        z4.append(posv[jj][2])

    ax.plot(x, y, z, color='b')
    ax.plot(x2, y2, z2, color='b')
    ax.plot(x1, y1, z1, color='b')
    ax.plot(x3, y3, z3, color='b')
    ax.plot(x4, y4, z4, color='b')
    x5 = [];y5 = [];z5 = [] 
    pos_1 = [6, 16]
    for jj in range(len(pos_1)):
        # print(jj)
        x5.append(posv[jj][0])
        y5.append(posv[jj][1])
        z5.append(posv[jj][2])

    ax.plot(x5, y5, z5, color='b')
    x6 = [];y6 = [];z6= [] 
    pos_1 = [4, 11, 26]
    for jj in range(len(pos_1)):
        # print(jj)
        x6.append(posv[jj][0])
        y6.append(posv[jj][1])
        z6.append(posv[jj][2])

    ax.plot(x6, y6, z6, color='b')
    for i in range(len(posa)):
        ax.scatter(posa[i][0], posa[i][1], posa[i][2], c="red")
        # ax.text(posa[i][0], posa[i][1], posa[i][2],  '%s' % (str(i)), size=10, zorder=1, color='k') 
    pos_2a = [23, 22, 21, 20, 19 ,18 ,17 ,16 ,15, 14 ,13 ,12 ,11, 10 ,9 ,8, 7 ,6 ,5 ,4 , 3, 2 ,1, 0]
    x = [];y = [];z = []    
    for jj in range(len(pos_2a)):
        x.append(posa[pos_2a[jj]][0])
        y.append(posa[pos_2a[jj]][1])
        z.append(posa[pos_2a[jj]][2])
    ax.plot(x, y, z, color='r')
    pos_2a = [44 ,46 ,45, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 12]
    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    for jj in range(len(pos_2a)):
        x.append(posa[pos_2a[jj]][0])
        y.append(posa[pos_2a[jj]][1])
        z.append(posa[pos_2a[jj]][2])
    ax.plot(x, y, z, color='r')
    pos_2a = [28 ,27, 26, 25, 24, 20]
    x = [];y = [];z = []   
    for jj in range(len(pos_2a)):
        x.append(posa[pos_2a[jj]][0])
        y.append(posa[pos_2a[jj]][1])
        z.append(posa[pos_2a[jj]][2])
    ax.plot(x, y, z, color='r')


    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    for i in range(len(posv2)):
        ax2.scatter(posv2[i][0], posv2[i][1], posv2[i][2], c="blue", s=20)
        if (i>=16 and i<=24) :
            x1.append(posv2[i][0])
            y1.append(posv2[i][1])
            z1.append(posv2[i][2])
        if (i>=26 and i<41)  :
            x3.append(posv2[i][0])
            y3.append(posv2[i][1])
            z3.append(posv2[i][2])
        if (i>41) or i==4:
            x2.append(posv2[i][0])
            y2.append(posv2[i][1])
            z2.append(posv2[i][2])
        # if i>4:
        #     raw_input()
    """ for i in range(len(posa)):
        ax.scatter(posa[i][0], posa[i][1], posa[i][2], c="red") """
    pos_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for jj in range(len(pos_1)):
        # print(jj)
        x4.append(posv2[jj][0])
        y4.append(posv2[jj][1])
        z4.append(posv2[jj][2])

    ax2.plot(x, y, z, color='b')
    ax2.plot(x2, y2, z2, color='b')
    ax2.plot(x1, y1, z1, color='b')
    ax2.plot(x3, y3, z3, color='b')
    ax2.plot(x4, y4, z4, color='b')
    # ax2.plot(x, y, z, color='b')
    for i in range(len(posa2)):
        ax2.scatter(posa2[i][0], posa2[i][1], posa2[i][2], c="red")
    pos_2a = [23, 22, 21, 20, 19 ,18 ,17 ,16 ,15, 14 ,13 ,12 ,11, 10 ,9 ,8, 7 ,6 ,5 ,4 , 3, 2 ,1, 0]
    x = [];y = [];z = []    
    for jj in range(len(pos_2a)):
        x.append(posa2[pos_2a[jj]][0])
        y.append(posa2[pos_2a[jj]][1])
        z.append(posa2[pos_2a[jj]][2])
    ax2.plot(x, y, z, color='r')
    pos_2a = [44 ,46 ,45, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 12]
    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    for jj in range(len(pos_2a)):
        x.append(posa2[pos_2a[jj]][0])
        y.append(posa2[pos_2a[jj]][1])
        z.append(posa2[pos_2a[jj]][2])
    ax2.plot(x, y, z, color='r')
    pos_2a = [28 ,27, 26, 25, 24, 20]
    x = [];y = [];z = []   
    for jj in range(len(pos_2a)):
        x.append(posa2[pos_2a[jj]][0])
        y.append(posa2[pos_2a[jj]][1])
        z.append(posa2[pos_2a[jj]][2])
    ax2.plot(x, y, z, color='r')
    
    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    
    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    for i in range(len(posv3)):
        ax3.scatter(posv3[i][0], posv3[i][1], posv3[i][2], c="blue", s=20)
        if (i>=16 and i<=24) :
            x1.append(posv3[i][0])
            y1.append(posv3[i][1])
            z1.append(posv3[i][2])
        if (i>=26 and i<41)  :
            x3.append(posv3[i][0])
            y3.append(posv3[i][1])
            z3.append(posv3[i][2])
        if (i>41) or i==4:
            x2.append(posv3[i][0])
            y2.append(posv3[i][1])
            z2.append(posv3[i][2])
        # if i>4:
        #     raw_input()
    """ for i in range(len(posa)):
        ax.scatter(posa[i][0], posa[i][1], posa[i][2], c="red") """
    pos_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for jj in range(len(pos_1)):
        # print(jj)
        x4.append(posv3[jj][0])
        y4.append(posv3[jj][1])
        z4.append(posv3[jj][2])

    ax3.plot(x, y, z, color='b')
    ax3.plot(x2, y2, z2, color='b')
    ax3.plot(x1, y1, z1, color='b')
    ax3.plot(x3, y3, z3, color='b')
    ax3.plot(x4, y4, z4, color='b')
    for i in range(len(posa3)):
        ax3.scatter(posa3[i][0], posa3[i][1], posa3[i][2], c="red")
    pos_2a = [23, 22, 21, 20, 19 ,18 ,17 ,16 ,15, 14 ,13 ,12 ,11, 10 ,9 ,8, 7 ,6 ,5 ,4 , 3, 2 ,1, 0]
    x = [];y = [];z = []    
    for jj in range(len(pos_2a)):
        x.append(posa3[pos_2a[jj]][0])
        y.append(posa3[pos_2a[jj]][1])
        z.append(posa3[pos_2a[jj]][2])
    ax3.plot(x, y, z, color='r')
    pos_2a = [44 ,46 ,45, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 12]
    x = [];y = [];z = []    
    x1 = [];y1 = [];z1 = []    
    x2 = [];y2 = [];z2 = []    
    x3 = [];y3 = [];z3 = []    
    x4 = [];y4 = [];z4 = [] 
    for jj in range(len(pos_2a)):
        x.append(posa3[pos_2a[jj]][0])
        y.append(posa3[pos_2a[jj]][1])
        z.append(posa3[pos_2a[jj]][2])
    ax3.plot(x, y, z, color='r')
    pos_2a = [28 ,27, 26, 25, 24, 20]
    x = [];y = [];z = []   
    for jj in range(len(pos_2a)):
        x.append(posa3[pos_2a[jj]][0])
        y.append(posa3[pos_2a[jj]][1])
        z.append(posa3[pos_2a[jj]][2])
    ax3.plot(x, y, z, color='r')
    
    ax.set_xlim(-5,5)

    ax.set_ylim(-5,5)

    ax.set_zlim(-5,5)
    ax2.set_xlim(-5,5)

    ax2.set_ylim(-5,5)

    ax2.set_zlim(-5,5)  
    ax3.set_xlim(-5,5)

    ax3.set_ylim(-5,5)

    ax3.set_zlim(-5,5)  
    ax.title.set_text("Position of the point generated by the simulation, set "+file_number)

    ax2.title.set_text("Position of the point generated by the regression model, set "+file_number)  
    ax3.title.set_text("Position of the point generated by the LSTM model, set "+file_number)  
    # ax.set_axis_off()
    # ax2.set_axis_off()

    
    plt.draw() 
    plt.pause(0.000001)
    if count==0:
        raw_input()
        count+=1


plt.show()





  
# Show plot
#plt.draw() 
#plt.pause(0.01) #is necessary for the plot to update for some reason
# c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\3dplotspline.py"


#send position and desired rotation of the kidney randomly or not