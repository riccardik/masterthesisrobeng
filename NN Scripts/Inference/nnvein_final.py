import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, Sequential
# Helper libraries
import numpy as np
from numpy import asarray

import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
import time
import datetime
""" 
from os.path import abspath
from inspect import getsourcefile
print(abspath(getsourcefile(lambda:0)))
"""


#random
#0906191003
#rotation
#0906185237

datasets=['0920114247', '0920121748', '0920125053', '0927225115', '0927232519', '0927235241', '0928113454', '0928120640', '0930134844', '0930142045', '0930144850', '0930151638', '0930174546', '0930184441', '0930195935', '0930203905', '0927221920', '0906191003', '0929110030', '0930212107', '1011195942','1011213606' ,'1011193035', '1011220541', '1011223321', '1012211411', '1012214652', '1012203101', '1013101208', '1013105041', '1013112504']

fname_model = "./"+"vein_nnmodel_final.h5"

model = tf.keras.models.load_model(fname_model)
fname_model = "./"+"000"
model.summary()



# --------

res_array = []
# file_number = "0929110030"

file_number = '0927221920'
# file_number = '0906191003'
# file_number = '1013115727'


dataset_choose = "v"
save_results_tofile = 0
load_results_inference = 1

fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"_denorm"+dataset_choose+".txt"
fname2 = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"_denorml.txt"
f = open(fname, "r")
all_of_it = f.read()
#print (all_of_it)
f.close()
f = open(fname, "r")
Lines = f.readlines()
count = 0
arr = []
for line in Lines:
  #count += 1
  #print (count)
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
  
  """ floats_list = []
  for item in a.split(','):
    floats_list.append(float(item))

  print(floats_list) """
  arr.append(a)

f.close()

dataY =  np.array(arr)
#print (dataX)
#print(dataX.shape)

f2 = open(fname2, "r")
Lines = f2.readlines()
count = 0
arr2 = []
for line in Lines:
  count += 1
  #print (count)
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
  
  """ floats_list = []
  for item in a.split(','):
    floats_list.append(float(item))

  print(floats_list) """
  arr2.append(a)

f2.close()
dataX =  np.array(arr2)
print(dataY.shape)
print(dataX.shape)

fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"minmaxdenorm.txt"
f = open(fname, "r")
Lines = f.readlines()
count = 0
arr = []
read_line = json.loads(Lines[0])
f.close()
max_vala = read_line['max_vala']
max_valv = read_line['max_valv']
max_vall = read_line['max_vall']
min_vala = read_line['min_vala']
min_valv = read_line['min_valv']
min_vall = read_line['min_vall']
variancea = read_line['variancea']
variancev = read_line['variancev']
variancel = read_line['variancel']








predicY = np.empty_like(dataY)
denomY  = dataY


if load_results_inference == 0:

  for i in range(dataX.shape[0]):
    print(i)
    x_totest = dataX[i]
    newX = asarray([x_totest])
    #print (newX)
    y_totest = dataY[i].tolist()
    predicY[i] = asarray(model.predict(newX)[0])
    res = np.sqrt(np.power(dataY[i]-predicY[i] , 2))*100
    #res = np.power(y_test[i], 2)-np.power(yhat, 2)
    # """ print (np.average(res)) """
    res_array.append(np.average(res))
  denompY = predicY



if load_results_inference==1:
  fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposveinregr.txt"
  f2 = open(fname, "r")
  Lines = f2.readlines()
  count = 0
  arr2 = []
  a=[]
  for line in Lines:
    count += 1
    #print (count)
    #print("Line{}: {}".format(count, line.strip()))
    #print line
    line = line[:-1]

    # linee = '{"line": ['+line+']}'    
    read_line = json.loads(line)
    
    a = read_line["arr_pos_v_pred"]
    b = []
    for i in range(0,len(a)):
      b.append(a[i][0])
      b.append(a[i][1])
      b.append(a[i][2])
    arr2.append(b)
  f2.close()

  prova_array=np.array(arr2)
  denompY = prova_array



if save_results_tofile == 1:
  linee=""
  array12 = []
  stringarr = ""
  for i in range (math.ceil(denompY.shape[0])):
    #print("i")
    row = []
    string1 = ""
    for j in range (0, int(denompY.shape[1]), 3):
      #print (str(denompY[j])+str(denompY[j+1])+str(denompY[j+2])+"\n")
      #string1 = string1+ "["+str(j)+" "+str(j+1)+" "+str(j+2)+"]"
      row.append([denompY[i][j],denompY[i][j+1], denompY[i][j+2]])
    #print(string1 +"\n")  
    linee =  linee+'{"arr_pos_v_pred": '+str(row)+'}'+"\n"
  npa = np.asarray(array12)
  string1 = string1+str(j)+" "+str(j+1)+" "+str(j+2)+" "
      
  fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"predictedposveinregr.txt"

  #print self.fname
  fnminmax = open(fname, "w")
  #print data_dump
  #data_dump = "ciao"+"\n"
  fnminmax.write(linee) 
  fnminmax.close()




res_den = np.sqrt(np.square(denomY-denompY))


str_den = np.array2string(denompY)

sq_dist = []
for i in range (0, denomY.shape[0]):
    row = []
    for j in range (0, int(denomY.shape[1]), 3):
        # print("["+str(denompY[i][j])+str(denompY[i][j+1])+str(denompY[i][j+2])+"]")
        ps = np.array([denomY[i][j],denomY[i][j+1], denomY[i][j+2]])
        pm = np.array([denompY[i][j],denompY[i][j+1], denompY[i][j+2]])
        ard = math.sqrt(np.sum(np.square(ps - pm)))

        row.append(ard)
    sq_dist.append(row)

sq_dist_arr =  np.array(sq_dist)
sq_dist_arr.mean(axis=0)




variancev_ar = []

for j in range (0, len(variancev), 3):
    # print("["+str(denompY[i][j])+str(denompY[i][j+1])+str(denompY[i][j+2])+"]")
    ps = [variancev[j],variancev[j+1], variancev[j+2]]
    

    variancev_ar.append(ps)

variancev_arr = np.array(variancev_ar)

sq_distbyaxis = []
for i in range (0, denomY.shape[0]):
    row = []
    for j in range (0, int(denomY.shape[1]), 3):
        # print("["+str(denompY[i][j])+str(denompY[i][j+1])+str(denompY[i][j+2])+"]")
        ps = np.array([denomY[i][j],denomY[i][j+1], denomY[i][j+2]])
        pm = np.array([denompY[i][j],denompY[i][j+1], denompY[i][j+2]])
        ard = np.array([math.sqrt(np.sum(np.square(ps[0] - pm[0]))), math.sqrt(np.sum(np.square(ps[1] - pm[1]))), math.sqrt(np.sum(np.square(ps[2] - pm[2])))])

        row.append(ard)
    sq_distbyaxis.append(row)
distbyaxis_arr =  np.array(sq_distbyaxis)
distbyaxis_arr.mean(axis=0)
distbyaxis_arr.max(axis=0) 
distbyaxis_arr.min(axis=0)

error_percentage_variance = distbyaxis_arr.mean(axis=0)/variancev_arr*100
# error_percentage_variance.round(decimals=1)
# np.set_printoptions(suppress=True)


n_arr = []
for i in range(distbyaxis_arr.shape[0]):
  n_arr.append(i)
n_arr_np =np.array(n_arr)

time_ar = []
for i in range(math.ceil(distbyaxis_arr.shape[0])):
  time_ar.append(i/10)


x_err = []
x_err = time_ar
fig3 = plt.figure(figsize=(10,10))
ax3 = fig3.add_subplot(111)
fig4 = plt.figure(figsize=(10,10))
ax4 = fig4.add_subplot(111)
fig5 = plt.figure(figsize=(10,10))
ax5 = fig5.add_subplot(111)
fig6 = plt.figure(figsize=(10,10))
ax6 = fig6.add_subplot(111)
fig7 = plt.figure(figsize=(4,4))
ax7 = fig7.add_subplot(111)



y_err_arr = distbyaxis_arr
for i in range(y_err_arr.shape[1]):
        aaa=y_err_arr[:,i,0].tolist()
        ax3.plot(x_err, aaa)
for i in range(y_err_arr.shape[1]):
        aaa=y_err_arr[:,i,1].tolist()
        ax4.plot(x_err, aaa)
for i in range(y_err_arr.shape[1]):
        aaa=y_err_arr[:,i,2].tolist()
        ax5.plot(x_err, aaa)
for i in range(sq_dist_arr.shape[1]):
        aaa=sq_dist_arr[:,i].tolist()
        ax6.plot(x_err, aaa)




cumulative_error = np.empty_like(sq_dist_arr[:,0])
for j in range(y_err_arr.shape[1]):
  # cumulative_error=cumulative_error+sq_dist_arr[:,i]
  cumulative_error=sq_dist_arr[:,i]+cumulative_error
  # cumulative_error=sq_dist_arr[:,0]
cumulative_error = cumulative_error/56
err12 = cumulative_error.tolist()
ax7.plot(x_err, err12)

ax3.title.set_text("Error on the x axis for all the monitored points")
ax4.title.set_text("Error on the y axis for all the monitored points")
ax5.title.set_text("Error on the z axis for all the monitored points")
ax6.title.set_text("Position error for the monitored points")
ax7.title.set_text("Average position error")

plt.draw() 
plt.pause(0.000001)


# plot points

# denomY
# denompY

denomY_lst = []
for i in range (0, denomY.shape[0]):
    row = []
    for j in range (0, int(denomY.shape[1]), 3):
        # print("["+str(denompY[i][j])+str(denompY[i][j+1])+str(denompY[i][j+2])+"]")
        ps = np.array([denomY[i][j],denomY[i][j+1], denomY[i][j+2]])

        row.append(ps)
    denomY_lst.append(row)
# denomY_lst =  np.array(sq_distbyaxis)
denomY_arr =  np.array(denomY_lst)

denompY_lst = []
for i in range (0, denompY.shape[0]):
    row = []
    for j in range (0, int(denompY.shape[1]), 3):
        # print("["+str(denompY[i][j])+str(denompY[i][j+1])+str(denompY[i][j+2])+"]")
        ps = np.array([denompY[i][j],denompY[i][j+1], denompY[i][j+2]])

        row.append(ps)
    denompY_lst.append(row)
# denomY_lst =  np.array(sq_distbyaxis)
denompY_arr =  np.array(denompY_lst)



xsimpoints = denomY_arr[:,:,0]
ysimpoints = denomY_arr[:,:,1]
zsimpoints = denomY_arr[:,:,2]

xmodpoints = denompY_arr[:,:,0]
ymodpoints = denompY_arr[:,:,1]
zmodpoints = denompY_arr[:,:,2]

distx = np.sqrt(np.power(xsimpoints-xmodpoints, 2))
disty = np.sqrt(np.power(ysimpoints-ymodpoints, 2))
distz = np.sqrt(np.power(zsimpoints-zmodpoints, 2))

fig8 = plt.figure(figsize=(4,4))
ax81 = fig8.add_subplot(311)
ax82 = fig8.add_subplot(312)
ax83 = fig8.add_subplot(313)

for i in range (distx.shape[1]):
  ax81.plot(x_err, distx[:,i])
  ax82.plot(x_err, disty[:,i])
  ax83.plot(x_err, distz[:,i])


plt.draw() 
plt.pause(0.000001)

#pick one point to plot
point_in = 50
fig9 = plt.figure(figsize=(4,4))
ax91 = fig9.add_subplot(311)
ax92 = fig9.add_subplot(312)
ax93 = fig9.add_subplot(313)

ax91.plot(x_err, xmodpoints[:,point_in])
ax91.plot(x_err, xsimpoints[:,point_in])
ax91.legend(['Model','Simulation'])

ax92.plot(x_err, ymodpoints[:,point_in])
ax92.plot(x_err, ysimpoints[:,point_in])

ax93.plot(x_err, zmodpoints[:,point_in])
ax93.plot(x_err, zsimpoints[:,point_in])
fig9.suptitle('[x,y,z] position of point '+str(point_in), fontsize=12)
# ax91.set_ylim(-2,4)
# ax92.set_ylim(-2,4)
# ax93.set_ylim(-2,4)

plt.draw() 
plt.pause(0.000001)

# for point_in in range(denompY_arr.shape[1]):

# for point_in in range(20):
#   fig9 = plt.figure(figsize=(4,4))
#   ax91 = fig9.add_subplot(311)
#   ax92 = fig9.add_subplot(312)
#   ax93 = fig9.add_subplot(313)

#   ax91.plot(x_err, xmodpoints[:,point_in])
#   ax91.plot(x_err, xsimpoints[:,point_in])
#   ax91.legend(['Model','Simulation'])

#   ax92.plot(x_err, ymodpoints[:,point_in])
#   ax92.plot(x_err, ysimpoints[:,point_in])

#   ax93.plot(x_err, zmodpoints[:,point_in])
#   ax93.plot(x_err, zsimpoints[:,point_in])
#   fig9.suptitle('[x,y,z] position of point '+str(point_in), fontsize=12)
  
  


#   plt.draw() 
#   plt.pause(0.000001)

# dataX = prevdataX
# dataY = prevdataY
# print(dataY.shape)
# print(dataX.shape)




fig9 = plt.figure()
r_in = 0
c_ind = 0
subfigs = fig9.subfigures(3, 7, wspace=0.07)
for point_in in range(0, 21):
  # fig9 = plt.figure(figsize=(4,4))
  ax91 = subfigs[r_in][c_ind].add_subplot(311)
  ax92 = subfigs[r_in][c_ind].add_subplot(312)
  ax93 = subfigs[r_in][c_ind].add_subplot(313)

  ax91.plot(x_err, xmodpoints[:,point_in])
  ax91.plot(x_err, xsimpoints[:,point_in])
  ax91.set_ylim(-1,4)
  ax91.legend(['Model','Simulation'])

  ax92.plot(x_err, ymodpoints[:,point_in])
  ax92.plot(x_err, ysimpoints[:,point_in])
  ax92.set_ylim(-3,4)


  ax93.plot(x_err, zmodpoints[:,point_in])
  ax93.plot(x_err, zsimpoints[:,point_in])
  ax93.set_ylim(-3,2)
  subfigs[r_in][c_ind].suptitle('[x,y,z] position of point '+str(point_in), fontsize=12)
  c_ind +=1
  if c_ind>6:
    c_ind=0
    r_in+=1
  
  


plt.draw() 
plt.pause(0.000001)


fig9 = plt.figure()
r_in = 0
c_ind = 0
subfigs = fig9.subfigures(3, 7, wspace=0.07)
for point_in in range(21, 42):
  # fig9 = plt.figure(figsize=(4,4))
  ax91 = subfigs[r_in][c_ind].add_subplot(311)
  ax92 = subfigs[r_in][c_ind].add_subplot(312)
  ax93 = subfigs[r_in][c_ind].add_subplot(313)

  ax91.plot(x_err, xmodpoints[:,point_in])
  ax91.plot(x_err, xsimpoints[:,point_in])
  ax91.set_ylim(-1,4)
  ax91.legend(['Model','Simulation'])

  ax92.plot(x_err, ymodpoints[:,point_in])
  ax92.plot(x_err, ysimpoints[:,point_in])
  ax92.set_ylim(-3,4)


  ax93.plot(x_err, zmodpoints[:,point_in])
  ax93.plot(x_err, zsimpoints[:,point_in])
  ax93.set_ylim(-3,2)
  subfigs[r_in][c_ind].suptitle('[x,y,z] position of point '+str(point_in), fontsize=12)
  c_ind +=1
  if c_ind>6:
    c_ind=0
    r_in+=1
  
  


plt.draw() 
plt.pause(0.000001)


fig9 = plt.figure()
r_in = 0
c_ind = 0
subfigs = fig9.subfigures(3, 7, wspace=0.07)
for point_in in range(42, 54):
  # fig9 = plt.figure(figsize=(4,4))
  ax91 = subfigs[r_in][c_ind].add_subplot(311)
  ax92 = subfigs[r_in][c_ind].add_subplot(312)
  ax93 = subfigs[r_in][c_ind].add_subplot(313)

  ax91.plot(x_err, xmodpoints[:,point_in])
  ax91.plot(x_err, xsimpoints[:,point_in])
  ax91.set_ylim(-1,4)
  ax91.legend(['Model','Simulation'])

  ax92.plot(x_err, ymodpoints[:,point_in])
  ax92.plot(x_err, ysimpoints[:,point_in])
  ax92.set_ylim(-3,4)


  ax93.plot(x_err, zmodpoints[:,point_in])
  ax93.plot(x_err, zsimpoints[:,point_in])
  ax93.set_ylim(-3,2)
  subfigs[r_in][c_ind].suptitle('[x,y,z] position of point '+str(point_in), fontsize=12)
  c_ind +=1
  if c_ind>6:
    c_ind=0
    r_in+=1
  
  


plt.draw() 
plt.pause(0.000001)



denomY_reshaped = np.reshape(denomY, -1)
denompY_reshaped = np.reshape(denompY, -1)
correlation = pearsonr(denomY_reshaped, denompY_reshaped)

distance, path = fastdtw(denomY, denompY, 30,  dist=euclidean)

# distbyaxis_arr.mean(axis=0)
# correlation
# distance

# arr_print=distbyaxis_arr.mean(axis=0)   
# print(" \\\\\n".join([" & ".join(map('{0:.3f}'.format, line)) for line in arr_print]))

print("mean axis distance")
distbyaxis_arr.mean(axis=0).mean(axis=0) 
print("mean gaussian distance")
sq_dist_arr.mean(axis=0).mean(axis=0)
print("pearson index")
correlation
print("dtw distance")
distance


# ------ P38 ------------
p38pred = denompY[:, 38*3:38*3+3]
p38sim = denomY[:, 38*3:38*3+3]
p38predres = np.reshape(denompY[:, 38*3:38*3+3], -1)
p38simres = np.reshape(denomY[:, 38*3:38*3+3], -1)

p38corr = pearsonr(p38simres, p38predres)
distance38, path38 = fastdtw(denomY[:, 38*3:38*3+3], denompY[:, 38*3:38*3+3], 30,  dist=euclidean)

distance38ax = np.sqrt(np.square(p38pred-p38sim)).mean(axis=0)
gaussd38 = np.sqrt(np.square(p38pred[:, 0]- p38sim[:, 0])+np.square(p38pred[:, 1]- p38sim[:, 1])+np.square(p38pred[:, 2]- p38sim[:, 2])).mean()

# ------ P32 -----------
p32predres = np.reshape(denompY[:, 32*3:32*3+3], -1)
p32simres = np.reshape(denomY[:, 32*3:32*3+3], -1)
p32pred = denompY[:, 32*3:32*3+3]
p32sim = denomY[:, 32*3:32*3+3]


p32corr = pearsonr(p32simres, p32predres)
distance32, path32 = fastdtw(denomY[:, 32*3:32*3+3], denompY[:, 32*3:32*3+3], 30,  dist=euclidean)

distance32ax = np.sqrt(np.square(p32pred-p32sim)).mean(axis=0)
gaussd32 = np.sqrt(np.square(p32pred[:, 0]- p32sim[:, 0])+np.square(p32pred[:, 1]- p32sim[:, 1])+np.square(p38pred[:, 2]- p32sim[:, 2])).mean()


# ------ P34 -----------
p4predres = np.reshape(denompY[:, 4], -1)
p4simres = np.reshape(denomY[:, 4], -1)
p4pred = denompY[:, 32*3:32*3+3]
p4sim = denomY[:, 32*3:32*3+3]

p4corr = pearsonr(p4predres, p4simres)

distance4, path4 = fastdtw(denomY[:, 34*3:34*3+3], denompY[:, 4*3:4*3+3], 30,  dist=euclidean)

distance4ax = np.sqrt(np.square(p4pred-p4sim)).mean(axis=0)
gaussd4 = np.sqrt(np.square(p4pred[:, 0]- p4sim[:, 0])+np.square(p4pred[:, 1]- p4sim[:, 1])+np.square(p4pred[:, 2]- p4sim[:, 2])).mean()

