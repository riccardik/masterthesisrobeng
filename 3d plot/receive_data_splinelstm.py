# -*- coding: iso-8859-15 -*-

import json, socket
from random import randint, randrange
import random
import datetime
import time
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, Sequential
# Helper libraries
import numpy as np
from numpy import asarray

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

class Server(object):
  """
  A JSON socket server used to communicate with a JSON socket client. All the
  data is serialized in JSON. How to use it:

  server = Server(host, port)
  while True:
    server.accept()
    data = server.recv()
    # shortcut: data = server.accept().recv()
    server.send({'status': 'ok'})
  """

  backlog = 5
  client = None

  def __init__(self, host, port):
    self.socket = socket.socket()
    self.socket.bind((host, port))
    self.socket.listen(self.backlog)
    

  def __del__(self):
    self.close()

  def accept(self):
    # if a client is already connected, disconnect it
    if self.client:
      self.client.close()
    self.client, self.client_addr = self.socket.accept()
    return self

  def send(self, data):
    if not self.client:
      raise Exception('Cannot send data, no client is connected')
    _send(self.client, data)
    return self

  def recv(self):
    if not self.client:
      raise Exception('Cannot receive data, no client is connected')
    return _recv(self.client)

  def close(self):
    if self.client:
      self.client.close()
      self.client = None
    if self.socket:
      self.socket.close()
      self.socket = None


class Client(object):
  """
  A JSON socket client used to communicate with a JSON socket server. All the
  data is serialized in JSON. How to use it:

  data = {
    'name': 'Patrick Jane',
    'age': 45,
    'children': ['Susie', 'Mike', 'Philip']
  }
  client = Client()
  client.connect(host, port)
  client.send(data)
  response = client.recv()
  # or in one line:
  response = Client().connect(host, port).send(data).recv()
  """

  socket = None

  def __del__(self):
    self.close()

  def connect(self, host, port):
    self.socket = socket.socket()
    self.socket.connect((host, port))
    return self

  def send(self, data):
    if not self.socket:
      raise Exception('You have to connect first before sending data')
    _send(self.socket, data)
    return self

  def recv(self):
    if not self.socket:
      raise Exception('You have to connect first before receiving data')
    return _recv(self.socket)

  def recv_and_close(self):
    data = self.recv()
    self.close()
    return data

  def close(self):
    if self.socket:
      self.socket.close()
      self.socket = None

## helper functions ##

def _send(socket, data):
  try:
    serialized = json.dumps(data)
  except (TypeError, ValueError) as e:
    raise Exception('You can only send JSON-serializable data')
  # send the length of the serialized data first
  socket.send('%d\n' % len(serialized))
  # send the serialized data
  socket.sendall(serialized)

def _recv(socket):
  # read the length of the data, letter by letter until we reach EOL
  length_str = ''
  char = socket.recv(1)
  while char.decode() != '\n':
    # length_str.join(chr(char))
    length_str+=char.decode()
    char = socket.recv(1)
    # print(length_str)
  total = int(length_str)
  # print(total)
  # use a memoryview to receive the data chunk by chunk efficiently
  view = memoryview(bytearray(total))
  next_offset = 0
  while total - next_offset > 0:
    recv_size = socket.recv_into(view[next_offset:], total - next_offset)
    next_offset += recv_size
  try:
    deserialized = json.loads(view.tobytes())
  except (TypeError, ValueError)as e:
    raise Exception('Data received was not in JSON format')
  return deserialized
file_number = '0927221920'

fname = "./new/Nuova Cartella/Completenewgood/log_status/data"+file_number+"minmaxnormatotale.txt"
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




host = 'LOCALHOST'
port = 55552

ts = time.time()
""" fname = "./log_status/data"+datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')+".txt"
print fname """


server = Server(host, port)
time = 0


# fname_model = "./"+"1hiddennewdatasets2.h5"
fname_model="lstmnewdataset13101724.h5"  
model = tf.keras.models.load_model(fname_model)
fname_model = "./"+"000"
model.summary()


# fig, ax = plt.subplots(figsize =(10, 7))
# fig1, ax1 = plt.subplots(figsize =(10, 7))
fig2 = plt.figure(figsize=(8,8))
ax3 = fig2.add_subplot(111, projection='3d')
fig3 = plt.figure(figsize=(8,8))
ax4 = fig3.add_subplot(111, projection='3d')

index12=0

x_arr = np.empty([30, 6])
while True:

  

  try:
    server.accept()
    #f = open(fname, "a")
    try: 
      data = server.recv()
      # print (data)
      ax3.cla()
      ax4.cla()
      # print(data['position_kidney'])
      # print(data['rotation_kidney'])
      pos_ = data['position_kidney']
      rot_ = data['rotation_kidney']
      x = np.asarray([[pos_[0][0],pos_[0][1],pos_[0][2], rot_[0][0], rot_[0][1], rot_[0][2]]])
      # print(x)
      x_arr = np.delete(x_arr, 0, 0)
      norm = np.empty_like(x)
      for j in range(0,x.shape[0]):
          #print a[j]
          if min_vall[j] == max_vall[j] :
              norm[j] = 0
          else:
              norm[j] = (x[j]-min_vall[j])/(max_vall[j]-min_vall[j])
      x_arr = np.append(x_arr, norm, axis = 0)
      print(x_arr)
      
      x_arr.shape

      x_arr_exp = np.expand_dims(x_arr, axis=0)
      
      # print (x_arr_exp)
      pred = model.predict([x_arr_exp])

      denompY = np.empty_like(pred)
      for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
          denompY[i][j] = pred[i][j]*(max_valv[j]-min_valv[j]) + min_valv[j]
      # print (pred)
      # force_a = data['arr_force_a']
      # #print force_a[0]
      # fa_str = str(force_a)
      # fa_str = fa_str.replace('[', '')
      # fa_str = fa_str.replace(']', '')
      # fa_str = fa_str.replace(' ', '') 

      """ rint type(force_a[0])
      a = np.array(force_a[0])
      
      b = [1, 2, 3]
      pprint a """
      # denompY = np.asarray(denompYp)
      # print(denompY.shape)
      array12 = []
      stringarr = ""

      origin=np.asarray([[pos_[0][0]],[pos_[0][1]],[pos_[0][2]]])
      tail1=np.asarray([[1],[0],[0]])
      tail2=np.asarray([[0],[-1],[0]])
      tail3=np.asarray([[0],[0],[1]])

      

      index12 +=1

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
      # print(origin)
      # print(tail1)
      # print(tail2)
      # print(tail3)
      # input()

      q1 = ax3.quiver(origin[0], origin[1], origin[2], tail1[0], tail1[1], tail1[2], color='red',linestyle='solid')
      q2=ax3.quiver(origin[0], origin[1], origin[2], tail2[0], tail2[1], tail2[2], color='blue',linestyle='solid')
      q3 = ax3.quiver(origin[0], origin[1], origin[2], tail3[0], tail3[1], tail3[2], color='green',linestyle='solid')
      q1 = ax4.quiver(origin[0], origin[1], origin[2], tail1[0], tail1[1], tail1[2], color='red',linestyle='solid')
      q2=ax4.quiver(origin[0], origin[1], origin[2], tail2[0], tail2[1], tail2[2], color='blue',linestyle='solid')
      q3 = ax4.quiver(origin[0], origin[1], origin[2], tail3[0], tail3[1], tail3[2], color='green',linestyle='solid')
      
      row = []
      string1 = ""
      for j in range (0, int(denompY.shape[1]), 3):
        #print (str(denompY[j])+str(denompY[j+1])+str(denompY[j+2])+"\n")
        #string1 = string1+ "["+str(j)+" "+str(j+1)+" "+str(j+2)+"]"
        row.append([denompY[0][j],denompY[0][j+1], denompY[0][j+2]])
        # print(row)
         
        
      npa = np.asarray(row)

      posv = data['arr_pos_v']
      posv2 = npa
      # print(posv2)


      
      # pos_as = str(pos_a)
      # pos_as = pos_as.replace('[', '')
      # pos_as = pos_as.replace(']', '')
      # pos_as = pos_as.replace(' ', '') 
      # #pos_as = "["+pos_as+"]"

      # print(posv[0])
      x = [];y = [];z = []    
      x1 = [];y1 = [];z1 = []    
      x2 = [];y2 = [];z2 = []    
      x3 = [];y3 = [];z3 = []    
      x4 = [];y4 = [];z4 = []    
      for i in range(len(posv)):
        # print(i)
        ax3.scatter(posv[i][0], posv[i][1], posv[i][2], c="blue", s=20, label=str(i))
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

      ax3.plot(x, y, z, color='b')
      ax3.plot(x2, y2, z2, color='b')
      ax3.plot(x1, y1, z1, color='b')
      ax3.plot(x3, y3, z3, color='b')
      ax3.plot(x4, y4, z4, color='b')

      # ax2.plot(x, y, z, color='b')
        

      x = [];y = [];z = []    
      x1 = [];y1 = [];z1 = []    
      x2 = [];y2 = [];z2 = []    
      x3 = [];y3 = [];z3 = []    
      x4 = [];y4 = [];z4 = [] 
      for i in range(len(posv2)):
          ax4.scatter(posv2[i][0], posv2[i][1], posv2[i][2], c="blue", s=20)
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

      ax4.plot(x, y, z, color='b')
      ax4.plot(x2, y2, z2, color='b')
      ax4.plot(x1, y1, z1, color='b')
      ax4.plot(x3, y3, z3, color='b')
      ax4.plot(x4, y4, z4, color='b')


    
      ax3.set_xlim(-5,5)

      ax3.set_ylim(-5,5)

      ax3.set_zlim(-5,5)
      ax4.set_xlim(-5,5)

      ax4.set_ylim(-5,5)

      ax4.set_zlim(-5,5)
      

      ax4.title.set_text("Position of the point generated by the trained model")  
      ax3.title.set_text("Position of the point generated by the simulation")







      plt.draw() 
      plt.pause(0.000001)
      # q1.remove()
      # q2.remove()
      # q3.remove()
     


      # server.send({"response":"ciao"})
      
				
				
    except socket.error:
          print("socket error")				
          server.socket.close()
          server = Server(host, port)
  except socket.timeout:
    b=1

server.close()

#c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\receive_data_spline.py"


#send position and desired rotation of the kidney randomly or not