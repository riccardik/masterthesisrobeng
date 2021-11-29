# -*- coding: iso-8859-15 -*-

import json, socket
from random import randint, randrange
import random
import datetime
import time
import math


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
    self.socket.settimeout(0.1)
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
  except (TypeError, ValueError), e:
    raise Exception('You can only send JSON-serializable data')
  # send the length of the serialized data first
  socket.send('%d\n' % len(serialized))
  # send the serialized data
  socket.sendall(serialized)

def _recv(socket):
  # read the length of the data, letter by letter until we reach EOL
  length_str = ''
  char = socket.recv(1)
  while char != '\n':
    length_str += char
    char = socket.recv(1)
  total = int(length_str)
  # use a memoryview to receive the data chunk by chunk efficiently
  view = memoryview(bytearray(total))
  next_offset = 0
  while total - next_offset > 0:
    recv_size = socket.recv_into(view[next_offset:], total - next_offset)
    next_offset += recv_size
  try:
    deserialized = json.loads(view.tobytes())
  except (TypeError, ValueError), e:
    raise Exception('Data received was not in JSON format')
  return deserialized

position_array = [[[[0.0, 1.0, 2.5]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[0.0, 2.0, 2.5]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[0.0, 0.0, 3]], [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[0, -1, 3]], [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[0.75, -1.0, 3.0]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[-1, -1, 3]], [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[-2, -1, 2.5]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[-2, 2, 2]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[0, 4, 2]],  [[0.0, 0.0, -2.0]], [[1, 1, 0]]],
  [[[-3.5, 0, 0]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-1.75, -4.0, 0.0]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-1.5, -4, 1.5]],  [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-3.5, -2, -1]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-3.5, 0, -1.5]],  [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-2.5, 0, -2.5]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-3.2, 0, 1]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-2, 4, 1]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-2, 5, 0]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[-2.2, 4, -1.5]], [[2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[0, 0, -4]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[0, -4, -1]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[-1, -4, -1.5]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[2, -4, -2]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[2, 0, -3.5]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[-2, 0, -3]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[1.5, 4, -2.5]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[-1, 2, -3.5]],[[0.0, 0.0, 2.0]], [[1, 1, 0]]],
  [[[3.5, 4, 0]], [[-2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[3.5, -4, 0]], [[-2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[4, 0, -2]], [[-2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[1.5, 0, 2]], [[-2.0, 0.0, 0.0]], [[0, 1, 1]]],
  [[[0, 6.5, 0]],  [[0.0, -2.0, 0.0]], [[1, 0, 1]]],
  [[[1.5, 6, 0]],  [[0.0, -2.0, 0.0]], [[1, 0, 1]]],
  [[[-0.5, 6.5, 0]],  [[0.0, -2.0, 0.0]], [[1, 0, 1]]],
  [[[0, 6, -1]],  [[0.0, -2.0, 0.0]], [[1, 0, 1]]],
  [[[0, 6, 0.5]],  [[0.0, -2.0, 0.0]], [[1, 0, 1]]],
  [[[0, -5.5, 0]], [[0.0, 2.0, 0.0]], [[1, 0, 1]]],
  [[[1.5, -5.5, 0]], [[0.0, 2.0, 0.0]], [[1, 0, 1]]],
  [[[-1.5, -4.5, 0]], [[0.0, 2.0, 0.0]], [[1, 0, 1]]],] 


def PositionAndForceRandom(time):
  index_ = randint(0, len(position_array)-1)
  posForce = {
    "position": position_array[index_][0],
    "force":  position_array[index_][1],
    "direction":  position_array[index_][2],
    "time": time
  }
  return posForce

def PositionAndForce(time):
  posForce = {
    "position": [[0.0, 1.0, 3.0]],
    "force": [[0.0, 0.0, -2.0]],
    "direction": [[1, 1, 0]],
    "time": time
  }
  return posForce

host = 'LOCALHOST'
port = 55551

ts = time.time()
fname = "./log_status/data"+datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')+".txt"
print fname


server = Server(host, port)
time = 0

""" while True:
  arr = [[[[0.0, 1.0, 3.0]],  [[0.0, 0.0, -2.0]]],
  [[[0.0, 2.0, 3.0]],  [[0.0, 0.0, -2.0]]],
  [[[0.0, 0.0, 4.0]], [[0.0, 0.0, -2.0]]],
  [[[0, -1, 3]], [[0.0, 0.0, -2.0]]],
  [[[0.75, -1.0, 4.0]],  [[0.0, 0.0, -2.0]]],
  [[[-1, -1, 3.5]], [[0.0, 0.0, -2.0]]],
  [[[-2, -1, 3]],  [[0.0, 0.0, -2.0]]],
  [[[-2, 2, 2.5]],  [[0.0, 0.0, -2.0]]],
  [[[0, 4, 2.5]],  [[0.0, 0.0, -2.0]]],
  [[[-4.0, 0, 0]], [[2.0, 0.0, 0.0]]],
  [[[-2.5, -4.0, 0.0]], [[2.0, 0.0, 0.0]]],
  [[[-2.5, -4, 1.5]],  [[2.0, 0.0, 0.0]]],
  [[[-4, -2, -1]], [[2.0, 0.0, 0.0]]],
  [[[-4, 0, -1.5]],  [[2.0, 0.0, 0.0]]],
  [[[-3.5, 0, -2.5]], [[2.0, 0.0, 0.0]]],
  [[[-4, 0, 1]], [[2.0, 0.0, 0.0]]],
  [[[-3, 4, 1]], [[2.0, 0.0, 0.0]]],
  [[[-2.5, 5, 0]], [[2.0, 0.0, 0.0]]],
  [[[-2.5, 4, -1.5]], [[2.0, 0.0, 0.0]]],
  [[[0, 0, -4.5]],[[0.0, 0.0, 2.0]]],
  [[[0, -4, -2.5]],[[0.0, 0.0, 2.0]]],
  [[[-1, -4, -2.5]],[[0.0, 0.0, 2.0]]],
  [[[2, -4, -2.5]],[[0.0, 0.0, 2.0]]],
  [[[2, 0, -5]],[[0.0, 0.0, 2.0]]],
  [[[-2, 0, -4]],[[0.0, 0.0, 2.0]]],
  [[[1.5, 4, -3]],[[0.0, 0.0, 2.0]]],
  [[[-1, 2, -4]],[[0.0, 0.0, 2.0]]],
  [[[4, 4, 0]], [[-2.0, 0.0, 0.0]]],
  [[[4, -4, 0]], [[-2.0, 0.0, 0.0]]],
  [[[4, 0, -2]], [[-2.0, 0.0, 0.0]]],
  [[[2, 0, 2]], [[-2.0, 0.0, 0.0]]],
  [[[0, 6.5, 0]],  [[0.0, -2.0, 0.0]]],
  [[[1.5, 6.5, 0]],  [[0.0, -2.0, 0.0]]],
  [[[-0.5, 6.5, 0]],  [[0.0, -2.0, 0.0]]],
  [[[0, 6, -1]],  [[0.0, -2.0, 0.0]]],
  [[[0, 6.5, 0.5]],  [[0.0, -2.0, 0.0]]],
  [[[0, -6, 0]], [[0.0, 2.0, 0.0]]],
  [[[1.5, -6, 0]], [[0.0, 2.0, 0.0]]],
  [[[-1.5, -5, 0]], [[0.0, 2.0, 0.0]]],] 

  #print position_array[38][0]
  value = randint(0, len(arr)-1)
  print  position_array[value][0] """
time = 0
data1 = PositionAndForceRandom(time)
dir_r = [1, 1, 1]
dir_t = [1, 1, 1]
count = 0
rotation_change_sec = [5.0, 0.0, 0.0]
position_change_sec = [0, 0.0, 0.0]

random_enable = 1

while True:
  
 

  try:
    server.accept()
    #f = open(fname, "a")
    try: 
      data = server.recv()

      data1['position_kidney'] = data['position_kidney']
      data1['rotation_kidney'] = data['rotation_kidney']

      print data1
      

      
      #print(data)
      time_p = round(data['time_counter']*10)
      time_p2 = round(data['time_counter']*10)/10.0
      #print time_p2
      #print count
      if time_p2%3==0 and random_enable == 1:

        #change rotation and velocity every n [s]
        print time_p2
        #rotation_change_sec = [randrange(0, 5), randrange(0, 5), randrange(0, 5)]
        rotation_change_sec = [random.choice([5.0, 0.0]), random.choice([5.0, 0.0]), random.choice([5.0, 0.0])]
        dir_r = [random.choice([1, -1]), random.choice([1, -1]), random.choice([1, -1])]

        position_change_sec = [randrange(0, 100)/1000.0, randrange(0, 100)/1000.0, randrange(0, 100)/1000.0]
        dir_t = [random.choice([1, -1]), random.choice([1, -1]), random.choice([1, -1])]
      
        
        print "changing rotation to:"+ str(rotation_change_sec)+str(dir_r )
        print "changing position to:"+ str(position_change_sec)+str(dir_t )


        """ count += 1
        if count>5:
          count = 0
          print "ciaone" """

      if time_p%1==0:
        #print(data)


        rotation_change = [rotation_change_sec[0]/10.0, rotation_change_sec[1]/10.0, rotation_change_sec[2]/10.0]
        position_change = [position_change_sec[0]/10.0, position_change_sec[1]/10.0, position_change_sec[2]/10.0]

        
        """ pos_ = data1['position_kidney']
        pos_[0][0] = pos_[0][0] - 0.5
        data1['position_kidney'] = pos_ """

        pos_ = data1['position_kidney']

        pos_[0][0] = pos_[0][0] + position_change[0]*dir_t[0]
        if pos_[0][0]>-0.5:
          dir_t[0] = - dir_t[0]
          pos_[0][0] = pos_[0][0] + position_change[0]*dir_t[0]
        if pos_[0][0]<-2.5:
          dir_t[0] = - dir_t[0]
          pos_[0][0] = pos_[0][0] + position_change[0]*dir_t[0]

        pos_[0][1] = pos_[0][1] + position_change[1]*dir_t[1]
        if pos_[0][1]>-0.3:
          dir_t[1] = - dir_t[1]
          pos_[0][1] = pos_[0][1] + position_change[1]*dir_t[1]
        if pos_[0][1]<-2.3:
          dir_t[1] = - dir_t[1]
          pos_[0][1] = pos_[0][1] + position_change[1]*dir_t[1]
        
        pos_[0][2] = pos_[0][2] + position_change[2]*dir_t[2]
        if pos_[0][2]>1:
          dir_t[2] = - dir_t[2]
          pos_[0][2] = pos_[0][2] + position_change[2]*dir_t[2]
        if pos_[0][2]<1:
          dir_t[2] = - dir_t[2]
          pos_[0][2] = pos_[0][2] + position_change[2]*dir_t[2]




        rot_ = data1['rotation_kidney']       


        rot_[0][0] = rot_[0][0] + rotation_change[0]*dir_r[0]
        if math.sqrt(rot_[0][0]*rot_[0][0])>180:
          dir_r[0] = - dir_r[0]
          rot_[0][0] = rot_[0][0] + rotation_change[0]*dir_r[0]

        rot_[0][1] = rot_[0][1] + rotation_change[1]*dir_r[1]
        if math.sqrt(rot_[0][1]*rot_[0][1])>75:
          dir_r[1] = - dir_r[1]
          rot_[0][1] = rot_[0][1] + rotation_change[1]*dir_r[1]
        
        rot_[0][2] = rot_[0][2] + rotation_change[2]*dir_r[2]
        if math.sqrt(rot_[0][2]*rot_[0][2])>30:
          dir_r[2] = - dir_r[2]
          rot_[0][2] = rot_[0][2] + rotation_change[2]*dir_r[2]
        
        
        #print rotation_change
        #print rot_

        data1['rotation_kidney'] = rot_

      """ if time_p%20==0:
        to_dump = {
          "position_ball": data["position_ball"],
          "force": data1["force"],
          "time": time_p,
          "position_monitor": data["position_monitor"] 
        }
        data_dump = json.dumps(to_dump)+"\n"
        f.write(data_dump)
        print time_p
        time = time_p
        #print (round(data['time_counter']))
        data1 = PositionAndForceRandom(time)
        f.close() """
      server.send({"response":data1})
				
				
    except socket.error:
          #print("socket error")				
          server.socket.close()
          server = Server(host, port)
  except socket.timeout:
    b=1

server.close()

# c:/Python27/python.exe "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\send_position_rotation.py"


#send position and desired rotation of the kidney randomly or not


