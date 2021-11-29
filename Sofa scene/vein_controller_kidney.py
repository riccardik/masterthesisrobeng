# -*- coding: iso-8859-15 -*-


import Sofa
import sys
import math
import datetime
import time
import os

#global liver
#global node
############################################################################################
# this is a PythonScriptController example script
############################################################################################

#file:jsonsocket.py
#https://github.com/mdebbar/jsonsocket
import json, socket

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

  backlog = 10
  client = None

  def __init__(self, host, port):
    self.socket = socket.socket()
    self.socket.bind((host, port))
    self.socket.listen(self.backlog)
    #self.socket.setblocking(0)
    self.socket.settimeout(0.001)


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
    self.socket.settimeout(0.01)
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
class Client2(object):
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
    self.socket.settimeout(0.01)
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


############################################################################################
# following defs are used later in the script
############################################################################################


def testNodes(node):
	""" node.findData('name').value = 'god'

	# Node creation
	adam = node.createChild('Adam')
	eve = node.createChild('Eve')
	abel = eve.createChild('Abel') """

	#you can animate simulation directly by uncommenting the following line:
	#node.animate=true

	return 0


# Python version of the "oneParticleSample" in cpp located in applications/tutorials/oneParticle
def oneParticleSample(node):
	""" node.findData('name').value='oneParticleSample'
	node.findData('gravity').value=[0.0, -9.81, 0.0]
	solver = node.createObject('EulerSolver',printLog='false')
	particule_node = node.createChild('particle_node')
	particle = particule_node.createObject('MechanicalObject')
	particle.resize(1)
	mass = particule_node.createObject('UniformMass',totalMass=1) """

	return 0



############################################################################################
# following defs are optional entry points, called by the PythonScriptController component;
############################################################################################

class ExampleController(Sofa.PythonScriptController):
	
	#global liver
	#global node
	
	
	# called once the script is loaded
	def onLoaded(self,node):
		self.counter = 0
		self.root = node
		""" self.MechanicalState = node.getObject('force92')
		self.indexs_force=[92, 105, 77, 22]
		self.actual_index_force=0 """
		self.actual_index_constraint=0
		""" self.parti = node.getObject('Particles')
		self.force_ball = node.getObject('force0')
		self.force_ball_constraint = node.getObject('forceBallConstraint')
		self.kidney_pos = node.getObject('../kidney/MechanicalModel')
		self.kidney_pos_initial = self.kidney_pos.position
		self.monitor_kidney = node.getObject('../kidney/Monitor') """


		self.vein_pos = node.getObject('./vene/MechanicalModelv')
		self.art_pos = node.getObject('./arterie/MechanicalModela')
		self.parti = node.getObject('./vene/MechanicalModelv')
		self.vein_pos_initial = self.vein_pos.position
		self.vein_vel_initial = self.vein_pos.velocity 
		self.art_pos_initial = self.art_pos.position
		self.art_vel_initial = self.art_pos.velocity 
		self.kidney_pos = self.vein_pos
		#self.kidney_coll_pos = node.getObject('../kidney/Collision/StoringForces')

		self.monitor_vein = node.getObject('./vene/Monitor')
		self.monitor_art = node.getObject('./arterie/Monitor')

		self.angleX = 0
		self.angleY = 0
		self.angleZ = 0

		#self.rotating_plane = node.getObject('./ParticleControl/Plane/mObject1')
		self.rotating_kidney = node.getObject('./kidney/MechanicalModel')
		self.parti = node.getObject('./ParticleControl/engine')
		print 'Controller script loaded from node %s'%node.findData('name').value
		host = 'LOCALHOST'
		port = 55550


		ts = time.time()
		self.fname = "./log_status/data"+datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')+".txt"
		#self.fname = "C:\Users\rick\Desktop\SOFA_v19.06.99_custom_Win64_v8.1\share\sofa\examples\Tutorials\training2020\new\Nuova cartella\completenewgood\log_status\data"+datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')+".txt"
		print self.fname
		f = open(self.fname, "a")
		f.close() 
		
		self.fname = "C:/Users/rick/Desktop/SOFA_v19.06.99_custom_Win64_v8.1/share/sofa/examples/Tutorials/training2020/new/Nuova cartella/completenewgood/log_status/data"+datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')+".txt"
		print self.fname
		


		#self.color = node.getObject('./kidney/Visual/VisualModel2')

		#self.server = Server(host, port)
		return 0

	# optionally, script can create a graph...
	def createGraph(self,node):
		print 'createGraph called (python side)'
		

		#uncomment to create nodes
		#testNodes(node)

		#uncomment to create the "oneParticle" sample
		oneParticleSample(node)

		return 0



	# called once graph is created, to init some stuff...
	def initGraph(self,node):
		print 'initGraph called (python side)'
		nodeParticle = node.getChild('ParticleControl')
		self.engine = nodeParticle.getObject('engine')


		
		

		
		return 0

	def bwdInitGraph(self,node):
		print 'bwdInitGraph called (python side)'
		sys.stdout.flush()
		self.vein_pos_initial = self.vein_pos.position
		self.vein_vel_initial = self.vein_pos.velocity  
		self.vein_force_initial = self.vein_pos.externalForce 
		self.vein_pos.externalForce = self.vein_force_initial
		
		self.art_pos_initial = self.art_pos.position
		self.art_vel_initial = self.art_pos.velocity  
		self.art_force_initial = self.art_pos.externalForce 
		self.art_pos.externalForce = self.art_force_initial
		#print self.vein_pos.externalForce
		#print self.vein_force_initial
		#print self.vein_pos_initial
		return 0

	def onIdle(self):
		sys.stdout.flush()
		return 0
		
	def onRecompile(self):
		print("The source ["+__file__+"] has changed and is reloaded.")
	
	def eucD(self, p1, p2):
		return math.sqrt(math.pow(p2[0]-p1[0], 2)+math.pow(p2[1]-p1[1], 2)+math.pow(p2[2]-p1[2], 2))
	# take second element for sort
	def takeSecond(self, elem):
		return elem[1]

	def findNearestIndices(self, curr_pos):
		print len(self.kidney_pos.position) 
		#print curr_pos

		vectt = []

		for j in range(0,len(self.kidney_pos.position)):
			#print self.eucD(curr_pos, self.kidney_pos.position[j])
			vectt.append([j, self.eucD(curr_pos, self.kidney_pos.position[j])])
			#print vectt[j]
		vectt.sort(key=self.takeSecond)
		#print vectt
		for j in range(0, 3):
			print vectt[j]
		return [[vectt[0][0], vectt[1][0], vectt[2][0]]]

		
		

		
	# called on each animation step
	total_time = 0
	time_counter = 0
	prev_response = ''
	indices_nearest = [[0, 0, 0]]
	position_indices = [[0,0,0], [0,0,0], [0,0,0]]
	velocity_indices = [[0,0,0], [0,0,0], [0,0,0]]
	time_counter2 = 0
	time_counter3 = 0
	
	
	def onBeginAnimationStep(self,dt):
		
		#print self.color.material
		""" rotation = self.engine.findData('rotation').value
		rotation[0][0] = rotation[0][0] + 0.5
		#print rotation[0][0]
		self.engine.findData('rotation').value = rotation
		#print self.rotating_plane.position[19]
		pos_ = self.vein_pos.position
		pos_[38] = self.rotating_plane.position[4]
		pos_[203] = self.rotating_plane.position[24]
		self.vein_pos.position = pos_ """

		#pos_ = self.vein_pos.position
		#pos_[38] = self.rotating_plane.position[4]
		""" pos_[2229] = self.rotating_plane.position[0]
		pos_[1835] = self.rotating_plane.position[14]
		pos_[1224] = self.rotating_plane.position[19] """
		#self.vein_pos.position = pos_
		#4 24
		#3 203

		
		self.total_time += dt

		#print self.engine.input_position
		#print self.engine.rotation

		



		host = 'LOCALHOST'
		port = 55550		
		""" try:
			self.server.accept()
			try: 
				data = self.server.recv()
				#self.server.send({"response":data})
				#print(data)

				#read current position and velocity
				pos = self.parti.position
				vel = self.parti.velocity
				
				resp_pos = {
					"position_ball" : self.parti.position[0]
				}
				print resp_pos
				self.server.send({"response":resp_pos})

				print data["position"]
				pos[0] = data["position"]
				vel = data["velocity"]
				print pos
				print

				#stop ball and move it
				self.parti.position = pos
				self.parti.velocity = vel
				self.force_ball.force= [[0.0, 0.0, 0.0]]
				#reset kidney initial positions and velocities
				self.kidney_pos.position = self.kidney_pos_initial
				self.kidney_pos.velocity = self.kidney_vel_initial
				
			except socket.error:
				print("socket error")				
				self.server.socket.close()
				self.server = Server(host, port)
		except socket.timeout:
			b=1 """
		
		port1 = 55551
		port2 = 55552
		graph_enable = 1
		self.time_counter+=1
		self.time_counter2 +=1
		self.time_counter3 +=1

		# self.time_counter3=3


		if self.time_counter2 >= 5:
			self.time_counter2 = 0
			print "pos:"+str(self.engine.input_position)+", rot:"+str(self.engine.rotation)
		if self.time_counter3 >=3 and graph_enable==1:
			print "graph enabled"
			self.time_counter3 = 0
			try:
				
				if len(self.monitor_vein.indices)>0:
					arr_pos_v = []
					arr_vel_v = []
					arr_force_v = []
					
					for j in range(0,len(self.monitor_vein.indices)):
						#print j
						posi = self.monitor_vein.indices[j][0]
						#print pos__
						arr_pos_v.append(self.vein_pos.position[posi])
						arr_vel_v.append(self.vein_pos.velocity[posi])
						arr_force_v.append(self.vein_pos.force[posi])

					# print str(arr_pos_v)+" \n"+ str(arr_vel_v)+" \n" + str(arr_force_v)+" \n" 

				if len(self.monitor_art.indices)>0:
					arr_pos_a = []
					arr_vel_a = []
					arr_force_a = []

					for j in range(0,len(self.monitor_art.indices)):
						posi = self.monitor_art.indices[j][0]
						#print pos__
						arr_pos_a.append(self.art_pos.position[posi])
						arr_vel_a.append(self.art_pos.velocity[posi])
						arr_force_a.append(self.art_pos.force[posi])

					#print str(arr_pos_a)+" \n" + str(arr_vel_a)+" \n" + str(arr_force_a)+" \n"
				pos__ = self.engine.input_position
				rot__ = self.engine.rotation
				resp_pos2 = {
						
						"time_counter": self.total_time,
                        "arr_pos_a": arr_pos_a,
                        "arr_force_a": arr_force_a,
                        "arr_pos_v": arr_pos_v,
                        "arr_force_v": arr_force_v,
						"position_kidney" : pos__,
						"rotation_kidney": rot__,


				}
				client1 = Client2()
				client1.connect(host, port2).send(resp_pos2)

				response1 = client1.recv()
				print response1['response']
				client1.close()
			except socket.error:
				#print("error, reconnecting")   

				client1.close()     
				#time.sleep(1)
			
				#print 'caught a timeout'		
				#return 0  

		if self.time_counter == 1:
			self.time_counter = 0
			#print self.total_time
			try:
				client = Client()

				pos__ = self.engine.input_position
				rot__ = self.engine.rotation
				
				#resp_pos = {
				#		"position_ball" : self.parti.position[0],
				#		"time_counter": self.total_time
				#	}
				#client.connect(host, port1).send(resp_pos)
				if len(self.monitor_vein.indices)>0:
					arr_pos_v = []
					arr_vel_v = []
					arr_force_v = []
					
					for j in range(0,len(self.monitor_vein.indices)):
						#print j
						posi = self.monitor_vein.indices[j][0]
						#print pos__
						arr_pos_v.append(self.vein_pos.position[posi])
						arr_vel_v.append(self.vein_pos.velocity[posi])
						arr_force_v.append(self.vein_pos.force[posi])

					# print str(arr_pos_v)+" \n"+ str(arr_vel_v)+" \n" + str(arr_force_v)+" \n" 

				if len(self.monitor_art.indices)>0:
					arr_pos_a = []
					arr_vel_a = []
					arr_force_a = []

					for j in range(0,len(self.monitor_art.indices)):
						posi = self.monitor_art.indices[j][0]
						#print pos__
						arr_pos_a.append(self.art_pos.position[posi])
						arr_vel_a.append(self.art_pos.velocity[posi])
						arr_force_a.append(self.art_pos.force[posi])

					#print str(arr_pos_a)+" \n" + str(arr_vel_a)+" \n" + str(arr_force_a)+" \n"
				

				

				""" for j in range (0, 3):
					self.position_indices[j] = self.kidney_pos.position[self.indices_nearest[0][j]]
					self.velocity_indices[j] = self.kidney_pos.velocity[self.indices_nearest[0][j]] """
				#print self.position_indices
				#print self.indices_nearest
				
				""" resp_pos = {
						"position_ball": self.parti.position[0], 
						"position_monitor" :arr_pos,
						"time_counter": self.total_time,
						"position_all": self.kidney_pos.position[0],
						#"indices_nearest": self.indices_nearest
						"indices_nearest": self.indices_nearest,
						"indices_position": self.position_indices,
						"indices_velocity": self.velocity_indices
				} """
				""" round_to_whole = [round(num) for num in a_list]
				print float(int(arr_force_a[0][0]*1000))/1000
				print arr_force_a[0][0]*1000/1000
				print asd[0][0]
 """
				to_dump = {
						"position_kidney" : pos__,
						"rotation_kidney": rot__,
						"time_counter": self.total_time,
						"arr_pos_v": arr_pos_v,
						"arr_vel_v": arr_vel_v,
						"arr_force_v": arr_force_v,
						"arr_pos_a": arr_pos_a,
						"arr_vel_a": arr_vel_a,
						"arr_force_a": arr_force_a,
						"indices_v":  self.monitor_vein.indices,
						"indices_a":  self.monitor_art.indices,
				}
				
				resp_pos = {
						"position_kidney" : pos__,
						"rotation_kidney": rot__,
						"time_counter": self.total_time
				}

				
				data_dump = json.dumps(to_dump)+"\n"
				#print self.fname
				fi = open(self.fname, "a")
				#print data_dump
				#data_dump = "ciao"+"\n"
				fi.write(data_dump) 
				fi.close()

				client.connect(host, port1).send(resp_pos)

				response = client.recv()

				""" if response == "{"response":data1}":
					print "ciao" """

				if response!=self.prev_response:
					#print (response)
					self.prev_response = response	

					pos_ = response['response']['position_kidney']
					self.engine.input_position = pos_
				
					rot_ = response['response']['rotation_kidney']
					self.engine.rotation = rot_
					
				
				""" if response!=self.prev_response:
					#print (response)
					self.prev_response = response					
					pos_ = response['response']['position']
					force_ = response['response']['force']
					direction_ = response['response']['direction']
					
					self.parti.position = pos_
					self.parti.velocity = [0.0, 0.0, 0.0]
					self.force_ball.force = force_
					self.force_ball_constraint.fixedDirections = direction_

					try:
						self.indices_nearest  = response['response']['indices']
					except KeyError:
						self.indices_nearest = self.findNearestIndices(self.parti.position[0])

					print "position: "+str(response['response']['position'])+",\t indices_monitored:" + str(self.indices_nearest)

					#reset kidney initial positions and velocities
					self.kidney_pos.position = self.kidney_pos_initial
					self.kidney_pos.velocity = self.kidney_vel_initial
					#self.indices_nearest = self.findNearestIndices(self.parti.position[0])
					#print self.indices_nearest """


				#print response['response']
				client.close()
			except socket.error:
				#print("error, reconnecting")   
				client.close()     
				#time.sleep(1)
			
		#print 'caught a timeout'		
		return 0  
		#print 'onBeginAnimatinStep (python) dt=%f total time=%f'%(dt,self.total_time)
		      

	def onEndAnimationStep(self,dt):
		sys.stdout.flush()
		return 0

	# called when necessary by Sofa framework... 
	""" def storeResetState(self):
		print 'storeResetState called (python side)'
		sys.stdout.flush()
		return 0

	def reset(self):
		print 'reset called (python side)'
		sys.stdout.flush()
		return 0

	def cleanup(self):
		print 'cleanup called (python side)'
		sys.stdout.flush()
		return 0 """


	# called when a GUIEvent is received
	def onGUIEvent(self,controlID,valueName,value):
		print 'GUIEvent received: controldID='+controlID+' valueName='+valueName+' value='+value
		sys.stdout.flush()
		return 0 

	def euler_to_quaternion(self):
		roll = self.angleX * math.pi /180
		pitch = self.angleY * math.pi /180
		yaw = self.angleZ * math.pi /180
		qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
		qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
		qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		return [qx, qy, qz, qw]

	def quaternion_to_euler(self, x, y, z, w):
		t0 = 2.0 * (w * x + y * z)
		t1 = 1.0 - 2.0 * (x * x + y * y)
		X = math.degrees(math.atan2(t0, t1))
		t2 = 2.0 * (w * y - z * x)
		t2 = 1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		Y = math.degrees(math.asin(t2))

		t3 = 2.0 * (w * z + x * y)
		t4 = 1.0 - 2.0 * (y * y + z * z)
		Z = math.degrees(math.atan2(t3, t4))

		self.angleX = X
		self.angleY = Y
		self.angleZ = Z
		return 0

	# key and mouse events; use this to add some user interaction to your scripts 
	def onKeyPressed(self,k):
		#print 'onKeyPressed '+k

		# Rotation of the support
		if k == '0'  :
			rotation = self.engine.findData('rotation').value
			rotation[0][0] = rotation[0][0] + 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			#print self.rotating_plane.position[19]
			
		
		if k == '9' :
			rotation = self.engine.findData('rotation').value
			rotation[0][0] = rotation[0][0] - 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			#print self.rotating_plane.position[19]
			

		# Increment force applied to a point of the veins' mesh
		if k == 'K' or k =='k':
			#print (self.parti)
			#print (self.parti.position)
			""" print self.vein_pos.position[202]
			pos_ = self.vein_pos.position
			pos_[202][1] = pos_[202][1]+1 

			self.vein_pos.position = pos_ """
			""" print self.vein_pos.velocity[202]
			vel_ = self.vein_pos.velocity
			vel_[202][1] = vel_[202][1]+1 
			vel_[203][1] = vel_[203][1]+1 
			vel_[207][1] = vel_[207][1]+1 

			self.vein_pos.velocity = vel_ """
			#print self.vein_pos.velocity[202]
			""" vel_ = self.vein_pos.velocity
			delta = -2
			axis = 1
			vel_[202][axis] = vel_[202][axis]+delta
			vel_[203][axis] = vel_[203][axis]+delta
			vel_[207][axis] = vel_[207][axis]+delta
			delta = 2
			axis = 2
			vel_[202][axis] = vel_[202][axis]+delta
			vel_[203][axis] = vel_[203][axis]+delta
			vel_[207][axis] = vel_[207][axis]+delta


			delta = -5
			axis = 1
			vel_[37][axis] = vel_[37][axis]+delta
			vel_[38][axis] = vel_[38][axis]+delta
			vel_[40][axis] = vel_[40][axis]+delta
			vel_[7][axis] = vel_[7][axis]+delta
			delta = 5
			axis = 2
			vel_[37][axis] = vel_[37][axis]+delta
			vel_[38][axis] = vel_[38][axis]+delta
			vel_[40][axis] = vel_[40][axis]+delta
			vel_[7][axis] = vel_[7][axis]+delta
			self.vein_pos.velocity = vel_ """
			
			""" print self.vein_pos.position[202]
			pos_ = self.vein_pos.position
			delta = 0.1
			axis = 0
			pos_[202][axis] = pos_[202][axis]+delta
			pos_[203][axis] = pos_[203][axis]+delta
			pos_[207][axis] = pos_[207][axis]+delta
			self.vein_pos.position = pos_ """
			#print self.vein_force_initial
			#self.vein_pos.externalForce = self.vein_force_initial
			#print self.vein_pos.externalForce[0]
			force_ = self.vein_pos.force
			""" delta = -5
			axis = 1
		
			force_[37][axis] = force_[37][axis]+delta
			force_[38][axis] = force_[38][axis]+delta
			force_[40][axis] = force_[40][axis]+delta
			force_[7][axis] = force_[7][axis]+delta """
			delta = -10
			axis = 2			
			force_[37][axis] = force_[37][axis]+delta
			force_[38][axis] = force_[38][axis]+delta
			force_[40][axis] = force_[40][axis]+delta
			force_[7][axis] = force_[7][axis]+delta
			self.vein_pos.externalForce = force_

			""" pos_ = self.vein_pos.position

			self.quaternion_to_euler(pos_[37],pos_[38],pos_[40])

			self.angleX = self.angleX - 5

			quat = self.euler_to_quaternion()

			pos_[37]= quat[0]
			pos_[38] = quat[1]
			pos_[40]= quat[2]
			self.vein_pos.position = pos_  """
            

			
			#202 203 207
			#37 38 40 7
			



		if k == 'B' or k =='b':
			
			""" print("reset kidney's position and velocity")
			#a = self.kidney_pos.position
			#print a
			#print self.kidney_pos_initial
			self.kidney_pos.position = self.kidney_pos_initial
			self.kidney_pos.velocity = self.kidney_vel_initial """

			#self.vein_pos.position = self.vein_pos_initial
			self.vein_pos.velocity = self.vein_vel_initial
			self.vein_pos.externalForce = self.vein_force_initial
			
			#self.art_pos.position = self.art_pos_initial
			self.art_pos.velocity = self.art_vel_initial
			self.art_pos.externalForce = self.art_force_initial

			#print self.vein_pos_initial
			#print self.vein_pos.position


			#print (self.parti.position)


			
			""" print len(self.monitor_kidney.indices)
			if len(self.monitor_kidney.indices)>0:
				# arr_pos=[self.kidney_pos.position[self.monitor_kidney.indices[0][0]]]
				#print arr_pos 
				arr_pos = []
				for j in range(0,len(self.monitor_kidney.indices)):
					pos__ = self.monitor_kidney.indices[j][0]
					print pos__
					arr_pos.append(self.kidney_pos.position[pos__])
				#for j in range(0,len(self.monitor_kidney.indices)):
				#arr_pos[0] = self.kidney_pos.position[10]
				#arr_pos.append (self.kidney_pos.position[10])
				print arr_pos
			#print self.monitor_kidney """

			#self.findNearestIndices(self.parti.position[0])
		
		if k == 'C' or k =='c':
			asd = self.kidney_pos.translation2
			print asd
			asd[0][0] = asd[0][0]+1
			self.kidney_pos.translation2 = asd




			
		if k == 'L' or k =='l':
			#print 'L pressed '
			""" print self.root.name
			self.ff=self.root.createObject('ConstantForceField', name="force92", indices=" 92 ", force="0 0 -300 ") """
			""" force = self.CFF.findData('force').value
			force[0][1] =math.cos((i*math.pi)/180)
			self.CFF.findData('force').value = force """
			#print (self.MechanicalState.force[0][2])
			#self.MechanicalState.force[0][2]=123.0
			#fff=123.2
			#forzeee=[[-1, -1, fff]]
			""" prev_f=self.MechanicalState.force[0][2]
			prev_f=prev_f+1000
			new_f=[[self.MechanicalState.force[0][0], self.MechanicalState.force[0][1], prev_f]]
			self.MechanicalState.force=new_f
			print ("The force is "+str(self.MechanicalState.force)) """
			prev_f=self.force_ball.force[0][2]
			prev_f=prev_f-1
			#new_f=[[self.force_ball.force[0][0], self.force_ball.force[0][1], prev_f]]
			new_f=[[prev_f, 0, 0]]
			self.force_ball.force=new_f
			print ("The force is "+str(self.force_ball.force))
			
		if k == 'i' or k =='I':
			self.actual_index_force=self.actual_index_force+1
			if self.actual_index_force>=len(self.indexs_force):
				self.actual_index_force=0
			#print (self.MechanicalState.indices)
			self.MechanicalState.indices=[[self.indexs_force[self.actual_index_force]]]
			print ("The force is being applied to: "+str(self.MechanicalState.indices[0][0]))
		
		if k == 'J' or k =='j':
			print (self.parti)
			print (self.parti.position)
			new_pos= self.parti.position[0][2]-0.2
			print (new_pos)
			self.parti.position=[[self.parti.position[0][0], self.parti.position[0][1], new_pos]]
			print (self.parti.position)

		## toggle the constraint on direction
		""" if k == 'M' or k =='m':

			# print (self.parti)
			#print (self.parti.position)
			#new_pos= self.parti.position[0][2]-0.2
			#print (new_pos)
			#self.parti.position=[[self.parti.position[0][0], self.parti.position[0][1], new_pos]]
			#print (self.parti.position)


			#force_constraint=self.force_ball_constraint.fixedDirections

			self.actual_index_constraint=self.actual_index_constraint+1
			if self.actual_index_constraint>2:
				self.actual_index_constraint=0
			if self.actual_index_constraint==0:
				self.force_ball_constraint.fixedDirections = [[1, 1, 0]]
				print ("force applied on z")
			if self.actual_index_constraint==1:
				self.force_ball_constraint.fixedDirections =[[0, 1, 1]]
				print ("force applied on x")
			if self.actual_index_constraint==2:
				self.force_ball_constraint.fixedDirections =[[1, 0, 1]]
				print ("force applied on y")
			print (self.actual_index_constraint)
			print(self.force_ball_constraint.fixedDirections) """



		

		## stop the ball
		if k == 'p' or k =='P':
			print (self.parti)
			
			self.parti.velocity=[[0, 0, 0]]
			self.force_ball.force= [[0.0, 0.0, 0.0]]
			print ("stopping ball")

		## Rotation on X
		if k == '1' or k =='1':
			rotation = self.engine.findData('rotation').value
			rotation[0][0] = rotation[0][0] + 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			print(rotation)

		if k == 'q' or k =='Q':
			rotation = self.engine.findData('rotation').value
			rotation[0][0] = rotation[0][0] - 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			print(rotation)
		## Rotation on y
		if k == '2' :
			rotation = self.engine.findData('rotation').value
			rotation[0][1] = rotation[0][1] + 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			print(rotation)
		if k == 'w' or k =='W':
			rotation = self.engine.findData('rotation').value
			rotation[0][1] = rotation[0][1] - 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
		## Rotation on Z
		if k == '3' :
			rotation = self.engine.findData('rotation').value
			rotation[0][2] = rotation[0][2] + 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation
			print(rotation)
		if k == 'e' or k =='E':
			rotation = self.engine.findData('rotation').value
			rotation[0][2] = rotation[0][2] - 0.5
			#print rotation[0][0]
			self.engine.findData('rotation').value = rotation

		""" ## Force on Z
		if k == '1' or k =='1':
			prev_f=self.force_ball.force
			prev_f[0][2]=prev_f[0][2]+1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"increasing force on z")

		if k == 'q' or k =='Q':
			prev_f=self.force_ball.force
			prev_f[0][2]=prev_f[0][2]-1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"decreasing force on z")
		## Force on X
		if k == '2' :
			prev_f=self.force_ball.force
			prev_f[0][0]=prev_f[0][0]+1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"increasing force on x")
		if k == 'w' or k =='W':
			prev_f=self.force_ball.force
			prev_f[0][0]=prev_f[0][0]-1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"decreasing force on x")
		## Force on Y
		if k == '3' :
			prev_f=self.force_ball.force
			prev_f[0][1]=prev_f[0][1]+1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"increasing force on y")
		if k == 'e' or k =='E':
			prev_f=self.force_ball.force
			prev_f[0][1]=prev_f[0][1]-1
			self.force_ball.force=prev_f
			print ("The force is "+str(self.force_ball.force)+"decreasing force on y") """


		## Position Z of the ball
		if k == '5' :
			prev_p=self.parti.input_position
			prev_p[0][2]=prev_p[0][2]+0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"increasing z position")

		if k == 't' or k =='T':
			prev_p=self.parti.input_position
			prev_p[0][2]=prev_p[0][2]-0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"decreasing  position")
		## Position on X
		if k == '6' :
			prev_p=self.parti.input_position
			prev_p[0][0]=prev_p[0][0]+0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"increasing x position")
		if k == 'y' or k =='Y':
			prev_p=self.parti.input_position
			prev_p[0][0]=prev_p[0][0]-0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"decreasing x position")
		## Position on Y
		if k == '7' :
			prev_p=self.parti.input_position
			prev_p[0][1]=prev_p[0][1]+0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"increasing y position")
		if k == 'U' or k =='u':
			prev_p=self.parti.input_position
			prev_p[0][1]=prev_p[0][1]-0.2
			self.parti.input_position=prev_p
			print ("The position is "+str(self.parti.input_position)+"decreasing y position")

			
		sys.stdout.flush()
		return 0 

	def onKeyReleased(self,k):
		#print 'onKeyReleased '+k
		sys.stdout.flush()
		return 0 

	def onMouseButtonLeft(self,x,y,pressed):
		print 'onMouseButtonLeft x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
		sys.stdout.flush()
		return 0

	def onMouseButtonRight(self,x,y,pressed):
		print 'onMouseButtonRight x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
		sys.stdout.flush()
		return 0

	def onMouseButtonMiddle(self,x,y,pressed):
		print 'onMouseButtonMiddle x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
		sys.stdout.flush()
		return 0

	def onMouseWheel(self,x,y,delta):
		print 'onMouseButtonWheel x='+str(x)+' y='+str(y)+' delta='+str(delta)
		sys.stdout.flush()
		return 0

        def onMouseMove(self,x,y):
		print 'onMouseMove x='+str(x)+' y='+str(y)
		sys.stdout.flush()
		return 0

	# called at each draw (possibility to use PyOpenGL)
	def draw(self):
		""" if self.counter < 10:
			print 'drawa ('+str(self.counter+1)+' / 10)'
			self.counter += 1 """
		sys.stdout.flush()

 
