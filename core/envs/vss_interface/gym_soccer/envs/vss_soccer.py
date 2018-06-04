from core.envs.vss_interface.command_pb2 import *
from core.envs.vss_interface.debug_pb2 import *
from core.envs.vss_interface.state_pb2 import *

import time
import numpy as np
import pdb
import zmq
import google.protobuf.text_format
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import subprocess
import os
import signal

path_simulator = 'vss_sim/VSS-Simulator'
path_viewer = 'vss_sim/VSS-Viewer'

class SoccerEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        self.prev_robot_ball_dist = None
        self.prev_ball_goal_dist = None

    def setup_connections(self, ip='127.0.0.1', port=5555, is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, '-r', '50', '-d', '-a', '-p', str(self.port)])
        # state socket
        self.socket_state = self.context.socket(zmq.SUB) #socket to listen vision/simulator
        self.socket_state.connect ("tcp://localhost:%d" % port)
        #self.socket_state.setsockopt_string(zmq.SUBSCRIBE, b"")#allow every topic
        try:
        		self.socket_state.setsockopt(zmq.SUBSCRIBE, b'')
        except TypeError:
        		self.socket_state.setsockopt_string(zmq.SUBSCRIBE, b'')
        self.socket_state.setsockopt(zmq.LINGER, 0)
        self.socket_state.setsockopt(zmq.RCVTIMEO, 5000)

        # commands socket
        self.socket_com1 = self.context.socket(zmq.PAIR) #socket to Team 1
        self.socket_com1.connect ("tcp://localhost:%d" % (port+1))

        self.socket_com2 = self.context.socket(zmq.PAIR) #socket to Team 2
        self.socket_com2.connect ("tcp://localhost:%d" % (port+2))

        # debugs socket
        self.socket_debug1 = self.context.socket(zmq.PAIR) #debug socket to Team 1
        self.socket_debug1.connect ("tcp://localhost:%d" % (port+3))

        self.socket_debug2 = self.context.socket(zmq.PAIR) #debug socket to Team 2
        self.socket_debug2.connect ("tcp://localhost:%d" % (port+4))

        self.last_state, reward, done = self.parse_state(self.receive_state())
        self.init_state = self.last_state
        shape = len(self.last_state)
        #todo fix obs and action spaces
        self.observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(shape,))
        self.action_space = spaces.Box(low=-100, high=100, dtype=np.float32, shape=(9,)) #wheels velocities

        # Initialize poll set
        self.poller.register(self.socket_state, zmq.POLLIN)

    def close_connections(self):
        self.socket_state.disconnect ("tcp://localhost:%d" % self.port)
        self.socket_com1.disconnect ("tcp://localhost:%d" % (self.port+1))
        self.socket_com2.disconnect ("tcp://localhost:%d" % (self.port+2))
        self.socket_debug1.disconnect ("tcp://localhost:%d" % (self.port+3))
        self.socket_debug2.disconnect ("tcp://localhost:%d" % (self.port+4))
        self.context.destroy()

    def __del__(self):
        self.close_connections()
        # Send SIGTER (on Linux)
        self.p.terminate()
        # Wait for process to terminate
        returncode = self.p.wait()
        print('Process destroyed',self.port)
        #self.render(close=True)

    def receive_state(self):
        try:
			state = Global_State()
			msg = self.socket_state.recv()
			state.ParseFromString(msg)
			count = 0
			while(count < 100):
				socks = dict(self.poller.poll(10))
				if self.socket_state in socks and socks[self.socket_state] == zmq.POLLIN:
					#discard messages
					msg = self.socket_state.recv()
					state.ParseFromString(msg)
					count += 1
				else:
					break
			return state
        except Exception as e:
			print("caught timeout:"+str(e))
			 
        return None

    def send_commands(self, global_commands):
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"

        self.dict = {0:(0,0),
                     1:(10,0),
                     2:(0,10),
                     3:(10,10),
                     4:(-10,10),
                     5:(10,-10),
                     6:(-10,0),
                     7:(0,-10),
                     8:(-10,-10)
                    }
        robot = c.robot_commands.add()
        robot.id = 0
        robot.left_vel = self.dict[global_commands][0]
        robot.right_vel = self.dict[global_commands][1]
        #print("command:"+str(global_commands)+" vel:["+str(robot.left_vel)+","+str(robot.right_vel)+"]");
        for i in range(2):
            robot = c.robot_commands.add()
            robot.id = i+1
            robot.left_vel = 0#global_commands[0][2*i]*10
            robot.right_vel = 0#global_commands[0][2*i+1]*10

        buf = c.SerializeToString()
        if (self.is_team_yellow):
            self.socket_com1.send(buf)
        else:
            self.socket_com2.send(buf)

    def send_debug(self, global_debug):
        buf = global_debug.SerializeToString()
        if (self.is_team_yellow):
            self.socket_debug1.send(buf)
        else:
            self.socket_debug2.send(buf)

    def step(self, global_commands):
        self.send_commands(global_commands)
                
        rcvd_state = self.receive_state()
        while rcvd_state == None:
			self.reset()
			rcvd_state = self.receive_state()
			
        self.last_state, reward, done = self.parse_state(rcvd_state)
        return self.last_state, reward, done, {}

    def reset(self):
        print('RESET')
        self.prev_robot_ball_dist = None
        self.prev_ball_goal_dist = None
        # Send SIGKILL (on Linux)
        self.p.terminate()
        returncode = self.p.wait()
        # Empty buffers
        while(True):
            socks = dict(self.poller.poll(100))
            if self.socket_state in socks and socks[self.socket_state] == zmq.POLLIN:
                #discard messages
                message = self.socket_state.recv()
            else:
                break

        self.p = subprocess.Popen([path_simulator, '-r', '50', '-d', '-a', '-p', str(self.port)])
        self.last_state, reward, done = self.parse_state(self.receive_state())
        return self.last_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode='human', close=False):
        pass
        #if (close == True):
        #    if (self.is_rendering):
        #        self.v.terminate()
        #        self.is_rendering = False
        #else:
        #    if (not self.is_rendering):
        #        self.v = subprocess.Popen([path_viewer, '-p', str(self.port)])
        #        self.is_rendering = True

    def parse_state(self, state):
        for idx, ball in enumerate(state.balls):
            #real values
            ball_state = (ball.pose.x, ball.pose.y,
                          ball.v_pose.x, ball.v_pose.y)

            #estimated values
            estimated_ball_state = (ball.k_pose.x, ball.k_pose.y,
                                    ball.k_v_pose.x, ball.k_v_pose.y)

        t1_state = ()
        estimated_t1_state = ()
        for idx, t1_robot in enumerate(state.robots_yellow):
            #real values
            t1_state += (t1_robot.pose.x, t1_robot.pose.y, t1_robot.pose.yaw,
                         t1_robot.v_pose.x, t1_robot.v_pose.y, t1_robot.v_pose.yaw)

            #estimated values
            estimated_t1_state += (t1_robot.k_pose.x, t1_robot.k_pose.y, t1_robot.k_pose.yaw,
                                   t1_robot.k_v_pose.x, t1_robot.k_v_pose.y, t1_robot.k_v_pose.yaw)

        t2_state = ()
        estimated_t2_state = ()
        for idx, t2_robot in enumerate(state.robots_blue):
            #real values
            t2_state += (t2_robot.pose.x, t2_robot.pose.y, t2_robot.pose.yaw,
                         t2_robot.v_pose.x, t2_robot.v_pose.y, t2_robot.v_pose.yaw)
            #estimated values
            estimated_t2_state += (t2_robot.k_pose.x, t2_robot.k_pose.y, t2_robot.k_pose.yaw,
                                   t2_robot.k_v_pose.x, t2_robot.k_v_pose.y, t2_robot.k_v_pose.yaw)

        done = False

        reward = 0;
        if self.is_team_yellow:
            reward = state.goals_yellow - state.goals_blue
        else:
            reward = state.goals_blue - state.goals_yellow

        if(reward != 0):
            #pdb.set_trace()
            print("******************GOAL****************")
            reward = 5*reward*(11 - state.time)
            done = True
        elif(state.time >= 10):
            done = True
        else:
            ball = np.array((ball_state[0],ball_state[1]))
            goalR = np.array((165,65))
            rb1 = np.array((t1_state[0],t1_state[1]))
            robot_ball_dist = np.linalg.norm(ball-rb1)
            ball_goal_dist = np.linalg.norm(goalR-ball)
            if (self.prev_robot_ball_dist == None):
            	reward = -1
            else:
            	ball_reward = self.prev_robot_ball_dist-robot_ball_dist
            	goal_reward = self.prev_ball_goal_dist-ball_goal_dist
            	reward = ball_reward + 2*goal_reward - 0.2
            	#print("%.2f" %ball_reward, "%.2f" %goal_reward, "%.2f" %reward)
            self.prev_robot_ball_dist = robot_ball_dist
            self.prev_ball_goal_dist = ball_goal_dist
        #print("Reward:"+str(reward))
        #time.sleep(0.100)#200ms

        env_state = ball_state + t1_state + t2_state
        #unused infos
        #state.name_yellow
        #state.name_blue
        #print(state.time)

        return np.array(env_state), reward, done

