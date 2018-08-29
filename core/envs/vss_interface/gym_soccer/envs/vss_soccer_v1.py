from core.envs.vss_interface.command_pb2 import *
from core.envs.vss_interface.debug_pb2 import *
from core.envs.vss_interface.state_pb2 import *
from core.envs.vss_interface.vss_utils import *

import time
import numpy as np
import pdb
import zmq
import google.protobuf.text_format
import gym

from gym import error, spaces
from gym import utils
from gym.utils import seeding
import math

import subprocess
import os
import signal

path_viewer = 'vss_sim/VSS-Viewer'
path_simulator = 'vss_sim/VSS-Simulator'
command_rate = 150 #ms
cmd_wait = 100 # 1/4 of 60 frames x (1s in ms)/fps  

class SoccerEnv_v1(gym.Env, utils.EzPickle):
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        
        self.initVars()

        #Positive potential constants
        self.u_B2G = -3
        self.u_R2B = -1
        #Negative potential constants
        self.u_B2OG = 0
        self.u_Col = 0
        
    def setup_connections(self, ip='127.0.0.1', port=5555, parameters = ['-a','-s','-d'] , is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.parameters = parameters
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, '-r', str(command_rate), '-p', str(self.port)] + self.parameters)
        # state socket
        self.socket_state = self.context.socket(zmq.SUB) #socket to listen vision/simulator
        self.socket_state.setsockopt(zmq.CONFLATE, 1)
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
        self.action_space = spaces.Box(low=-100, high=100, dtype=np.float32, shape=(6,)) #wheels velocities

        # Initialize poll set
        self.poller.register(self.socket_state, zmq.POLLIN)

    def initVars(self):
        #robot related
        self.my_agent = None
        self.ball = None
        self.target = {"theta":None,
                      "x": None,
                      "y": None}
        self.adv_goal = {"x":185, #adversary goal
                         "y":65}

        #potential vars
        self.old_p_B2G = None
        self.old_p_B2OG = None
        self.old_p_R2B = None
        self.old_p_Col = None

        #episode vars
        self.prev_advantage = 0
        self.steps = 0
        self.rewardSum = 0

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
        state = None
        
        while state == None:
            try:
                state = Global_State()
                msg = self.socket_state.recv()
                state.ParseFromString(msg)
                return state
            except Exception as e:
                print("caught timeout:"+str(e))
                self.reset()
                state = None

        return state
    
    def send_commands(self, global_commands):
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"
        robot = c.robot_commands.add()
        robot.id = 0
        
        robot.left_vel, robot.right_vel  = get_action_from_command(global_commands, self.my_agent,
                                                                   self.ball, self.target, self.adv_goal)

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
        #print("DEBUG")
        d = Global_Debug()
        pose = d.final_poses.add()
        pose.id = 0
        pose.x = global_debug[0]
        pose.y = global_debug[1]
        pose.yaw = global_debug[2]

        buf = d.SerializeToString()
        if (self.is_team_yellow):
            self.socket_debug1.send(buf)
        else:
            self.socket_debug2.send(buf)

    def step(self, global_commands):
        #send the command:
        self.send_commands(global_commands)
        #register current simulation timestamp:
        sentTime = self.receive_state().time
        #wait until the espected timestamp arrives
        currentTime = sentTime
        while currentTime<sentTime+cmd_wait:
            rcvd_state = self.receive_state()
            currentTime = rcvd_state.time
            if (currentTime<sentTime): # a simulator crash and restart?
                break
        
        #prev_state = self.last_state    
        self.last_state, reward, done = self.parse_state(rcvd_state)

        return self.last_state, reward, done, {}
        
    def reset(self):
        print("RESET\nAcum_reward:%.5f"%(self.rewardSum))
        self.initVars()

        # Send SIGKILL (on Linux)
        self.p.terminate()
        returncode = self.p.wait()
        # Empty buffers
        while(True):
            socks = dict(self.poller.poll(5))
            if self.socket_state in socks and socks[self.socket_state] == zmq.POLLIN:
                #discard messages
                message = self.socket_state.recv()
            else:
                break

        self.p = subprocess.Popen([path_simulator, '-r', str(command_rate), 
                                  '-p', str(self.port)] + self.parameters)
        self.last_state, reward, done = self.parse_state(self.receive_state())
        return self.last_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode='human', close=False):
        pass

    def potentialFunc(self, t_yellow, t_blue, u_B2OG, u_B2G, u_R2B, u_Col, state_terminal):
        #Based on artivle - DEEP REINFORCEMENTLEARNING   IN PARAMETERIZED ACTION SPACE 
        #potential to adversary goal
        p_B2G = np.linalg.norm(np.array((self.ball["x"],self.ball["y"]))-np.array((160,65)))/100
        #potential to own goal
        p_B2OG = 0
        #potential controled robot to ball
        p_R2B = np.linalg.norm(np.array((self.ball["x"],self.ball["y"]))-np.array((self.my_agent["x"],self.my_agent["y"])))/100
        #potential collision
        team_col, adv_col, wall_col, ball_col = check_collision({"x":self.my_agent["x"],"y":self.my_agent["y"]},
                                                                {"x":self.ball["x"],"y":self.ball["y"]}, 
                                                                t_yellow, t_blue, self.is_team_yellow)
        p_Col = 0.5*ball_col

        if (self.old_p_B2G == None):
            self.old_p_B2G = p_B2G
            self.init_p_B2G = p_B2G

        if (self.old_p_B2OG == None):
            self.old_p_B2OG = p_B2OG
            self.init_p_B2OG = p_B2OG

        if (self.old_p_R2B == None):
            self.old_p_R2B = p_R2B
            self.init_p_R2B = p_R2B

        if (self.old_p_Col == None):
            self.old_p_Col = p_Col
            self.init_p_Col = p_Col

        if (state_terminal == False):
            potencial = u_B2OG*(p_B2OG - self.old_p_B2OG) + u_B2G*(p_B2G - self.old_p_B2G) + \
                        u_R2B*(p_R2B - self.old_p_R2B) + u_Col*(p_Col - self.old_p_Col)

            self.old_p_B2G = p_B2G
            self.old_p_B2OG = p_B2OG
            self.old_p_R2B = p_R2B
            self.old_p_Col = p_Col
        else:
            potencial = u_B2OG*(self.init_p_B2OG-self.old_p_B2OG) + u_B2G*(self.init_p_B2G-self.old_p_B2G) + \
                        u_R2B*(self.init_p_R2B-self.old_p_R2B) + u_Col*(self.init_p_Col-self.old_p_Col)

            self.old_p_B2G = None
            self.old_p_B2OG = None
            self.old_p_R2B = None
            self.old_p_Col = None

        return potencial

    def parse_state(self, state):

        #***********************get state of environment******************************

        env_state, balls, robot = get_observation_from_state(state, mode="self_centered")    
        self.ball = balls[0]

        self.my_agent = robot
        #***************************** get reward *****************************
        done = False
        reward = 0;
        if self.is_team_yellow:
            advantage = state.goals_yellow - state.goals_blue
        else:
            advantage = state.goals_blue - state.goals_yellow
        
        if (advantage != self.prev_advantage):
            #terminal state
            reward = 5*(advantage - self.prev_advantage) + self.potentialFunc(state.robots_yellow, state.robots_blue, 
                                                                            self.u_B2OG, self.u_B2G, self.u_R2B, 
                                                                            self.u_Col, True)

        elif(state.time > 180000):
            done = True
            reward = self.potentialFunc(state.robots_yellow, state.robots_blue, 
                                        self.u_B2OG, self.u_B2G, self.u_R2B, 
                                        self.u_Col, True)

        else:
            reward = self.potentialFunc(state.robots_yellow, state.robots_blue, 
                                        self.u_B2OG, self.u_B2G, self.u_R2B, 
                                        self.u_Col, False)

        #**********************************************************************

        self.prev_advantage = advantage
        self.rewardSum = self.rewardSum + reward
        self.steps = self.steps + 1
        
        return env_state, reward, done
