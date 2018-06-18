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
import math

import subprocess
import os
import signal

path_viewer = 'vss_sim/VSS-Viewer'
path_simulator = 'vss_sim/VSS-Simulator'
command_rate = 300 #ms
cmd_wait = 0.200 #s

class SoccerEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        self.prev_robot_ball_dist = None
        self.prev_ball_goal_dist = None
        self.cmd = 3
        
        self.KRHO = 1
        self.RHO_INC = 30 #increase rho a little bit, so it can catch up with the ball
        self.KALPHA = 1.8
        self.KBETA = -0.6
        self.HALF_AXIS = 8
        self.WHEEL_RADIUS = 2
        self.BALL_X_APPROACH = -5
        
        self.x = 0
        self.y = 0 
        self.theta = 0
        self.target_x = None
        self.target_y = None
        self.ball_x = None
        self.ball_y = None
        self.decAlpha = 0.75

    def setup_connections(self, ip='127.0.0.1', port=5555, is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, '-a', '-d', '-r', str(command_rate), '-p', str(self.port)])
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
        self.action_space = spaces.Box(low=-100, high=100, dtype=np.float32, shape=(5,)) #wheels velocities

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
                    #print("discard");
                else:
                    break
            return state
        except Exception as e:
            print("caught timeout:"+str(e))
             
        return None

    def clip(self, val, vmin, vmax):
        return min(max(val, vmin), vmax)
        
    def roundTo5(self, x, base=5):
        return int(base * round(float(x)/base))
    
    def send_commands(self, global_commands):
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"
        robot = c.robot_commands.add()
        robot.id = 0
                
        if (self.target_x == None): #Reset target to robot position if None
            self.target_x = self.x
            self.target_y = self.y
        else: # Apply decrements
            self.target_x = self.target_x*self.decAlpha + self.x*(1-self.decAlpha)
            self.target_y = self.target_y*self.decAlpha + self.y*(1-self.decAlpha)
        
        #global_commands = 0
        
        if global_commands == 0: #default command: carry ball to goal
            goal_x = 165
            goal_y = 65
            goal_theta = math.atan2((self.ball_y-goal_y),(self.ball_x-goal_x))
            robot.left_vel, robot.right_vel = self.getWheelSpeeds(self.ball_x+self.BALL_X_APPROACH, self.ball_y, goal_theta, self.RHO_INC)
        else:
            self.dict = {1:(-5,0),
                         2:(5,0),
                         3:(0,-5),
                         4:(0,5)
                        }
            self.target_x = self.clip(self.target_x + self.dict[global_commands][0], -20, 190)
            self.target_y = self.clip(self.target_y + self.dict[global_commands][1], -20, 150)
            self.send_debug([self.target_x,self.target_y,math.atan2((self.target_y-self.y),(self.target_x-self.x))])
            robot.left_vel, robot.right_vel = self.getWheelSpeeds(self.target_x, self.target_x, math.atan2((self.target_y-self.y),(self.target_x-self.x)))
        
        #print("DX:%.1f"%(self.target_x-self.x)+ " DY:%.1f"%(self.target_y-self.y))
        #print("lin:"+str(self.speed_lin)+"\tang:"+str(self.speed_ang)+"\tvel:["+str(robot.left_vel)+","+str(robot.right_vel)+"]")
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
            
    def to180range(self,angle):
        angle = math.fmod(angle, 2 * math.pi)
        if (angle < -math.pi):
            angle = angle + 2 * math.pi
        elif (angle > math.pi):
            angle = angle - 2 * math.pi
        
        return angle

    def getWheelSpeeds(self, target_x, target_y, target_theta, rho_inc = 0):
        rho  = np.sqrt((target_x-self.x)*(target_x-self.x) + (target_y-self.y)*(target_y-self.y))+rho_inc
        lambda_ = math.atan2((target_y-self.y),(target_x-self.x))
        alpha = self.to180range(lambda_ - self.theta)
        reverse = False
        if (abs(alpha)>math.pi/2):
            self.theta = self.to180range(self.theta+math.pi)
            alpha = self.to180range(lambda_ - self.theta)
            reverse = True
            #print("Reverse")
                
        beta = -target_theta - alpha 
        self.linearSpeed = self.KRHO*rho
        self.angularSpeed = self.KALPHA*alpha + self.KBETA*beta
        
        #print (math.degrees(self.theta-lambda_), math.degrees(self.theta), math.degrees(lambda_), math.degrees(alpha))

        leftSpeed  = (self.linearSpeed - self.angularSpeed*self.HALF_AXIS)/self.WHEEL_RADIUS
        rightSpeed = (self.linearSpeed + self.angularSpeed*self.HALF_AXIS)/self.WHEEL_RADIUS

        if reverse:
            return -leftSpeed, -rightSpeed
                
        return rightSpeed, leftSpeed

    def send_debug(self, global_debug):
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
        self.send_commands(global_commands)
        time.sleep(cmd_wait)#wait for the command to became effective
        rcvd_state = self.receive_state()
        while rcvd_state == None:
            self.reset()
            rcvd_state = self.receive_state()
        
        #prev_state = self.last_state    
        self.last_state, reward, done = self.parse_state(rcvd_state)

        #self.debugStep(prev_state, self.last_state, global_commands)
        return self.last_state, reward, done, {}

    def debugStep(self, prev_state, new_state, cmd):
        self.dict = {0:(0,0),
                     1:(10,0), 2:(0,10), 3:(10,10), 4:(-10,10), 5:(10
                     ,-10), 6:(-10,0), 7:(0,-10), 8:(-10,-10)
                    }
        left_vel = self.dict[cmd][0]
        right_vel = self.dict[cmd][1]

        last_x = prev_state[4]
        last_y = prev_state[5]
        last_yaw = prev_state[6]
        
        new_x = new_state[4]
        new_y = new_state[5]
        new_yaw = new_state[6]

        print("prev:", "%.1f" %last_x, "%.1f" %last_y, "%.1f" %last_yaw, left_vel, right_vel)
        print("new: ", "%.1f" %new_x, "%.1f" %new_y, "%.1f" %new_yaw)
        
    def reset(self):
        print('RESET')
        self.prev_robot_ball_dist = None
        self.prev_ball_goal_dist = None
        self.target_x = None
        self.target_y = None
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

        self.p = subprocess.Popen([path_simulator, '-a', '-d', '-r', str(command_rate), '-p', str(self.port)])
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
            reward = 1000*reward
            done = True
            print("******************GOAL****************")
            print("Reward:"+str(reward))
        elif(state.time >= 10):
            done = True
        else:
            ball = np.array((ball_state[0],ball_state[1]))
            goalR = np.array((165,65))
            goalL = np.array((10,65))
            rb1 = np.array((t1_state[0],t1_state[1]))
            robot_ball_dist = np.linalg.norm(ball-rb1)
            ball_goalR_dist = np.linalg.norm(goalR-ball)
            ball_goalL_dist = np.linalg.norm(goalL-ball)
            ball_goal_dist = (ball_goalR_dist - ball_goalL_dist)/2
            if (self.prev_robot_ball_dist == None):
                reward = -0.2
            else:
                ball_reward = self.prev_robot_ball_dist-robot_ball_dist
                goal_reward = self.prev_ball_goal_dist-ball_goal_dist
                
                dist_scale = (50/robot_ball_dist) #50 is about a quarter of the field diagonal
                reward = (ball_reward + 5*goal_reward)*dist_scale - 0.2
                if (abs(reward)>2):
                    print(".rob:("+"%.1f"%rb1[0]+ ", %.1f"%rb1[1]+") " + "bal:("+"%.1f"%ball[0]+", %.1f"%ball[1]+") "+ "%.2f" %ball_reward+ ", %.2f" %goal_reward+ ", %.2f" %reward)
            self.prev_robot_ball_dist = robot_ball_dist
            self.prev_ball_goal_dist = ball_goal_dist

        self.x = t1_state[0]
        self.y = t1_state[1]
        self.theta = t1_state[2]
        self.ball_x = ball_state[0]
        self.ball_y = ball_state[1]

        #print("lin:%.1f"%self.speed_lin+"\tang:%.1f"%self.speed_ang)
        #print(t1_state)
        
        #print("Reward:"+str(reward))
        #time.sleep(0.100)#200ms

        env_state = ball_state + t1_state + t2_state
        #unused infos
        #state.name_yellow
        #state.name_blue
        #print(state.time)

        return np.array(env_state), reward, done

