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
import pdb, os
path_simulator = 'vss_sim/VSS-Simulator'
path_viewer = 'vss_sim/VSS-Viewer'
cmd_rate = '250'
cmd_delay = .050

class ConSoccerEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        self.prev_robot_ball_dist = None
        self.prev_ball_goal_dist = None
        self.socket_state = None
        self.socket_com1 = None
        self.socket_com2 = None
        self.socket_debug1 = None
        self.socket_debug2 = None
        self.time = time.time()
        print(os.getpid())

    def setup_connections(self, ip='127.0.0.1', port=5555, is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, '-r', cmd_rate, '-d', '-a', '-p', str(self.port)])

        # state socket
        self.socket_state_ = self.context.socket(zmq.SUB) #socket to listen vision/simulator
        self.socket_state_.setsockopt(zmq.CONFLATE, 1)
        self.socket_state_.connect ("tcp://localhost:%d" % self.port)
        #self.socket_state.setsockopt_string(zmq.SUBSCRIBE, b"")#allow every topic
        try:
            self.socket_state_.setsockopt(zmq.SUBSCRIBE, b'')
        except TypeError:
            self.socket_state_.setsockopt_string(zmq.SUBSCRIBE, b'')
        self.socket_state_.setsockopt(zmq.LINGER, 0)
        self.socket_state_.setsockopt(zmq.RCVTIMEO, 500)
        

        self.last_state, reward, done = self.parse_state(self.receive_state(self.socket_state_))
        self.init_state = self.last_state
        shape = len(self.last_state)
        #todo fix obs and action spaces
        self.observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(shape,))
        self.action_space = spaces.Box(low=-100, high=100, dtype=np.float32, shape=(2,)) #wheels velocities
        #self.socket_state_tmp.close()

        # Send SIGKILL (on Linux)
        self.p.terminate()
        returncode = self.p.wait()
        self.socket_state_.close()

    def connect(self):
        print("CONNECT")
        self.close_connections()
        # start simulation
        self.context.destroy
        self.context = zmq.Context()
        self.p = subprocess.Popen([path_simulator, '-r', cmd_rate, '-d', '-a', '-p', str(self.port)])
        # state socket
        self.socket_state = self.context.socket(zmq.SUB) #socket to listen vision/simulator
        self.socket_state.setsockopt(zmq.CONFLATE, 1)
        self.socket_state.connect ("tcp://localhost:%d" % self.port)
        #self.socket_state.setsockopt_string(zmq.SUBSCRIBE, b"")#allow every topic
        try:
            self.socket_state.setsockopt(zmq.SUBSCRIBE, b'')
        except TypeError:
            self.socket_state.setsockopt_string(zmq.SUBSCRIBE, b'')
        self.socket_state.setsockopt(zmq.LINGER, 0)
        self.socket_state.setsockopt(zmq.RCVTIMEO, 50000)
        #self.socket_state.setsockopt(zmq.CONFLATE, 1)

        # commands socket
        self.socket_com1 = self.context.socket(zmq.PAIR) #socket to Team 1
        self.socket_com1.connect ("tcp://localhost:%d" % (self.port+1))

        self.socket_com2 = self.context.socket(zmq.PAIR) #socket to Team 2
        self.socket_com2.connect ("tcp://localhost:%d" % (self.port+2))

        # debugs socket
        self.socket_debug1 = self.context.socket(zmq.PAIR) #debug socket to Team 1
        self.socket_debug1.connect ("tcp://localhost:%d" % (self.port+3))

        self.socket_debug2 = self.context.socket(zmq.PAIR) #debug socket to Team 2
        self.socket_debug2.connect ("tcp://localhost:%d" % (self.port+4))
        #end

        self.last_state, reward, done = self.parse_state(self.receive_state(self.socket_state))
        self.init_state = self.last_state
        shape = len(self.last_state)
        #todo fix obs and action spaces
        self.observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(shape,))

        # Initialize poll set
        self.poller.register(self.socket_state, zmq.POLLIN)

    def close_connections(self):
        if(self.socket_state != None):
            self.socket_state.close()#disconnect ("tcp://localhost:%d" % self.port)
            self.socket_com1.close()#disconnect ("tcp://localhost:%d" % (self.port+1))
            self.socket_com2.close()#disconnect ("tcp://localhost:%d" % (self.port+2))
            self.socket_debug1.close()#disconnect ("tcp://localhost:%d" % (self.port+3))
            self.socket_debug2.close()#disconnect ("tcp://localhost:%d" % (self.port+4))
            self.context.destroy()

    def __del__(self):
        self.close_connections()
        # Send SIGTER (on Linux)
        self.p.terminate()
        # Wait for process to terminate
        returncode = self.p.wait()
        print('Process destroyed',self.port)
        #self.render(close=True)

    def receive_state(self, socket_state):
        try:
            state = Global_State()
            msg = socket_state.recv()
            count = 0
            while(count < 10):
                socks = dict(self.poller.poll(10))
                if socket_state in socks and socks[socket_state] == zmq.POLLIN:
                    #discard messages
                    msg = socket_state.recv()
                    count += 1
                    #print(count)
                else:
                    break
            state.ParseFromString(msg)
            return state
        except Exception as e:
            print("caught timeout:"+str(e))
             
        return None

    def send_commands(self, global_commands):
        c = Global_Commands()
        #act_time = time.time()
        #print('total',self.time - act_time)
        #self.time = act_time
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"

        robot = c.robot_commands.add()
        robot.id = 0
        lin_vel = np.clip(global_commands[0][0]*20, -20, 20)
        ang_vel = np.clip(global_commands[0][1]*20, -20, 20)
        robot.left_vel  = (lin_vel - ang_vel)
        robot.right_vel = (lin_vel + ang_vel)
        #robot.left_vel = 10
        #robot.right_vel = -10
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
        time.sleep(cmd_delay)
        rcvd_state = self.receive_state(self.socket_state)
        while rcvd_state == None:
            self.reset()
            rcvd_state = self.receive_state(self.socket_state)
        #print('cmd-rcv',self.time - time.time())
        self.last_state, reward, done = self.parse_state(rcvd_state)
        return self.last_state, reward, done, {}

    def reset(self):
        print('RESET')
        if(self.socket_state == None):
            self.connect()
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

        self.p = subprocess.Popen([path_simulator, '-r', cmd_rate, '-a', '-d', '-p', str(self.port)])
        self.last_state, reward, done = self.parse_state(self.receive_state(self.socket_state))
        return self.last_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_normalize(self, ball, t1, t2):
        max_vel_x_y = 20
        max_v_yaw = 4.2 #max 21
        max_x = 170
        max_y = 130
        max_yaw = 2*np.pi
        norm_ball = ()
        norm_t1 = ()
        norm_t2 = ()

        #ball x,y,vx,vy = tamanho 4
        i = int(len(ball)/4)
        l = 4
        for idx in range(i):
            norm_ball += ((ball[l*idx + 0])/(max_x), (ball[l*idx + 1])/(max_y),
                          (ball[l*idx + 2]-(-max_vel_x_y))/(2*max_vel_x_y), (ball[l*idx + 3]-(-max_vel_x_y))/(2*max_vel_x_y))

        #t1 x,y,yaw,vx,vy,v_yaw = tamaho 6
        i = int(len(t1)/6)
        l = 6
        for idx in range(i):
            norm_t1 += ((t1[l*idx + 0])/(max_x), (t1[l*idx + 1])/(max_y), (t1[l*idx + 2]-(-max_yaw/2))/max_yaw,
                        (t1[l*idx + 3]-(-max_vel_x_y))/(2*max_vel_x_y), (t1[l*idx + 4]-(-max_vel_x_y))/(2*max_vel_x_y), (t1[l*idx + 5]-(-max_v_yaw))/(2*max_v_yaw))
        
        #t2 x,y,yaw,vx,vy,v_yaw = tamanho 6
        i = int(len(t2)/6)
        l = 6
        for idx in range(i):
            norm_t2 += ((t2[l*idx + 0])/(max_x), (t2[l*idx + 1])/(max_y), (t2[l*idx + 2]-(-max_yaw/2))/max_yaw,
                        (t2[l*idx + 3]-(-max_vel_x_y))/(2*max_vel_x_y), (t2[l*idx + 4]-(-max_vel_x_y))/(2*max_vel_x_y), (t2[l*idx + 5]-(-max_v_yaw))/(2*max_v_yaw))

        return norm_ball, norm_t1, norm_t2


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

    def check_collision(self, t_yellow, t_blue):
        robot_id = 0
        COL_DIST=12
        if(self.is_team_yellow):
            robot_x = t_yellow[0].pose.x
            robot_z = t_yellow[0].pose.y
        else:
            robot_x = t_blue[0].pose.x
            robot_z = t_blue[0].pose.y

        same_team_col = False
        adv_team_col = False
        wall_col = False
        
        #for every robot in robots team
        for idx, t1_robot in enumerate(t_yellow):
            tmp_robot_x = t1_robot.pose.x
            tmp_robot_z = t1_robot.pose.y
            if (np.linalg.norm([tmp_robot_x-robot_x,tmp_robot_z-robot_z]) < COL_DIST):
                if(self.is_team_yellow):
                    if (idx !=robot_id):
                        same_team_col = True
                else:
                    adv_team_col = True

        #for every robot in adversary team
        for idx, t2_robot in enumerate(t_blue):
            tmp_robot_x = t2_robot.pose.x
            tmp_robot_z = t2_robot.pose.y
            if (np.linalg.norm([tmp_robot_x-robot_x,tmp_robot_z-robot_z]) < COL_DIST):
                if(not self.is_team_yellow):
                    if (idx !=robot_id):
                        same_team_col = True
                else:
                    adv_team_col = True

        #def wall collision when:
        #walls z +- 0 ou z +- 130
        #walls 45 < z < 85 e x +- 0,x+-170 
        if (robot_z < 6 or robot_z > 124):
            wall_col = True
        else:
            if (robot_z < 51 or robot_z > 79): #outside goal height +- 6
                if (robot_x < 16 or robot_x > 154): #near goal line walls
                    wall_col = True
            else: #inside goal height
                if (robot_x < 6 or robot_x > 164):
                    wall_col = True

        #TODO: test corners

        return same_team_col, adv_team_col, wall_col

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
        same_team_col, adv_team_col, wall_col = self.check_collision(state.robots_yellow, state.robots_blue)
        penalty = -0.1*same_team_col - 0.05*wall_col - 0.05*adv_team_col
        reward = 0;
        if self.is_team_yellow:
            reward = state.goals_yellow - state.goals_blue
        else:
            reward = state.goals_blue - state.goals_yellow

        if(reward != 0):
            #pdb.set_trace()
            print("******************GOAL****************")
            reward = 30*reward
            done = True
        elif(state.time >= 20):
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
                if (self.prev_robot_ball_dist-robot_ball_dist > 0.1):
                    reward = 0.1 + self.prev_robot_ball_dist-robot_ball_dist
                elif (self.prev_robot_ball_dist-robot_ball_dist < -0.1):
                    reward = -0.2 + self.prev_robot_ball_dist-robot_ball_dist
                else:
                    reward = -0.5
                reward += self.prev_ball_goal_dist - ball_goal_dist
                reward += penalty
            self.prev_robot_ball_dist = robot_ball_dist
            self.prev_ball_goal_dist = ball_goal_dist
        #print("Reward:"+str(reward))
        #time.sleep(0.100)#200ms
        ball_state, t1_state, t2_state = self.state_normalize(ball_state, t1_state, t2_state)
        env_state = ball_state + t1_state + t2_state
        #unused infos
        #state.name_yellow
        #state.name_blue
        #print(state.time)

        return np.array(env_state), reward/300, done

