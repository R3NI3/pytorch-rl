# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:41:11 2018

@author: hans
"""

import pygame
from pygame.locals import *

from core.envs.vss_interface.command_pb2 import *
from core.envs.vss_interface.debug_pb2 import *
from core.envs.vss_interface.state_pb2 import *

import numpy as np
import zmq
import google.protobuf.text_format
import gym.spaces
from gym import spaces
import math

import subprocess

path_viewer = 'vss_sim/VSS-Viewer'
path_simulator = 'vss_sim/VSS-Simulator'
command_rate = 330 #ms
cmd_wait = 266 # 1/4 of 60 frames x (1s in ms)/fps  

class KeyboardControl:
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        self.cmd = 3
        
        self.KRHO = 2
        self.RHO_INC = 20 #increase rho, so it can catch up with the ball
        self.KALPHA = 2.5
        self.KBETA = -2
        self.HALF_AXIS = 8
        self.WHEEL_RADIUS = 2
        self.BALL_APPROACH = -20
        self.decAlpha = 0.3
        self.decLin = 0.9
        self.decAng = 0.9
        
        self.x = 0
        self.y = 0 
        self.theta = 0
        self.target_x = None
        self.target_y = None
        self.ball_x = None
        self.ball_y = None
        self.prev_ball_potential = None
        self.prev_robot_ball_dist = None
        self.linearSpeed = 0
        self.angularSpeed = 0
        self.send_time = 0
#        self.maxX = -1000
#        self.minX = 1000
#        
#        self.maxVx = -1000
#        self.minVx = 1000
#        
#        self.maxT = -1000
#        self.minT = 1000
#        
#        self.maxVt = -1000
#        self.minVt = 1000
        
    def setup_connections(self, ip='127.0.0.1', port=5555, is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, '-a', '-d', '-r', str(command_rate), '-p', str(self.port)])
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
        state = None
        
        while state == None:
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
                self.reset()
                state = None

        return state

    def clip(self, val, vmin, vmax):
        return min(max(val, vmin), vmax)
        
    def roundTo5(self, x, base=5):
        return int(base * round(float(x)/base))
    
    def send_commands(self, global_commands):
        #print(".")
        self.cmd = global_commands
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"
        robot = c.robot_commands.add()
        robot.id = 0

#        if (self.target_x == None): #Reset target to robot position if None
#            self.target_x = self.x
#            self.target_y = self.y
#        else: # Apply decrements
#            self.target_x = self.target_x*self.decAlpha + self.x*(1-self.decAlpha)
#            self.target_y = self.target_y*self.decAlpha + self.y*(1-self.decAlpha)

#        if (self.cmd == 1):
#            self.cmd = 2
#        else:
#            self.cmd = 1
#            
        #global_commands = 0#self.cmd
        
        self.linearSpeed = self.linearSpeed*self.decLin
        self.angularSpeed = self.angularSpeed*self.decAng

        if global_commands == 0: #default command: carry ball to goal
            goal_x = 165
            goal_y = 65
            goal_theta = math.atan2((self.ball_y-goal_y),(self.ball_x-goal_x))
            rho  = np.sqrt((self.ball_x-self.x)*(self.ball_x-self.x) + (self.ball_y-self.y)*(self.ball_y-self.y))
            apr = max(self.BALL_APPROACH,-rho/2)
            ball_appr_x = self.ball_x - apr*math.cos(goal_theta)
            ball_appr_y = self.ball_y - (apr/2)*math.sin(goal_theta)
            robot.left_vel, robot.right_vel = self.getWheelSpeeds(ball_appr_x, ball_appr_y, goal_theta, self.KRHO, self.RHO_INC)
            #self.target_x = None
            #print(str(global_commands)+":X:%.1f"%(self.x)+ " Y:%.1f"%(self.y))
            #self.send_debug([ball_appr_x, ball_appr_y, goal_theta])
        else:
            self.dict = {1:(15,0),
                         2:(-15,0),
                         3:(0,30),
                         4:(0,-30),
                         5:(0,0)
                        }
            #self.target_x = self.clip(self.target_x + self.dict[global_commands][0], -20, 190)
            #self.target_y = self.clip(self.target_y + self.dict[global_commands][1], -20, 150)
            #target_theta = math.atan2((self.target_y-self.y),(self.target_x-self.x))
            #self.send_debug([self.target_x,self.target_y,target_theta])
            self.linearSpeed = self.clip(self.linearSpeed + self.dict[global_commands][1],-80,80)
            self.angularSpeed = self.clip(self.angularSpeed + self.dict[global_commands][0],-60,60)
            robot.left_vel = self.linearSpeed - self.angularSpeed
            robot.right_vel  = self.linearSpeed + self.angularSpeed
            
            #robot.left_vel, robot.right_vel = self.getWheelSpeeds(self.target_x, self.target_y, target_theta, 4)
            #print(str(global_commands)+":X:%.1f"%(self.x)+ " DX:%.1f"%(self.target_x)+ " Y:%.1f"%(self.y)+ " DY:%.1f"%(self.target_y)+" DT:%.1f"%math.degrees(target_theta))

        #print("lin:"+str(self.linearSpeed)+"\tang:"+str(self.angularSpeed))
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

    def getWheelSpeeds(self, target_x, target_y, target_theta, KRHO=1, rho_inc = 0):
        rho  = np.sqrt((target_x-self.x)*(target_x-self.x) + (target_y-self.y)*(target_y-self.y))
                    
        lambda_ = math.atan2((target_y-self.y),(target_x-self.x))
        alpha = self.to180range(lambda_ - self.theta)
        beta = self.to180range(-target_theta - alpha)
        
        reverse = False
        if (abs(alpha)>math.pi/2):
            self.theta = self.to180range(self.theta+math.pi)
            alpha = self.to180range(lambda_ - self.theta)
            reverse = True
            #print("Reverse")

        self.linearSpeed = KRHO*(rho+rho_inc)
        self.angularSpeed = self.KALPHA*alpha + self.KBETA*beta

        if reverse:
            self.linearSpeed = -self.linearSpeed
            #self.angularSpeed = -self.angularSpeed
            
        #print (math.degrees(self.theta), math.degrees(target_theta), math.degrees(lambda_), math.degrees(alpha), math.degrees(beta))

        leftSpeed  = (self.linearSpeed - self.angularSpeed*self.HALF_AXIS)/self.WHEEL_RADIUS
        rightSpeed = (self.linearSpeed + self.angularSpeed*self.HALF_AXIS)/self.WHEEL_RADIUS
        
        #print(self.linearSpeed, self.angularSpeed)
                
        return rightSpeed, leftSpeed

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
        
        #print("t1:%d"%sentTime, "t2:%d"%currentTime, "dt:%d"%(currentTime-sentTime))
        #print("t1:%d"%self.send_time, "t2:%d"%currentTime, "dt:%d"%(currentTime-self.send_time))
        #print("dt_cmd:%d"%(currentTime-sentTime), "dt_step:%d"%(currentTime-self.send_time))
        #self.send_time = currentTime
        
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
        self.prev_ball_potential = None
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


    def normX(self, x):
        return x/170.0

    def normVx(self, vx):
        return vx/80.0

    def normVt(self, vt):
        return vt/12

    def parse_state(self, state):
        for idx, ball in enumerate(state.balls):
            #real values
            ball_state = (self.normX(ball.pose.x), self.normX(ball.pose.y),
                          self.normVx(ball.v_pose.x), self.normVx(ball.v_pose.y))

            self.ball_x = ball.pose.x
            self.ball_y = ball.pose.y
            #estimated values
            #estimated_ball_state = (ball.k_pose.x, ball.k_pose.y,ball.k_v_pose.x, ball.k_v_pose.y)

        t1_state = ()
        #estimated_t1_state = ()
        for idx, t1_robot in enumerate(state.robots_yellow):
            #real values
            #encode yaw as sin(yaw) and cos(yaw)
            t1_state += (self.normX(t1_robot.pose.x), self.normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         self.normVx(t1_robot.v_pose.x), self.normVx(t1_robot.v_pose.y), self.normVt(t1_robot.v_pose.yaw))

            if (idx==0):
                self.x = t1_robot.pose.x
                self.y = t1_robot.pose.y
                self.theta = t1_robot.pose.yaw
                self.linearSpeed = math.sqrt(t1_robot.v_pose.x*t1_robot.v_pose.x + t1_robot.v_pose.y*t1_robot.v_pose.y)
                if (abs(math.atan2(t1_robot.v_pose.y,t1_robot.v_pose.x)-self.theta)>math.pi/2):
                    self.linearSpeed = - self.linearSpeed
                
                self.angularSpeed = t1_robot.v_pose.yaw*8/1.63 # VangWheel = vang*RobotWidth/WheelRadius

            #estimated values
            #estimated_t1_state += (t1_robot.k_pose.x, t1_robot.k_pose.y, t1_robot.k_pose.yaw,t1_robot.k_v_pose.x, t1_robot.k_v_pose.y, t1_robot.k_v_pose.yaw)

        t2_state = ()
        #estimated_t2_state = ()
        for idx, t2_robot in enumerate(state.robots_blue):
            #real values
            t2_state += (self.normX(t2_robot.pose.x), self.normX(t2_robot.pose.y), math.sin(t2_robot.pose.yaw), math.cos(t2_robot.pose.yaw),
                         self.normVx(t2_robot.v_pose.x), self.normVx(t2_robot.v_pose.y), self.normVt(t2_robot.v_pose.yaw))            #estimated values
            #estimated_t2_state += (t2_robot.k_pose.x, t2_robot.k_pose.y, t2_robot.k_pose.yaw, t2_robot.k_v_pose.x, t2_robot.k_v_pose.y, t2_robot.k_v_pose.yaw)

        same_team_col, adv_team_col, wall_col = self.check_collision(state.robots_yellow, state.robots_blue)
        #penalty = -0.2/(1+abs(0.1*self.linearSpeed)) - 0.2*same_team_col - 0.1*wall_col - 0.1*adv_team_col

        done = False
        reward = 0;
        if self.is_team_yellow:
            reward = state.goals_yellow - state.goals_blue
        else:
            reward = state.goals_blue - state.goals_yellow

        if(reward != 0):
            #pdb.set_trace()
            reward = 600*reward
            done = True
            #print("******************GOAL****************")
            #print("Reward:"+str(reward))
#        else:
#            ball = np.array((self.ball_x,self.ball_y))
#            rb1 = np.array((self.x,self.y))
#            robot_ball_dist = np.linalg.norm(ball-rb1)
#            
#            #Compute reward:
#            ball_potential = ((self.ball_x-80)**3-(self.ball_x-80)*(self.ball_y-65)**2)*0.000175+self.ball_x
#            #https://academo.org/demos/3d-surface-plotter/?expression=x%2B((x-80)%5E3-(x-80)*(y-65)%5E2)*0.000175&xRange=-0%2C165&yRange=0%2C130&resolution=58
#            if (self.prev_ball_potential == None):
#                reward = 0
#            else:
#                ball_to_goal_reward =  ball_potential - self.prev_ball_potential
#                robot_to_ball_reward = self.prev_robot_ball_dist-robot_ball_dist
#                #print(".rob:("+"%.1f"%self.ball_x+ ", %.1f"%self.ball_y+") %.2f"%ball_to_goal_reward)
#
#                if (robot_ball_dist>15 or ball_to_goal_reward<0):#No donuts if the ball is far
#                    ball_to_goal_reward = 0
#                    
#                if (robot_to_ball_reward<0):
#                    robot_to_ball_reward = 0
#
#                reward = ((0.2*robot_to_ball_reward + ball_to_goal_reward) + penalty)
#                #if (abs(reward)>2):
#                print(".rob:("+"%.1f"%rb1[0]+ ", %.1f"%rb1[1]+") %.2f" %(0.2*robot_to_ball_reward) + ", %.2f" %ball_to_goal_reward + ", %.2f" %penalty + ", %.2f" %reward + " cmd:%d"%self.cmd)
#
#            self.prev_robot_ball_dist = robot_ball_dist
#            self.prev_ball_potential = ball_potential
        #print("lin:%.1f"%self.linearSpeed+"\tang:%.1f"%self.angularSpeed)
        #print(t1_state)

        #print("Reward:"+str(reward))
        #time.sleep(0.100)#200ms

        env_state = ball_state + t1_state + t2_state

        print ("XY: (%.2f"%t1_state[0]+",%.2f"%t1_state[1]+") Theta: (%.2f"%t1_state[2]+",%.2f"%t1_state[3]+") Vxy: (%.2f"%t1_state[4]+", %.2f"%t1_state[5]+") Vtheta: %.2f"%t1_state[6])
 
        return np.array(env_state), reward/600.0, done

    def display(self,str_):
        text = self.font.render(str_, True, (255, 255, 255), (159, 182, 205))
        textRect = text.get_rect()
        textRect.centerx = self.screen.get_rect().centerx
        textRect.centery = self.screen.get_rect().centery
    
        self.screen.blit(text, textRect)
        pygame.display.update()
    
    def loop(self):
        pygame.init()
        clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode( (640,480) )
        pygame.display.set_caption('Python numbers')
        self.screen.fill((159, 182, 205))
    
        self.font = pygame.font.Font(None, 17)
        
        done = False
        while not done:
            rcvd_state = self.receive_state()
            while rcvd_state == None:
                rcvd_state = self.receive_state()
    
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            key = ""
            cmd = 5
            if keys[K_q]:
                done = True
                print("quit")
            elif keys[K_a]:
                key = "a"
                cmd = 0
                print(key, cmd)
            elif keys[K_LEFT]:
                key = "<"
                cmd = 1
                print(key, cmd)
            elif keys[K_RIGHT]:
                key = ">"
                cmd = 2
                print(key, cmd)
            elif keys[K_UP]:
                key = "A"
                cmd = 3
                print(key, cmd)
            elif keys[K_DOWN]:
                key = "v"
                cmd = 4
                print(key, cmd)
            
            self.display(str(key))
        
            self.send_commands(cmd)
            #register current simulation timestamp:
            sentTime = self.receive_state().time
            #wait until the espected timestamp arrives
            currentTime = sentTime
            while currentTime<sentTime+cmd_wait:
                rcvd_state = self.receive_state()
                currentTime = rcvd_state.time
                if (currentTime<sentTime): # a simulator crash and restart?
                    break

            #for e in pygame.event.get(): 
            #    pass # proceed other events. 
                # always call event.get() or event.poll() in the main loop
            
            #print("t1:%d"%sentTime, "t2:%d"%currentTime, "dt:%d"%(currentTime-sentTime))
            #print("t1:%d"%self.send_time, "t2:%d"%currentTime, "dt:%d"%(currentTime-self.send_time))
            #print("dt_cmd:%d"%(currentTime-sentTime), "dt_step:%d"%(currentTime-self.send_time))
            #self.send_time = currentTime
            
            #prev_state = last_state    
            last_state, reward, _ = self.parse_state(rcvd_state)
            clock.tick(120)

kcontrol = KeyboardControl()
kcontrol.setup_connections(port=7777)
kcontrol.loop()
pygame.quit()