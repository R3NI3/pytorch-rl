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
command_rate = 330 #ms
cmd_wait = 266 # 1/4 of 60 frames x (1s in ms)/fps  

class SoccerEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        #start simulator and viewer
        self.is_rendering = False
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        #self.last_state = np.array((0,0,0,0)+(0,0,0,0,0,0)+(0,0,0,0,0,0))
        
        self.initVars()
        
        # Constants of the World
        self.KRHO = 1.75
        self.RHO_INC = 20 #increase rho a little bit, so it can catch up with the ball
        self.KALPHA = 4
        self.KBETA = -0.4
        self.HALF_AXIS = 8
        self.WHEEL_RADIUS = 2
        self.BALL_APPROACH = -20
        self.decAlpha = 0.3
        self.decLin = 0.9
        self.decAng = 0.6

        #Positive potential constants
        self.u_B2G = 1
        self.u_R2B = 0.5
        #Negative potential constants
        self.u_B2OG = 1
        self.u_Col = 0.5
        
    def setup_connections(self, ip='127.0.0.1', port=5555, parameters = '-a' , is_team_yellow = True):
        self.ip = ip
        self.port = port
        self.parameters = parameters
        self.is_team_yellow = is_team_yellow
        self.context = zmq.Context()
        # start simulation
        self.p = subprocess.Popen([path_simulator, parameters, '-d', '-r', str(command_rate), '-p', str(self.port)])
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
        self.x = 0
        self.y = 0 
        self.theta = 0
        self.target_x = None
        self.target_y = None
        self.target_theta = None
        self.target_rho = 0
        self.ball_x = None
        self.ball_y = None

        self.linearSpeed = 0
        self.angularSpeed = 0

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

    def clip(self, val, vmin, vmax):
        return min(max(val, vmin), vmax)
    
    def send_commands(self, global_commands):
        c = Global_Commands()
        c.id = 0
        c.is_team_yellow = self.is_team_yellow
        c.situation = 0
        c.name = "Teste"
        robot = c.robot_commands.add()
        robot.id = 0
        
        if self.target_theta == None:
            self.target_theta = self.theta

        if self.target_x == None:
            self.target_x = self.x
            self.target_y = self.y
            self.target_rho = 0
            self.target_theta = self.theta

        self.target_rho = math.sqrt((self.target_x - self.x)*(self.target_x - self.x) + (self.target_y - self.y)*(self.target_y - self.y))

        if abs(self.target_rho) > 0.01:
            self.target_theta = math.atan2((self.target_y - self.y),(self.target_x - self.x))

        if abs(self.smallestAngleDiff(self.target_theta, self.theta))>math.pi/2:
            self.target_rho   = -self.target_rho
            self.target_theta =  self.to180range(self.target_theta+math.pi)

        if global_commands == 0: #default command: carry ball to goal
            goal_x = 185
            goal_y = 65
            goal_theta = math.atan2((self.ball_y-goal_y),(self.ball_x-goal_x))
            rho  = np.sqrt((self.ball_x-self.x)*(self.ball_x-self.x) + (self.ball_y-self.y)*(self.ball_y-self.y))
            apr = max(self.BALL_APPROACH,-rho/2)
            self.target_x = ball_appr_x = self.ball_x - apr*math.cos(goal_theta)
            self.target_y = ball_appr_y = self.ball_y - (apr/2)*math.sin(goal_theta)
            self.target_theta = goal_theta
            robot.left_vel, robot.right_vel = self.getWheelSpeeds(ball_appr_x, ball_appr_y, goal_theta, self.KRHO, self.RHO_INC)
            #self.target_x = None
            #print(str(global_commands)+":X:%.1f"%(self.x)+ " Y:%.1f"%(self.y))
            #self.send_debug([ball_appr_x, ball_appr_y, goal_theta])
        else:
            self.dict = {1:(-math.pi/36,0),
                         2:( math.pi/36,0),
                         3:(0, 12),
                         4:(0,-12),
                         5:(0,0)
                        }
            
            self.target_rho = self.clip(self.target_rho + self.dict[global_commands][1],-60,60)        
            self.target_theta = self.to180range(self.target_theta+self.dict[global_commands][0])

            if self.target_rho<0:
                rbt_theta = self.to180range(self.theta+math.pi)
                cmd_theta = self.to180range(self.target_theta+math.pi)
                #print("target_rho:%.1f"%self.target_rho +" rbt_t:%.1f"%math.degrees(rbt_theta)+" cmd_t:%.1f"%math.degrees(cmd_theta))
            else:
                rbt_theta = self.theta
                cmd_theta = self.target_theta

            self.angularSpeed = self.clip(-30*self.smallestAngleDiff(cmd_theta,rbt_theta),-30, 30)
            self.linearSpeed = 1.5*self.target_rho

            if global_commands!=5:
                self.target_x = self.clip(self.x + self.target_rho*math.cos(self.target_theta),0,170)
                self.target_y = self.clip(self.y + self.target_rho*math.sin(self.target_theta),0,130)

            robot.left_vel = self.linearSpeed - self.angularSpeed
            robot.right_vel  = self.linearSpeed + self.angularSpeed

            #print("target_theta:%.1f"%math.degrees(self.target_theta) + " theta:%.1f"%math.degrees(self.theta) + " ang:%.1f"%self.angularSpeed + " target_rho:%.1f"%self.target_rho + " lin:%.1f"%self.linearSpeed)
            #robot.left_vel, robot.right_vel = self.getWheelSpeeds(self.target_x, self.target_y, target_theta, 4)
        #self.send_debug([self.target_x,self.target_y, self.target_theta])
            

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

    def toPositiveAngle(self, angle):
        return math.fmod(angle + 2*math.pi, 2 * math.pi)
    
    def smallestAngleDiff(self, target, source):
        a = self.toPositiveAngle(target) - self.toPositiveAngle(source)
    
        if (a > math.pi):
            a = a - 2 * math.pi
        elif (a < -math.pi):
            a = a + 2 * math.pi
        
        return a
            
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

        self.p = subprocess.Popen([path_simulator, self.parameters, '-d', '-r', str(command_rate), '-p', str(self.port)])
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

    def potentialFunc(self, t_yellow, t_blue, u_B2OG, u_B2G, u_R2B, u_Col, state_terminal):
        if (state_terminal == False):
            #gaussian used
            #https://academo.org/demos/3d-surface-plotter/?expression=1*exp(-(((x-160)%5E2%2B(y-65)%5E2)%2F2*((0.01)%5E2)))&xRange=-0%2C165&yRange=0%2C130&resolution=39
            #potential to adversary goal
            p_B2G = 1*np.exp(-(((self.ball_x-160)**2+(self.ball_y-65)**2)/2*((0.025)**2))) #40cm radius
            #potential to own goal
            p_B2OG = -1*np.exp(-(((self.ball_x)**2+(self.ball_y-65)**2)/2*((0.025)**2))) #40cm radius
            #potential controled robot to ball
            p_R2B = 1*np.exp(-(((self.x-self.ball_x)**2+(self.y-self.ball_y)**2)/2*((0.0125)**2))) #80cm radius
            #potential collision
            team_col, adv_col, wall_col = self.check_collision(t_yellow, t_blue)
            p_Col = -max(wall_col,team_col,adv_col)

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

            if (idx==0):
                if self.target_x == None:
                    self.target_x = t1_robot.pose.x
                    self.target_y = t1_robot.pose.y
                    
                t1_state += (self.normX(self.target_x), self.normX(self.target_y), self.normX(t1_robot.pose.x), self.normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         self.normVx(t1_robot.v_pose.x), self.normVx(t1_robot.v_pose.y), self.normVt(t1_robot.v_pose.yaw))

                self.x = t1_robot.pose.x
                self.y = t1_robot.pose.y
                self.theta = t1_robot.pose.yaw
                self.linearSpeed = math.sqrt(t1_robot.v_pose.x*t1_robot.v_pose.x + t1_robot.v_pose.y*t1_robot.v_pose.y)
                if (abs(math.atan2(t1_robot.v_pose.y,t1_robot.v_pose.x)-self.theta)>math.pi/2):
                    self.linearSpeed = - self.linearSpeed
                
                self.angularSpeed = t1_robot.v_pose.yaw*8/1.63 # VangWheel = vang*RobotWidth/WheelRadius
            else:
                t1_state += (self.normX(t1_robot.pose.x), self.normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         self.normVx(t1_robot.v_pose.x), self.normVx(t1_robot.v_pose.y), self.normVt(t1_robot.v_pose.yaw))

            #estimated values
            #estimated_t1_state += (t1_robot.k_pose.x, t1_robot.k_pose.y, t1_robot.k_pose.yaw,t1_robot.k_v_pose.x, t1_robot.k_v_pose.y, t1_robot.k_v_pose.yaw)

        t2_state = ()
        #estimated_t2_state = ()
        for idx, t2_robot in enumerate(state.robots_blue):
            #real values
            t2_state += (self.normX(t2_robot.pose.x), self.normX(t2_robot.pose.y), math.sin(t2_robot.pose.yaw), math.cos(t2_robot.pose.yaw),
                         self.normVx(t2_robot.v_pose.x), self.normVx(t2_robot.v_pose.y), self.normVt(t2_robot.v_pose.yaw))            #estimated values
            #estimated_t2_state += (t2_robot.k_pose.x, t2_robot.k_pose.y, t2_robot.k_pose.yaw, t2_robot.k_v_pose.x, t2_robot.k_v_pose.y, t2_robot.k_v_pose.yaw)

        #***************************** get reward *****************************
        done = False
        reward = 0;
        if self.is_team_yellow:
            advantage = state.goals_yellow - state.goals_blue
        else:
            advantage = state.goals_blue - state.goals_yellow
        
        if (advantage != self.prev_advantage):
            #terminal state
            reward = (advantage - self.prev_advantage) + self.potentialFunc(state.robots_yellow, state.robots_blue, 
                                                                            self.u_B2OG, self.u_B2G, self.u_R2B, 
                                                                            self.u_Col, True)
            done = True
        else:
            reward = (advantage - self.prev_advantage) + self.potentialFunc(state.robots_yellow, state.robots_blue, 
                                                                            self.u_B2OG, self.u_B2G, self.u_R2B, 
                                                                            self.u_Col, False)

        #pack state:
        env_state = ball_state + t1_state + t2_state

        self.prev_advantage = advantage
        self.rewardSum = self.rewardSum + reward
        self.steps = self.steps + 1
        
        return np.array(env_state), reward, done
