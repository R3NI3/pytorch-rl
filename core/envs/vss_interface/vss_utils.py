import numpy as np
import math

# Constants of the World
KRHO = 1.75
RHO_INC = 20 #increase rho a little bit, so it can catch up with the ball
KALPHA = 4
KBETA = -0.4
HALF_AXIS = 8
WHEEL_RADIUS = 2
BALL_APPROACH = -20
decAlpha = 0.3
decLin = 0.9
decAng = 0.6

goal_y_global = 65
goal_x_global = 185

goal_y = 65
goal_x = 185

def clip(val, vmin, vmax):
    return min(max(val, vmin), vmax)

def normX(x):
    return x/170.0

def normVx(vx):
    return vx/80.0

def normVt(vt):
    return vt/12

def toPositiveAngle(angle):
    return math.fmod(angle + 2*math.pi, 2 * math.pi)

def to180range(angle):
    angle = math.fmod(angle, 2 * math.pi)
    if (angle < -math.pi):
        angle = angle + 2 * math.pi
    elif (angle > math.pi):
        angle = angle - 2 * math.pi
    
    return angle
    
def smallestAngleDiff(target, source):
    a = toPositiveAngle(target) - toPositiveAngle(source)

    if (a > math.pi):
        a = a - 2 * math.pi
    elif (a < -math.pi):
        a = a + 2 * math.pi
    
    return a

def getWheelSpeeds(robot, target_x, target_y, target_theta, KRHO=1, rho_inc = 0):
    rho  = np.sqrt((target_x-robot["x"])*(target_x-robot["x"]) + (target_y-robot["y"])*(target_y-robot["y"]))
                
    lambda_ = math.atan2((target_y-robot["y"]),(target_x-robot["x"]))
    alpha = to180range(lambda_ - robot["theta"])
    beta = to180range(-target_theta - alpha)

    reverse = False
    if (abs(alpha)>math.pi/2):
        robot["theta"] = to180range(robot["theta"]+math.pi)
        alpha = to180range(lambda_ - robot["theta"])
        reverse = True
        #print("Reverse")

    linearSpeed = KRHO*(rho+rho_inc)
    angularSpeed = KALPHA*alpha + KBETA*beta
    
    if reverse:
        linearSpeed = -linearSpeed
        #self.angularSpeed = -self.angularSpeed
        
    #print (math.degrees(self.theta), math.degrees(target_theta), math.degrees(lambda_), math.degrees(alpha), math.degrees(beta))

    leftSpeed  = (linearSpeed + angularSpeed*HALF_AXIS)/WHEEL_RADIUS
    rightSpeed = (linearSpeed - angularSpeed*HALF_AXIS)/WHEEL_RADIUS
    
    return leftSpeed, rightSpeed

def getTransformMatrix_g2l(theta, pos_x, pos_y):#global 2 local transform
    R_l2g = np.array([[math.cos(theta),math.sin(theta)],
                     [-math.sin(theta),math.cos(theta)]]) #2x2
    R_g2l = np.transpose(R_l2g)

    D = -np.array([pos_x,pos_y]) #2x1
    D = np.matmul(R_g2l,D) #2x2 * 2x1 -> 2x1

    H_g2l = np.column_stack((R_g2l, D)) #2x3
    H_g2l = np.row_stack((H_g2l, np.array([0,0,1]))) #3x3

    return H_g2l

def check_collision(robot, ball, t_yellow, t_blue, is_team_yellow):
    robot_id = 0
    COL_DIST=12

    robot_x = robot["x"]
    robot_z = robot["y"]

    ball_x = ball["x"]
    ball_z = ball["y"]

    same_team_col = False
    adv_team_col = False
    wall_col = False
    ball_col = False
    
    #for every robot in robots team
    for idx, t1_robot in enumerate(t_yellow):
        tmp_robot_x = t1_robot.pose.x
        tmp_robot_z = t1_robot.pose.y
        if (np.linalg.norm([tmp_robot_x-robot_x,tmp_robot_z-robot_z]) < COL_DIST):
            if(is_team_yellow):
                if (idx !=robot_id):
                    same_team_col = True
            else:
                adv_team_col = True

    #for every robot in adversary team
    for idx, t2_robot in enumerate(t_blue):
        tmp_robot_x = t2_robot.pose.x
        tmp_robot_z = t2_robot.pose.y
        if (np.linalg.norm([tmp_robot_x-robot_x,tmp_robot_z-robot_z]) < COL_DIST):
            if(not is_team_yellow):
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

    #def ball cossilio when:
    #ball inside 7cm from robots center
    if (np.linalg.norm(np.array((ball_x,ball_z))-np.array((robot_x,robot_z))) < 7):
        ball_col = True

    return same_team_col, adv_team_col, wall_col, ball_col

def get_raw_obs(state):
    #get balls values
    balls = []
    ball_state = ()
    for idx, ball in enumerate(state.balls):
        ball_idx = {}
        ball_state += (normX(ball.pose.x), normX(ball.pose.y),
                       normVx(ball.v_pose.x), normVx(ball.v_pose.y)) #check if this doesnt lead to errors
        ball_idx["x"] = ball.pose.x
        ball_idx["y"] = ball.pose.y
        balls.append(ball_idx)

    t1_state = ()
    my_agent = {}    
    for idx, t1_robot in enumerate(state.robots_yellow):
        if (idx==0):     
            t1_state += (normX(t1_robot.pose.x), normX(t1_robot.pose.y), 
                         math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         normVx(t1_robot.v_pose.x), normVx(t1_robot.v_pose.y), normVt(t1_robot.v_pose.yaw))

            my_agent["x"] = t1_robot.pose.x
            my_agent["y"] = t1_robot.pose.y
            my_agent["theta"] = t1_robot.pose.yaw
            
        else:
            t1_state += (normX(t1_robot.pose.x), normX(t1_robot.pose.y),
                         math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                          normVx(t1_robot.v_pose.x), normVx(t1_robot.v_pose.y), normVt(t1_robot.v_pose.yaw))

    t2_state = ()
    for idx, t2_robot in enumerate(state.robots_blue):
        t2_state += (normX(t2_robot.pose.x), normX(t2_robot.pose.y), 
                     math.sin(t2_robot.pose.yaw), math.cos(t2_robot.pose.yaw),
                     normVx(t2_robot.v_pose.x), normVx(t2_robot.v_pose.y), normVt(t2_robot.v_pose.yaw))            #estimated values
        
    env_state = ball_state + t1_state + t2_state
    return np.array(env_state), balls, my_agent

def get_sc_obs(state):
    my_agent = {}
    t1_state = ()    
    for idx, t1_robot in enumerate(state.robots_yellow):
        if (idx==0):     
            my_agent["x"] = t1_robot.pose.x
            my_agent["y"] = t1_robot.pose.y
            my_agent["theta"] = t1_robot.pose.yaw
            theta_g2l = t1_robot.pose.yaw
            H_matrix = getTransformMatrix_g2l(t1_robot.pose.yaw, t1_robot.pose.x, t1_robot.pose.y) #3x3
            vel_global = np.transpose(np.array([t1_robot.v_pose.x, t1_robot.v_pose.y, 1])) #3x1

            vel_local = np.matmul(H_matrix, vel_global) # 3x1
            t1_state += (0, 0, #x , y 
                         0, 1,  #sin(theta=0), cos(theta=0)
                         normVx(vel_local[0]), normVx(vel_local[1]), normVt(t1_robot.v_pose.yaw))
        else:

            vel_global = np.transpose(np.array([t1_robot.v_pose.x, t1_robot.v_pose.y, 1])) #3x1
            pos_global = np.transpose(np.array([t1_robot.pose.x, t1_robot.pose.y, 1])) #3x1
            
            theta_local = t1_robot.pose.yaw - theta_g2l
            pos_local = np.matmul(H_matrix, pos_global) # 3x1
            vel_local = np.matmul(H_matrix, vel_global) # 3x1

            t1_state += (normX(pos_local[0]), normX(pos_local[1]),
                         math.sin(theta_local), math.cos(theta_local),
                         normVx(vel_local[0]), normVx(vel_local[1]), normVt(t1_robot.v_pose.yaw))


    #get balls values
    balls = []
    ball_state = ()
    for idx, ball in enumerate(state.balls):
        ball_idx = {}

        pos_global = np.transpose(np.array([ball.pose.x, ball.pose.y, 1])) #3x1
        vel_global = np.transpose(np.array([ball.v_pose.x, ball.v_pose.y, 1])) #3x1

        pos_local = np.matmul(H_matrix, pos_global) # 3x1
        vel_local = np.matmul(H_matrix, vel_global) # 3x1

        ball_idx["x"] = ball.pose.x
        ball_idx["y"] = ball.pose.y

        ball_state += (normX(pos_local[0]),normX(pos_local[1]),
                       normVx(vel_local[0]),normVx(vel_local[1]))

        balls.append(ball_idx)

    t2_state = ()
    for idx, t2_robot in enumerate(state.robots_blue):        
        vel_global = np.transpose(np.array([t2_robot.v_pose.x, t2_robot.v_pose.y, 1])) #3x1
        pos_global = np.transpose(np.array([t2_robot.pose.x, t2_robot.pose.y, 1])) #3x1
        
        theta_local = t2_robot.pose.yaw - theta_g2l
        pos_local = np.matmul(H_matrix, pos_global) # 3x1
        vel_local = np.matmul(H_matrix, vel_global) # 3x1

        t2_state += (normX(pos_local[0]), normX(pos_local[1]),
                     math.sin(theta_local), math.cos(theta_local),
                     normVx(vel_local[0]), normVx(vel_local[1]), normVt(t1_robot.v_pose.yaw))
    
    #adv goal position
    pos_global = np.transpose(np.array([goal_x_global, goal_y_global, 1])) #3x1
    pos_local = np.matmul(H_matrix, pos_global) # 3x1
    #goal_x = pos_local[0]
    #goal_y = pos_local[1]
    adv_goal_state = (pos_local[0],pos_local[1])

    #own goal postition
    pos_global = np.transpose(np.array([5, goal_y_global, 1])) #3x1
    pos_local = np.matmul(H_matrix, pos_global) # 3x1
    #goal_x = pos_local[0]
    #goal_y = pos_local[1]
    own_goal_state = (pos_local[0],pos_local[1])

    env_state = ball_state + t1_state + t2_state + adv_goal_state + own_goal_state
    return np.array(env_state), balls, my_agent

def get_rho_phi_act(global_commands, robot, ball, target, adv_goal):
    if target["theta"] == None:
        target["theta"] = robot["theta"]

    if target["x"] == None:
        target["x"] = robot["x"]
        target["y"] = robot["y"]
        target["theta"] = robot["theta"]

    target_rho = np.linalg.norm(np.array((target["x"], target["y"])) - np.array((robot["x"], robot["y"])))

    if target_rho > 0.01:
        target["theta"] = math.atan2((target["y"] - robot["y"]),(target["x"] - robot["x"]))

    if abs(smallestAngleDiff(target["theta"], robot["theta"]))>math.pi/2:
        target_rho   = -target_rho
        target["theta"] =  to180range(target["theta"]+math.pi)

    if global_commands >= 0: #default command: carry ball to goal
        goal_theta = math.atan2((ball["y"]-goal_y),(ball["x"]-goal_x))
        rho  = np.linalg.norm(np.array((ball["x"], ball["y"])) - np.array((robot["x"], robot["y"])))
        apr = max(BALL_APPROACH,-rho/2)

        target["x"] = ball_appr_x = ball["x"] - apr*math.cos(goal_theta)
        target["y"] = ball_appr_y = ball["y"] - (apr/2)*math.sin(goal_theta)
        target["theta"] = goal_theta

        robot_leftVel, robot_rightVel = getWheelSpeeds({"x":robot["x"],"y":robot["y"],"theta":robot["theta"]},
                                                         ball_appr_x, ball_appr_y, goal_theta, KRHO, RHO_INC)

    else:
        dict_cmd = {1:(-math.pi/12,0),
                2:( math.pi/12,0),
                3:(0, 15),
                4:(0,-15),
                5:(0,0)
                }
        
        target_rho = clip(target_rho + dict_cmd[global_commands][1],-60,60)        
        target["theta"] = to180range(target["theta"] + dict_cmd[global_commands][0])

        if target_rho < 0:
            rbt_theta = to180range(target["theta"] + math.pi)
            cmd_theta = to180range(target["theta"] + math.pi)

        else:
            rbt_theta = robot["theta"]
            cmd_theta = target["theta"]

        angularSpeed = clip(-30*smallestAngleDiff(cmd_theta,rbt_theta),-30, 30)
        linearSpeed = 1.5*target_rho

        target["x"] = clip(robot["x"] + target_rho*math.cos(target["theta"]),0,170)
        target["y"] = clip(robot["y"] + target_rho*math.sin(target["theta"]),0,130)

        robot_leftVel = linearSpeed - angularSpeed
        robot_rightVel  = linearSpeed + angularSpeed

    return robot_leftVel, robot_rightVel

def get_observation_from_state(state, mode = "raw"):
    if (mode == "raw"):
        env_state, balls, my_agent = get_raw_obs(state)

    elif (mode == "self_centered"):
        env_state, balls, my_agent = get_sc_obs(state)

    else:
        env_state, balls, my_agent = None, None, None

    return env_state, balls, my_agent

def get_action_from_command(commands, robot, ball, target, adv_goal, mode = "phi_rho"):
    if(mode == "phi_rho"):
        robot_leftVel, robot_rightVel = get_rho_phi_act(commands, robot, ball, target, adv_goal)

    else:
        robot_leftVel, robot_rightVel = None, None

    return robot_leftVel, robot_rightVel

