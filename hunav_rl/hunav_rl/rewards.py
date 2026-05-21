"""Reward functions of DRL-SFM."""


import numpy as np
import random

def goal_reward(goal_reached, t_max, t_min, t, w_goal=10.0, w_time=5.0):
    """Compute goal-reaching reward.
    
    Args:
        goal_reached: Boolean indicating if the goal is reached
        t_max: Maximum time allowed to reach the goal
        t_min: Minimum time threshold for reward scaling
        t: Time taken to reach the goal
        w_goal: Weight for goal-reaching reward
        w_time: Weight for time-bonus

    Returns:
        goal_reward: Computed goal-reaching reward
    """
    if goal_reached:
        goal_reward = w_goal + w_time * ((t_max - t) / (t_max - t_min))
    else:
        goal_reward = 0.0
    return goal_reward

def truncation_reward(truncated, l_t, l_max, w_trun=50.0):
    """Compute truncation reward.

    Args:
        truncated: Boolean indicating if the episode is truncated
        l_t: Remaining length of the planned path
        l_max: Maximum path length

    Returns:
        truncation_reward: Computed truncation reward
    """
    if truncated:
        truncation_reward = -w_trun * (l_t / l_max)
    else:
        truncation_reward = 0.0
    return truncation_reward

def progress_reward(d_prog, w_prog=4.5, w_rev=5.5):
    """Compute progress reward.

    Args:
        d_prog: Progress made towards the waypoint
        w_prog: Weight for positive progress
        w_rev: Weight for negative progress

    Returns:
        progress_reward: Computed progress reward
    """
    if d_prog < 0:
        progress_reward = w_rev * d_prog
    else:
        progress_reward = w_prog * d_prog
    return progress_reward

def heading_reward(alpha_wp, w_heading=0.1, angle_tol=np.pi/9):
    """Compute heading reward.

    Args:
        alpha_wp: Angle to the waypoint
        w_heading: Weight for heading reward
        angle_tol: Tolerated deviation from the optimal heading
            for which the reward remains positive.

    Returns:
        heading_reward: Computed heading reward
    """
    heading_reward = -w_heading * (abs(alpha_wp) - angle_tol) / angle_tol
    return heading_reward

def cost(costmap):
    """Compute cost in the middle of the costmap.
    
    Args:
        costmap: Costmap array

    Returns:
        cost: Cost value at the center of the costmap
    """
    h, w, _ = costmap.shape
    if h % 2 == 1 and w % 2 == 1:
        cost = float(costmap[h//2, w//2, 0])
    else:
        cost = float(np.max([
            costmap[h//2, w//2, 0],
            costmap[h//2-1, w//2, 0],
            costmap[h//2, w//2-1, 0],
            costmap[h//2-1, w//2-1, 0]
        ]))
    return cost

def cost_reward(local_costmap, w_cost=0.01):
    """Compute cost reward based on the local costmap.
    
    Args:
        local_costmap: Local costmap array (with value range [0, 100])
        w_cost: Weight for cost reward

    Returns:
        cost_reward: Computed cost reward
    """
    cost_reward = -(w_cost * cost(local_costmap)) ** 2
    return cost_reward

def sfm_reward(p_robot, p_sfm, dt, w_sfm=0.3, d_sfm=0.3):
    """Compute SFM-reward based on the distance to the SFM-predicted position.
    
    Args:
        p_robot: Robot position [x, y]
        p_sfm: SFM-predicted position [x, y]
        dt: Time step duration
        w_sfm: Weight for SFM reward
        d_sfm: Tolerated deviation for which the reward is positive

    Returns:
        sfm_reward: Computed SFM reward
    """
    distance_to_sfm_prediction = np.linalg.norm(np.array(p_robot) - np.array(p_sfm))/dt
    sfm_reward = -w_sfm * (distance_to_sfm_prediction - d_sfm)
    return sfm_reward

def vo_reward(
    goal, agents, v_x, r_angle=0.6, angle_thresh=np.pi/6, robot_radius=0.25,
):
    """Compute VO-reward based on velocity obstacles.
    This reward function is reconstructed from DRL-VO.
    
    Args:
        goal: Target position [x, y] relative to robot
        agents: People message containing pedestrian information (robot frame)
        v_x: Robot linear velocity 
        r_angle: Reward scaling factor for angle
        angle_thresh: Angle threshold for reward calculation
        robot_radius: Robot radius in meters (default 0.25m)
    
    Returns:
        vo_reward: Computed reward based on velocity obstacles
    """
    theta_pre = np.arctan2(goal[1], goal[0])
    d_theta = theta_pre
    if hasattr(agents, 'people') and len(agents.people) != 0:
        d_theta = np.pi/2
        N = 60
        theta_min = 1000
        for i in range(N):
            theta = random.uniform(-np.pi, np.pi)
            free = True
            for ped in agents.people:
                p_x = ped.position.x
                p_y = ped.position.y
                p_vx = ped.velocity.x
                p_vy = ped.velocity.y
                ped_dis = np.linalg.norm([p_x, p_y])
                if(ped_dis <= 7):
                    ped_theta = np.arctan2(p_y, p_x)
                    vo_theta = np.arctan2(
                        3*robot_radius,
                        np.sqrt(ped_dis**2 - (3*robot_radius)**2)
                    )
                    # collision cone:
                    theta_rp = np.arctan2(
                        v_x * np.sin(theta) - p_vy,
                        v_x * np.cos(theta) - p_vx
                    )
                    if (theta_rp >= (ped_theta - vo_theta)
                            and theta_rp <= (ped_theta + vo_theta)):
                        free = False
                        break
            # reachable available theta:
            if(free):
                theta_diff = (theta - theta_pre)**2
                if(theta_diff < theta_min):
                    theta_min = theta_diff
                    d_theta = theta
    else: # no obstacles:
        d_theta = theta_pre
    vo_reward = r_angle*(angle_thresh - abs(d_theta))
    return vo_reward

