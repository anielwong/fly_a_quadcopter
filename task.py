import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.03*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class TakeOff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pos=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        
        # If some arguments aren't implemented
        self.init_pos = init_pos if init_pos is not None else np.array([0., 0., 0., 0., 0., 0.])
        self.init_velocities = init_velocities if init_velocities is not None else np.array([0., 0., 0.])
        self.init_angle_velocities = init_angle_velocities if init_angle_velocities is not None else np.array([0., 0., 0.])
        
        # Simulation
        self.sim = PhysicsSim(init_pos, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 20.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        reward = 0.0
        punish = 0.0
        current_pos = self.sim.pose[:3]
        
        # for a steady takeoff, the angles velocities of the drone are penalized
        punish += (self.sim.angular_v[0] - self.init_angle_velocities[0])**2 + (self.sim.angular_v[1] - self.init_angle_velocities[1])**2 + (self.sim.angular_v[2] - self.init_angle_velocities[2])**2
        
        # penalize for the distance from the target for each axis
        punish += (current_pos[0] - self.target_pos[0])**2 + (current_pos[1] - self.target_pos[1])**2 + 10 * (current_pos[2] - self.target_pos[2])**2
        
        punish += abs(abs(current_pos - self.target_pos).sum() - abs(self.sim.v).sum())
        
        # calculate the distance between the current position to the target position in 3D space
        dist = np.sqrt((current_pos[0] - self.target_pos[0])**2 + (current_pos[1] - self.target_pos[1])**2 + (current_pos[2] - self.target_pos[2])**2)
        
        # extra reward if this calculated distance is near from the current position
        if dist < 10:
            reward += 1000
        
        # reward for flying
        reward += 100
        
        # final total reward
        final_reward = reward - punish * 0.0002
        
        return final_reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

