import gym, ray																	
from ray.rllib.agents import ppo
import random
from ray import tune
from ray.tune.registry import register_env
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
from MotionMatching_training import MotionMatching_training


class MyEnv(gym.Env):
    def __init__(self, env_config = None):

        observation_dimension = 2
        self.observation_space = gym.spaces.Box(low = -1 * np.inf, high = np.inf, shape = (observation_dimension,), dtype = np.float64)

        action_dimension = 1
        self.action_space = gym.spaces.Box(low = -1 * np.pi/2, high = np.pi/2, shape = (action_dimension,), dtype = np.float32)

        self.motion_matching_system = MotionMatching_training()
        
        self.global_goal_position = np.array([0, 0])
        self.draw_init_query()
        self.set_global_goal_position()
        self.set_state()
        self.have_to_reset = False

    def set_global_goal_position(self):

        goal_xpos = random.randrange(-600, 600)
        goal_zpos = random.randrange(-600, 600)


        self.global_goal_position = np.array([goal_xpos, goal_zpos])

    def set_state(self):
        character_local_frame = self.motion_matching_system.getCharacterLocalFrame()
        global_goal_position = np.array([self.global_goal_position[0], 0, self.global_goal_position[1], 1])
        local_goal_position = np.linalg.inv(character_local_frame) @ global_goal_position

        new_state = local_goal_position[0::2]
        self.state = np.array(new_state)


    def reset(self):
        # self.motion_matching_system.reset_motion_matching()
        self.set_state()
        self.have_to_reset = False

        return self.state

    def draw_init_query(self):
        self.motion_matching_system.change_curFrame()
        self.motion_matching_system = self.motion_matching_system.calculate_10st_frame_from_start_frame()
        

    def step(self, action):
        
        if self.have_to_reset == True:
            self.set_global_goal_position()
            self.reset()
            done = True
            reward = 1

        else:
            character_local_future_direction = self.calculate_future_direction(action)
            self.motion_matching_system.change_curFrame(target_direction = character_local_future_direction)
            
            self.motion_matching_system = self.motion_matching_system.calculate_10st_frame_from_start_frame()

            self.set_state()
            is_reached, goal_distance = self.determine_goal_reached()

            reward = 0
            done = False

            if is_reached:
                self.have_to_reset = True
                reward = 1
                # self.motion_matching_system.reset_motion_matching()
                

            else:
                reward = np.exp(-1 * np.sqrt(goal_distance))
            


        return (self.state, reward, done, {"prob":1.0})
    
    def calculate_future_direction(self, action):
        future_local_direction = np.array([np.sin(action[0]), 0, np.cos(action[0])], dtype = np.float64)
        return future_local_direction
    
    def determine_goal_reached(self):
        
        goal_distance = np.linalg.norm(self.state)

        if goal_distance < 30:
            return True, goal_distance

        else:
            return False, goal_distance

    
def main():
    ray.init()
	
    register_env("my_env", lambda config: MyEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "my_env"
    config['horizon'] = 300
    

    tune.run(ppo.PPOTrainer, 
			config=config,
			stop={"training_iteration": 50},
    		checkpoint_at_end=True,
			)

if __name__ == "__main__":
	main()