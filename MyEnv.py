import gym, ray																	
from ray.rllib.agents import ppo
import random
from ray import tune
from ray.tune.registry import register_env
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
from MotionMatching import MotionMatching


class MyEnv(gym.Env):
    def __init__(self, env_config = None):

        observation_dimension = 2
        self.observation_space = gym.spaces.Box(low = -1 * np.inf, high = np.inf, shape = (observation_dimension,), dtype = np.float64)

        action_dimension = 1
        self.action_space = gym.spaces.Box(low = -1 * np.pi, high = np.pi, shape = (action_dimension,), dtype = np.float32)

        self.motion_matching_system = MotionMatching()
        
        self.global_goal_position = np.array([0, 0])
        self.set_global_goal_position()
        self.set_state()

    def set_global_goal_position(self):

        goal_xpos = random.randrange(-15*40, 15*40)
        goal_zpos = random.randrange(-15*40, 15*40)

        global_goal_position = np.array([goal_xpos, 0, goal_zpos, 1])
        self.global_goal_position = global_goal_position[:3]

    def set_state(self):
        character_local_frame = MotionMatching.character.getCharacterLocalFrame()
        global_goal_position = [self.global_goal_position[0], 0, self.global_goal_position, 1]
        local_goal_position = np.inv(character_local_frame) @ global_goal_position

        new_state = local_goal_position[0::2]
        self.state = new_state

    def reset(self):
        self.motion_matching_system.reset_motion_matching()
        self.set_goal_position()
        self.set_state()
        

    def step(self, action):
        local_directions = self.calculate_local_directions_from_action(action)
        local_positions = self.calculate_local_positions_from_actoion(action)

        self.motion_matching_system.change_curFrame()
        self.motion_matching_system = self.motion_matching_system.calculate_10st_frame_from_start_frame()

        self.set_state()
        is_reached, goal_distance = self.determine_goal_reached()

        reward = 0
        done = False

        if is_reached:
            reward = 1
            self.reset()
            done = True

        else:
            reward = np.exp(-4 * (goal_distance ** 2))


        return (self.state, reward, done, {"prob":1.0})

    def determine_goal_reached(self, character_local_goal_position):
        
        goal_distance = np.linalg.norm(self.state)

        if goal_distance < 0.5:
            return True, goal_distance

        else:
            return False, goal_distance

    
def main():
    ray.init()
	
    register_env("my_env", lambda config: MyEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "my_env"

    tune.run(ppo.PPOTrainer, 
			config=config,
			stop={"training_iteration": 100},
    		checkpoint_at_end=True,
			)

if __name__ == "__main__":
	main()