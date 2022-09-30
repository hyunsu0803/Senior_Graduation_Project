from MotionMatching import MotionMatching
import os
import pickle
import numpy as np
import ray
from MyEnv import MyEnv
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from OpenGL.GL import *
from OpenGL.GLU import *

class MotionMatching_task(MotionMatching):

    def __init__(self):
        super().__init__()
        ray.init()
        config = ppo.DEFAULT_CONFIG.copy()
        config["env"] = "my_env"
        config['horizon'] = 300
        config['num_workers'] = 0

        register_env("my_env", lambda config: MyEnv(config))
        self.agent = ppo.PPOTrainer(config=config, env = "my_env")
        self.agent.restore("/Users/jangbogyeong/Desktop/PPO_2022-09-26_07-42-04/PPO_my_env_b6cf3_00000_0_2022-09-26_07-42-04/checkpoint_000150")
        
        self.env = MyEnv()


    def get_matching_10frames(self, target_direction = None):
        tree_file = open(self.DB, 'rb')

        DB = pickle.load(tree_file)
        query = np.zeros((27,))

        if self.curFrame == []:
            query = self.init_query

            # query = np.zeros((27,))
        else:
            query = self.set_query_vector(target_direction)
            

        ans = DB.query(query)
        qidx = ans[1]

        bvh_name, nearest_frame_idx= self.find_matching_bvh(qidx)

        # bvh_folder = './lafan1'
        bvh_folder = "/Users/jangbogyeong/my-awesome-project/lafan1"
        bvh_path = os.path.join(bvh_folder, bvh_name)
        bvh_file = open(bvh_path, "r")

        coming_soon_10frames = bvh_file.readlines()
        coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]
        coming_soon_10frames = [i.split() for i in coming_soon_10frames]

        for i in range(len(coming_soon_10frames)):
            for j in range(len(coming_soon_10frames[i])):
                coming_soon_10frames[i][j] = float(coming_soon_10frames[i][j])


        return coming_soon_10frames
    
    def set_query_vector(self, target_direction= None):
        two_foot_position = []
        two_foot_velocity = []
        hip_velocity = []

        character = self.character
        character_local_frame = character.getCharacterLocalFrame()

        character_root_joint = character.getCharacterRootJoint()
        character_foot_joint_list = character.getCharacterTwoFootJoint()

        hip_velocity.append(character_root_joint.get_character_local_velocity())
        for i in range(0, 2):
            two_foot_position.append(character_foot_joint_list[i].get_character_local_position())
            two_foot_velocity.append(character_foot_joint_list[i].get_character_local_velocity())

        self.target_global_direction = self.character.getCharacterLocalFrame()[:3, :3] @ target_direction
        # global_future_direction = self.target_global_direction
        local_future_direction = target_direction
        local_2Dposition_future = self.calculate_local_2Dposition_future(local_future_direction)
        local_2Ddirection_future = self.calculate_local_2Ddirection_future(local_future_direction)
        global_3Dposition_future = self.calculate_global_3Dposition_future(local_2Dposition_future)
        global_3Ddirection_future = self.calculate_global_3Ddirection_future(local_2Ddirection_future)
    
        self.query_vector.set_global_future_position(global_3Dposition_future)
        self.query_vector.set_global_future_direction(global_3Ddirection_future)
        self.query_vector.set_future_position(np.array(local_2Dposition_future).reshape(6, ))
        self.query_vector.set_future_direction(np.array(local_2Ddirection_future).reshape(6, ))
        self.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
        self.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
        self.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))

        flatten_query_vector =self.query_vector.get_noramlize_feature_vector(self.mean_array, self.std_array)
        return flatten_query_vector

    def change_curFrame(self):
        if self.curFrame == []:
            self.coming_soon_10frames = self.get_matching_10frames()
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = (self.matching_num+1) % 10
            self.curFrame = self.coming_soon_10frames[self.matching_num]
        elif self.matching_num % 10 == 9:
            self.make_equal_two_motion_matching_system(self.env.motion_matching_system)
            target_angle = self.agent.compute_single_action(self.env.state)
            self.env.step(target_angle)
            target_direction = self.env.calculate_future_direction(target_angle)
            self.coming_soon_10frames = self.get_matching_10frames(target_direction)
            self.matching_num = (self.matching_num+1) % 10
            self.curFrame = self.coming_soon_10frames[self.matching_num]
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
        else: 
            self.matching_num = (self.matching_num+1) % 10
            self.curFrame = self.coming_soon_10frames[self.matching_num]
            

    def draw_goal_position(self):
        goal_position_2D = self.env.global_goal_position
        goal_position_3D = np.array([goal_position_2D[0]/self.character.resize, 0, goal_position_2D[1]/self.character.resize])
        glColor3ub(255, 255, 0)
        glBegin(GL_LINES)
        glVertex3fv(goal_position_3D)
        glVertex3fv(goal_position_3D + np.array([0., 5., 0]))
        glEnd()
        
        
