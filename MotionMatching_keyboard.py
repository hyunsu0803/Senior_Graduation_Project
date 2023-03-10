from MotionMatching import MotionMatching
import os
import pickle
import numpy as np

class MotionMatching_keyboard(MotionMatching):

    def __init__(self):
        super().__init__()

    def get_matching_10frames(self, key_input = None):
        tree_file = open(self.DB, 'rb')

        DB = pickle.load(tree_file)
        query = np.zeros((27,))

        if self.curFrame == []:
            query = self.init_query

            # query = np.zeros((27,))
        else:
            query = self.set_query_vector(key_input = key_input)
            

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

    def change_curFrame(self, key_input = None):
        if self.curFrame == []:
            self.coming_soon_10frames = self.get_matching_10frames(key_input="init")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()

        elif key_input == "UP":
            self.coming_soon_10frames = self.get_matching_10frames(key_input="UP")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1
            
        elif key_input == "DOWN":
            self.coming_soon_10frames = self.get_matching_10frames(key_input="DOWN")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1

        elif key_input == "LEFT":
            self.coming_soon_10frames = self.get_matching_10frames(key_input="LEFT")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1

        elif key_input == "RIGHT":
            self.coming_soon_10frames = self.get_matching_10frames(key_input="RIGHT")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1

        elif self.matching_num % 10 == 9:
            self.coming_soon_10frames = self.get_matching_10frames()
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()

        self.matching_num = (self.matching_num+1) % 10
        self.curFrame = self.coming_soon_10frames[self.matching_num]

    def set_query_vector(self, key_input = None):

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

        
        self.set_target_global_direction(key_input)
        global_future_direction = self.target_global_direction
        local_future_direction = character_local_frame[:3, :3].T @ global_future_direction
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

    def set_target_global_direction(self, key_input):
        if key_input == "LEFT":
            self.target_global_direction = np.array([1., 0., 0.])
        elif key_input == "RIGHT":
            self.target_global_direction = np.array([-1., 0., 0.])
        elif key_input == "UP":
            self.target_global_direction = np.array([0., 0., 1.])
        elif key_input == "DOWN":
            self.target_global_direction = np.array([0., 0., -1.])

    

