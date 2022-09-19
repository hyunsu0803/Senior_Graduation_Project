import numpy as np
from Character import Character
from scipy.spatial.transform import Rotation as R
import utils
import pickle
import os
from Feature import Feature


class MotionMatching:
    def __init__(self):
        self.character = Character()
        self.curFrame = []
        self.coming_soon_10frames = []
        self.matching_num = -1
        

        TposeX = [0, 1, 0]
        TposeY = [1, 0, 0]
        TposeZ = [0, 0, 1]
        self.real_global_orientation = np.array([TposeX, TposeY, TposeZ]).T
        self.real_global_position = np.array([0., 0., 0.])
        self.bvh_to_real_yrotation = np.identity(3)
        self.bvh_past_orientation = np.array([])
        self.bvh_past_position = np.array([])

        self.target_global_orientation = np.array([1., 0., 0.])
        self.abs_global_velocity = 45
        self.query_vector = Feature()


        self.DB = "tree_dump.bin"
        self.init_query = np.array([ 2.96960973e-01,  9.96564847e+00, -4.50558696e+00,  3.36485732e+01,
                        -7.10655357e+00,  6.16891943e+01,  5.23026731e-02,  9.98631278e-01,
                        5.05232841e-02,  9.98722883e-01, -3.49997476e-03,  9.99993875e-01,
                        5.86174130e+00,  6.39006306e+00, -7.39693220e+00, -1.58474458e+01,
                        1.36000753e+01,  3.77870203e+00,  8.16262709e-02,  9.95738551e-01,
                        -5.71020553e-02,  2.73323919e+01, -1.12751758e+01,  9.64274966e+01,
                        5.65231538e+00,  3.75823975e-01,  1.62885360e+01])
        self.init_query = np.zeros((27,))

        # LaFAN1
        self.mean_array =  np.array([ 6.07492335e-01,  1.83831508e+01,  1.67441441e+00,  3.62060169e+01,
                        3.08432562e+00,  5.19082369e+01,  1.27288395e-02,  9.28758360e-01,
                        2.05411614e-02,  8.14135642e-01,  2.61480465e-02,  7.07203011e-01,
                        1.27510255e+01,  1.75917816e+01, -2.46474355e+00, -1.32840152e+01,
                        1.73819857e+01, -2.66479918e+00,  1.81058656e+00,  5.35151017e-02,
                        5.54863363e+01, -1.45422479e-01,  5.37388807e-02,  5.63968049e+01,
                        8.66350606e-01,  4.33641191e-01,  5.36846555e+01])
        self.std_array = np.array([1.58055459e+01, 3.03224344e+01, 3.20132426e+01, 5.85939596e+01,
                        5.02713793e+01, 8.40305700e+01, 3.29957160e-01, 1.68446306e-01,
                        4.72506881e-01, 3.36895332e-01, 5.38893312e-01, 4.56918131e-01,
                        1.71375122e+01, 2.07184428e+01, 2.37371029e+01, 1.64938960e+01,
                        2.05451891e+01, 2.33365727e+01, 1.08223472e+02, 5.76045354e+01,
                        1.85230903e+02, 1.09276157e+02, 5.89094758e+01, 1.86618171e+02,
                        9.25301673e+01, 5.00009977e+01, 1.47302141e+02])

    def calculate_character_motion(self):
        self.calculate_joint_motion(np.identity(4), self.character.getCharacterRootJoint())


    def calculate_joint_motion(self, parentMatrix, joint, characterMatrix = None):
        parent_local_transformation_matrix = np.identity(4)

        # get current joint's offset from parent joint
        curoffset = joint.get_offset()

        # move transformation matrix's origin using offset data
        temp = np.identity(4)
        if len(joint.get_channel()) != 6:
            temp[:3, 3] = curoffset
        else: 
            joint.bvh_matrix[:3, 3] = np.array(self.curFrame[:3])
        parent_local_transformation_matrix = parent_local_transformation_matrix @ temp

        # channel rotation
        # ROOT
        if len(joint.get_channel()) == 6:
            bvh_current_orientation = R.from_euler('ZYX', self.curFrame[3:6], degrees=True).as_matrix()

            self.calculate_real_global_orientation(bvh_current_orientation)
            self.calculate_real_global_position()

            ROOTPOSITION = np.array(self.real_global_position, dtype='float32')
            # move root's transformation matrix's origin using translation data
            parent_local_transformation_matrix[:3, 3] = ROOTPOSITION
            parent_local_transformation_matrix[:3, :3] = self.real_global_orientation

        # JOINT
        else:
            index = (joint.get_index() + 1) * 3
            parent_local_transformation_matrix[:3, :3] = R.from_euler('ZYX', self.curFrame[index:index+3], degrees=True).as_matrix()
            
        joint.set_transform_matrix(parentMatrix @ parent_local_transformation_matrix)
        transform_matrix = joint.get_transform_matrix()
        global_position = transform_matrix @ np.array([0., 0., 0., 1.])

        if joint.get_is_root():
            characterMatrix = self.character.getCharacterLocalFrame().copy()
            # drawLocalFrame(characterMatrix)
   
        # get root local position and root local velocity
        new_global_position = global_position.copy()
        past_global_position = joint.get_global_position()
        global_velocity = (new_global_position - past_global_position) * 30
        character_local_velocity = (np.linalg.inv(characterMatrix) @ global_velocity)[:3]
        character_local_position = (np.linalg.inv(characterMatrix) @ global_position)[:3]
        
        # set joint class's value
        joint.set_global_position(new_global_position)
        joint.set_character_local_velocity(character_local_velocity)
        joint.set_character_local_position(character_local_position)

        if joint.get_end_site() is None:
            for j in joint.get_child():
                self.calculate_joint_motion(joint.get_transform_matrix(), j, characterMatrix = characterMatrix)

    # only used in class
    def calculate_real_global_orientation(self, bvh_current_orientation):
        # calculate real global orientation
        # A-B about global frame : A @ B.T
        if len(self.bvh_past_orientation) != 0: #Continuous motion playback received via the QnA function
            self.real_global_orientation = self.bvh_to_real_yrotation @ bvh_current_orientation

        else:   # if QnA is newly called
            bvh_current_orientation_direction = bvh_current_orientation[:3, 1].copy()
            bvh_current_orientation_direction[1] = 0
            bvh_current_orientation_direction = utils.normalized(bvh_current_orientation_direction)

            real_global_orientation_direction = self.real_global_orientation[:3, 1].copy()
            real_global_orientation_direction[1] = 0
            real_global_orientation_direction = utils.normalized(real_global_orientation_direction)

            th = np.arccos(np.dot(bvh_current_orientation_direction, real_global_orientation_direction))
            crossing = np.cross(bvh_current_orientation_direction, real_global_orientation_direction)

            if crossing[1] < 0:
                th *= -1
            
            self.bvh_to_real_yrotation = np.array([[np.cos(th), 0, np.sin(th)],
                                            [0, 1, 0],
                                            [-np.sin(th), 0, np.cos(th)]]) # about global frame

            self.real_global_orientation = self.bvh_to_real_yrotation @ bvh_current_orientation

        self.bvh_past_orientation = bvh_current_orientation

    # only used in class
    def calculate_real_global_position(self):
        # calculate real global position
        if len(self.bvh_past_position) != 0: # Continuous motion playback received via the QnA function
            movement_vector = (np.array(self.curFrame[:3]) - np.array(self.bvh_past_position))
            self.real_global_position += self.bvh_to_real_yrotation @ movement_vector    
        else:   # if QnA is newly called
            self.real_global_position[1] = self.curFrame[1]
            
        self.bvh_past_position = self.curFrame[:3]

    def get_matching_10frames(self, key_input = None):
        tree_file = open(self.DB, 'rb')

        DB = pickle.load(tree_file)
        query = np.zeros((27,))

        if key_input == "init":
            query = self.init_query

            # query = np.zeros((27,))
        else:
            query = self.set_query_vector(key_input=key_input)
            

        ans = DB.query(query)
        qidx = ans[1]

        bvh_name, nearest_frame_idx, FPS = self.find_matching_bvh(qidx)
        print()
        print("bvh name", bvh_name, nearest_frame_idx)

        print("!!!!!!!! Query !!!!!!!!", query, sep='\n')
        print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx], sep='\n')

        print("############query feature difference##############")
        print(np.linalg.norm(query - np.array(DB.data[qidx])))
        print("############query feature difference vector##############")
        print(query - np.array(DB.data[qidx]))

        bvh_folder = './lafan1'
        bvh_path = os.path.join(bvh_folder, bvh_name)
        bvh_file = open(bvh_path, "r")

        coming_soon_10frames = bvh_file.readlines()
        future_10frame = coming_soon_10frames[nearest_frame_idx + 20]
        future_20frame = coming_soon_10frames[nearest_frame_idx + 30]
        future_30frame = coming_soon_10frames[nearest_frame_idx + 40]


        coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]

        coming_soon_10frames = [i.split() for i in coming_soon_10frames]
        future_10frame = future_10frame.split()
        future_20frame = future_20frame.split()
        future_30frame = future_30frame.split()

        for i in range(len(coming_soon_10frames)):
            for j in range(len(coming_soon_10frames[i])):
                coming_soon_10frames[i][j] = float(coming_soon_10frames[i][j])

        for i in range(len(future_10frame)):
            future_10frame[i] = float(future_10frame[i])
            future_20frame[i] = float(future_20frame[i])
            future_30frame[i] = float(future_30frame[i])


        return coming_soon_10frames, FPS , [future_10frame, future_20frame, future_30frame]


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

        # future direction setting
        local_future_direction = None
        global_future_direction = None

        if key_input == None:
            global_future_direction = self.target_global_orientation
            local_future_direction = character_local_frame[:3, :3].T @ global_future_direction
        elif key_input == "LEFT":
            self.target_global_orientation = np.array([1., 0., 0.])
            global_future_direction = self.target_global_orientation
            local_future_direction = character_local_frame[:3, :3].T @ global_future_direction
        elif key_input == "RIGHT":
            self.target_global_orientation = np.array([-1., 0., 0.])
            global_future_direction = self.target_global_orientation
            local_future_direction = character_local_frame[:3, :3].T @ global_future_direction
        elif key_input == "UP":
            self.target_global_orientation = np.array([0., 0., 1.])
            global_future_direction = self.target_global_orientation
            local_future_direction = character_local_frame[:3, :3].T @ global_future_direction
        elif key_input == "DOWN":
            self.target_global_orientation = np.array([0., 0., -1.])
            global_future_direction = self.target_global_orientation
            local_future_direction = character_local_frame[:3, :3].T @ global_future_direction

        
        abs_global_velocity = self.abs_global_velocity
        local_3Dposition_future = np.zeros((3, 3))

        for i in range(3):
            local_3Dposition_future[i] = local_future_direction * (abs_global_velocity * (i+1))

        local_2Dposition_future = local_3Dposition_future[:, 0::2]
        
        local_future_direction[1] = 0
        local_future_direction = utils.normalized(local_future_direction)
        local_3Ddirection_future = np.array([local_future_direction, local_future_direction, local_future_direction])
    
        
        local_2Ddirection_future = local_3Ddirection_future[:, 0::2]

        # local_3Dposition_future, local_3Ddirection_future = setRealFutureInfo()
        # local_2Dposition_future = local_3Dposition_future[:, 0::2]
        # local_2Ddirection_future = local_3Ddirection_future[:, 0::2]

        
        # global position setting
        global_3Dposition_future = np.zeros((3, 3))

        for i in range(3):
            global_3Dposition_future[i] = self.real_global_position + global_future_direction * (abs_global_velocity * (i+1))
        global_3Dposition_future[:, 1] = 0
        for i in range(3):
            temp = np.array([0., 0., 0., 1])
            temp[:3] = global_3Dposition_future[i]

        global_3Dposition_future = []
        for i in range(3):
            temp = np.array([0., 0., 0., 1])
            temp[:3] = local_3Dposition_future[i]
            global_3Dposition_future.append((character_local_frame @ temp)[:3])
        
        global_3Dposition_future = np.array(global_3Dposition_future)
        
        # global direction setting
        global_3Ddirection_future = []
        for i in range(3):
            global_3Ddirection_future.append(character_local_frame[:3, :3] @ local_3Ddirection_future[i])

        global_3Ddirection_future = np.array(global_3Ddirection_future)


        self.query_vector.set_global_future_position(global_3Dposition_future)
        self.query_vector.set_global_future_direction(global_3Ddirection_future)

        self.query_vector.set_future_position(np.array(local_2Dposition_future).reshape(6, ))
        self.query_vector.set_future_direction(np.array(local_2Ddirection_future).reshape(6, ))
        # state.query_vector.set_future_position(np.zeros_like(np.array(local_2Dposition_future).reshape(6, )))
        # state.query_vector.set_future_direction(np.zeros_like(np.array(local_2Ddirection_future).reshape(6, )))
        self.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
        self.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
        self.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))

        flatten_query_vector =self.query_vector.get_noramlize_feature_vector(self.mean_array, self.std_array)
            
        return flatten_query_vector
        
    # only used in class
    def find_matching_bvh(self, query):
        info_txt = open('db_index_info.txt', 'r')

        info = info_txt.readlines()
        info = [i.split() for i in info]
        for i in info:
            i[0] = int(i[0])
            i[2] = int(i[2])

        best = info[-1]
        for i in range(len(info) - 1):
            if info[i][0] <= query and info[i+1][0] > query:
                best = info[i]
        
        bvh_name = best[1]
        bvh_line = query - best[0] + best[2]
        FPS = best[-1]

        return bvh_name, bvh_line, int(FPS)

    def draw_future_info(self):
        self.query_vector.draw_future_info(self.character.resize)

    def calculate_and_draw_character(self):
        self.calculate_character_motion()
        self.character.drawCharacter()

    def is_character_drawable(self):
        if len(self.curFrame) > 0:
            return True
        else:
            return False

    def change_curFrame(self, key_input = None):

        FPS = 0

        if self.curFrame == []:
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames(key_input="init")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
			# self.timer.setInterval(1000 / self.FPS * 2)

        elif key_input == "UP":
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames(key_input="UP")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1
            
        elif key_input == "DOWN":
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames(key_input="DOWN")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1

        elif key_input == "LEFT":
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames(key_input="LEFT")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1

        elif key_input == "RIGHT":
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames(key_input="RIGHT")
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
            self.matching_num = -1
        

        elif self.matching_num % 10 == 9:
            print("~~~~~~~~~~~new query~~~~~~~~~~~~~~")
            self.coming_soon_10frames, FPS, temp = self.get_matching_10frames()
            self.reset_bvh_past_position()
            self.reset_bvh_past_orientation()
			# self.timer.setInterval(1000/self.FPS * 2)

        self.matching_num = (self.matching_num+1) % 10
        self.curFrame = self.coming_soon_10frames[self.matching_num]
        print("curFrame change")

        return FPS

    def reset_bvh_past_position(self):
        self.bvh_past_position = np.array([])

    def reset_bvh_past_orientation(self):
        self.bvh_past_orientation = np.array([])



        