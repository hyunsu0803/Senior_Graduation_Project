import numpy as np

class Joint:
    resize = 1

    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.index = 0  
        self.channel = []
        self.offset = np.array([0., 0., 0.], dtype='float32')
        self.end_site = None  
        self.is_root = False 
        self.is_foot = False 
        self.child_joint = []
        self.transformation_matrix = np.identity(4)
        self.past_parent_local_matrix = np.identity(4) 

        self.global_position = np.array([0., 0., 0., 1.], dtype='float32') 
        self.global_velocity = np.array([0., 0., 0., 0.], dtype = 'float32')
        self.character_local_position = np.array([0., 0., 0.], dtype='float32')  
        self.character_local_velocity = np.array([0., 0., 0.], dtype='float32') 
 
    def set_channel(self, channel):
        self.channel = channel

    def set_offset(self, offset):
        self.offset = offset

    def set_end_site(self, end_site):
        self.end_site = end_site

    def append_child_joint(self, joint):
        self.child_joint.append(joint)

    def set_transform_matrix(self, matrix):
        self.transformation_matrix = matrix

    def set_global_position(self, position):
        self.global_position = position.copy()

    def set_global_velocity(self, velocity):
        self.global_velocity = velocity

    def set_index(self, index):
        self.index = index
        if self.index == 0:
            self.is_root = True

    def set_character_local_position(self, vector):
        self.character_local_position = vector

    def set_character_local_velocity(self, vector):
        self.character_local_velocity = vector

    def set_is_root(self, value):
        self.is_root = value

    def set_is_foot(self, value):
        self.is_foot = value

    def get_joint_name(self):
        return self.joint_name.copy()

    def get_channel(self):
        return self.channel.copy()

    def get_offset(self):
        return self.offset.copy()

    def get_index(self):
        return self.index

    def get_end_site(self):
        return self.end_site

    def get_child(self):
        return self.child_joint.copy()

    def get_transform_matrix(self):
        return self.transformation_matrix.copy()

    def get_global_position(self):
        return self.global_position.copy()

    def get_global_velocity(self):
        return self.global_velocity.copy()

    def get_character_local_position(self):
        return self.character_local_position.copy()

    def get_character_local_velocity(self):
        return self.character_local_velocity.copy()

    def get_is_root(self):
        return self.is_root

    def get_is_foot(self):
        return self.is_foot



