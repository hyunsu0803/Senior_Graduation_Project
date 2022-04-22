import numpy as np


class Joint:
    resize = 1

    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.index = 0  # if this value is 0, this joint is root joint
        self.channel = []
        self.offset = np.array([0., 0., 0.], dtype='float32')
        self.end_site = None  # Indicates whether it's end site
        self.is_root = False  # Indictes whether it's root joint
        self.is_foot = False  # Indicates whether it's foot joint
        self.child_joint = []
        self.matrix = np.identity(4)

        self.global_position = np.array([0., 0., 0.], dtype='float32')  # global position

        self.root_local_position = np.array([0., 0., 0.], dtype='float32')  # local to root joint
        self.root_local_velocity = np.array([0., 0., 0.], dtype='float32')  # local to root joint
        self.root_local_rotation = np.array([1., 0., 0., 0.], dtype='float32')  # local to root joint, quaternion
        self.root_local_rotvel = np.array([0., 0., 0.], dtype='float32')  # rotational velocity local to root joint

    def set_channel(self, channel):
        self.channel = channel

    def set_offset(self, offset):
        self.offset = offset

    def set_end_site(self, end_site):
        self.end_site = end_site

    def append_child_joint(self, joint):
        self.child_joint.append(joint)

    def set_transform_matrix(self, matrix):
        self.matrix = matrix

    def set_global_position(self, position):
        self.global_position = position

    def set_index(self, index):
        self.index = index
        if self.index == 0:
            self.is_root = True
            print(self.joint_name)

    def set_root_local_position(self, vector):
        self.root_local_position = vector

    def set_root_local_velocity(self, vector):
        self.root_local_velocity = vector

    def set_is_root(self, value):
        self.is_root = value

    def set_is_foot(self, value):
        self.is_foot = value

    def set_root_local_rotation(self, vector):
        self.root_local_rotation = vector

    def set_root_local_rotvel(self, vector):
        self.root_local_rotvel = vector

    def get_joint_name(self):
        return self.joint_name

    def get_channel(self):
        return self.channel

    def get_offset(self):
        return self.offset

    def get_index(self):
        return self.index

    def get_end_site(self):
        return self.end_site

    def get_child(self):
        return self.child_joint

    def get_transform_matrix(self):
        return self.matrix

    def get_global_position(self):
        return self.global_position

    def get_root_local_position(self):
        return self.root_local_position

    def get_root_local_velocity(self):
        return self.root_local_velocity

    def get_is_root(self):
        return self.is_root

    def get_is_foot(self):
        return self.is_foot

    def get_root_local_rotation(self):
        return self.root_local_rotation

    def get_root_local_rotvel(self):
        return self.root_local_rotvel
