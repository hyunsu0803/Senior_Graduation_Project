import numpy as np


class Joint:
    resize = 1

    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.index = 0
        self.channel = []
        self.offset = np.array([0., 0., 0.], dtype='float32')
        self.end_site = None
        self.child_joint = []
        self.matrix = np.identity(4)
        self.position = np.array([0., 0., 0., 1.], dtype='float32')

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

    def set_position(self, position):
        self.position = position

    def set_index(self, index):
        self.index = index

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
