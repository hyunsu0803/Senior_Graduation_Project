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
        self.matrix = np.identity(4) # real position, real orienatation
        self.bvh_matrix = np.identity(4)

        self.global_position = np.array([0., 0., 0., 1.], dtype='float32')  # global position
        self.global_velocity = np.array([0., 0., 0., 0.], dtype = 'float32') # global velocity

        self.character_local_position = np.array([0., 0., 0.], dtype='float32')  # local to root joint
        self.character_local_velocity = np.array([0., 0., 0.], dtype='float32')  # local to root joint
        self.character_local_rotation = np.array([1., 0., 0., 0.], dtype='float32')  # local to root joint, quaternion
        # self.character_local_rotvel = np.array([0., 0., 0.], dtype='float32')  # rotational velocity local to root joint

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
        self.global_position = position.copy()

    def set_global_velocity(self, velocity):
        self.global_velocity = velocity

    def set_index(self, index):
        self.index = index
        if self.index == 0:
            self.is_root = True
            print(self.joint_name)

    def set_character_local_position(self, vector):
        self.character_local_position = vector

    def set_character_local_velocity(self, vector):
        self.character_local_velocity = vector

    def set_is_root(self, value):
        self.is_root = value

    def set_is_foot(self, value):
        self.is_foot = value

    def set_character_local_rotation(self, vector):
        self.character_local_rotation = vector

    # def set_character_local_rotvel(self, vector):
    #     self.character_local_rotvel = vector

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
        return self.matrix.copy()

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

    def get_character_local_rotation(self):
        return self.character_local_rotation.copy()


    

    # def getBvhCharacterLocalFrame(self):
    #     M = self.bvh_matrix.copy()

    #     # origin projection 
    #     newOrigin = M[:3, 3]
    #     newOrigin[1] = 0

    #     newDirection = M[:3, 1]
    #     newDirection[1] = 0.

    #     newZaxis=  utils.normalized(newDirection)

    #     newYaxis = np.array([0., 1., 0.])
    #     newXaxis = np.cross(newYaxis, newZaxis)

    #     newTransformationMatrix = np.identity(4)
    #     newTransformationMatrix[:3, 0] = newXaxis
    #     newTransformationMatrix[:3, 1] = newYaxis
    #     newTransformationMatrix[:3, 2] = newZaxis
    #     newTransformationMatrix[:3, 3] = newOrigin

    #     return newTransformationMatrix
