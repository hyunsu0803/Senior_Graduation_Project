# ==This class is for defining a feature vector==
import numpy as np
from OpenGL.GL import *


class Feature:
    def __init__(self):
        self.future_position = None  # 6 dim, 60Hz에서 미래의 20, 40, 60 프레임
        self.future_direction = None  # 6 dim, 60Hz에서 미래의 20, 40, 60 프레임
        self.foot_position = None  # 6 dim
        self.foot_velocity = None  # 6 dim
        self.hip_velocity = None  # 3 dim
        # for draw arrow and point
        self.global_future_position = np.zeros(9)  # 9 dim 
        self.global_future_direction = np.zeros(9) # 9 dim


    def set_future_position(self, value):
        self.future_position = value

    def set_future_direction(self, value):
        self.future_direction = value

    def set_foot_position(self, value):
        self.foot_position = value

    def set_foot_velocity(self, value):
        self.foot_velocity = value

    def set_hip_velocity(self, value):
        self.hip_velocity = value

    def set_global_future_position(self, value):
        self.global_future_position = value

    def set_global_future_direction(self, value):
        self.global_future_direction = value

    def get_future_position(self):
        return self.future_position

    def get_future_direction(self):
        return self.future_direction

    def get_foot_position(self):
        return self.foot_position

    def get_foot_velocity(self):
        return self.foot_velocity

    def get_hip_velocity(self):
        return self.hip_velocity

    def get_global_future_position(self):
        return self.global_future_position

    def get_global_future_direction(self):
        return self.global_future_direction

    def get_feature_list(self):
        data1 = np.concatenate( (self.future_position, self.future_direction) )
        data2 = np.concatenate( (self.foot_position, self.foot_velocity) ) 
        data3 = np.concatenate( (data1, data2) )
        data = np.concatenate( (data3, self.hip_velocity) )
    
        return data

    
    def get_noramlize_feature_vector(self, mean_array, std_array):
        
        feature_vector = self.get_feature_list().copy()
        for i in range(0, 27):
            feature_vector[i] = (feature_vector[i] - mean_array[i]) / std_array[i]


        return feature_vector

    def draw_future_info(self, resize):
        # global 3d info
        future_position = self.get_global_future_position().reshape(3, 3)
        future_direction = self.get_global_future_direction().reshape(3, 3)

        future_position[:, 1] = 0.
        future_direction[:, 1] = 0.

        glPointSize(20.)
        glBegin(GL_POINTS)
        glVertex3fv(future_position[0]/resize)
        glVertex3fv(future_position[1]/resize)
        glVertex3fv(future_position[2]/resize)
        glEnd()

        glLineWidth(5.)
        glBegin(GL_LINES)
        glVertex3fv(future_position[0]/resize)
        glVertex3fv(future_position[0]/resize+future_direction[0])
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv(future_position[1]/resize)
        glVertex3fv(future_position[1]/resize+future_direction[1])
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv(future_position[2]/resize)
        glVertex3fv(future_position[2]/resize+future_direction[2])
        glEnd()


