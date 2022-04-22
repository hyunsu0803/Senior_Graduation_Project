# ==This class is for defining a feature vector==
import numpy as np


class Feature:

    def __init__(self):
        self.future_position = None  # 6 dim, 60Hz에서 미래의 20, 40, 60 프레임
        self.future_direction = None  # 6 dim, 60Hz에서 미래의 20, 40, 60 프레임
        self.foot_position = None  # 6 dim
        self.foot_velocity = None  # 6 dim
        self.hip_velocity = None  # 3 dim

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

    # def get_feature_string(self):
    #     data = ""

    #     for i in range(0, 6):
    #         data += str(self.future_position[i]) + " "

    #     for i in range(0, 6):
    #         data += str(self.future_direction[i]) + " "

    #     for i in range(0, 6):
    #         data += str(self.foot_position[i]) + " "

    #     for i in range(0, 6):
    #         data += str(self.foot_velocity[i]) + " "

    #     for i in range(0, 3):
    #         data += str(self.hip_velocity[i]) + " "

    #     return data

    def get_feature_list(self):
        data1 = np.concatenate( (self.future_position, self.future_direction) )
        data2 = np.concatenate( (self.foot_position, self.foot_velocity) ) 
        data3 = np.concatenate( (data1, data2) )
        data = np.concatenate( (data3, self.hip_velocity) )
    
        return data
