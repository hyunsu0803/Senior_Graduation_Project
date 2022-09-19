# from telnetlib import theNULL
# from threading import local
# import numpy as np
# from OpenGL.GL import *
# from OpenGL.GLU import *
# from scipy.spatial.transform import Rotation as R

# from Joint import Joint
# from Feature import Feature
# import utils
# import MyWindow


# class state:
#     joint_list = []
#     frame_list = []
#     query_vector = Feature()
#     bvh_past_position = np.array([])
#     bvh_past_orientation = np.array([])
#     target_orientation = np.array([1., 0., 0.])
    
#     # The global position of the character on the window
#     real_global_position = np.array([0., 0., 0.])
#     TposeX = [0, 1, 0]
#     TposeY = [1, 0, 0]
#     TposeZ = [0, 0, 1]
#     real_global_orientation = np.array([TposeX, TposeY, TposeZ]).T
#     bvh_to_real_yrotation = np.identity(3)

#     # LaFAN1
#     mean_array =  np.array([ 6.07492335e-01,  1.83831508e+01,  1.67441441e+00,  3.62060169e+01,
#                         3.08432562e+00,  5.19082369e+01,  1.27288395e-02,  9.28758360e-01,
#                         2.05411614e-02,  8.14135642e-01,  2.61480465e-02,  7.07203011e-01,
#                         1.27510255e+01,  1.75917816e+01, -2.46474355e+00, -1.32840152e+01,
#                         1.73819857e+01, -2.66479918e+00,  1.81058656e+00,  5.35151017e-02,
#                         5.54863363e+01, -1.45422479e-01,  5.37388807e-02,  5.63968049e+01,
#                         8.66350606e-01,  4.33641191e-01,  5.36846555e+01])
#     std_array = np.array([1.58055459e+01, 3.03224344e+01, 3.20132426e+01, 5.85939596e+01,
#                         5.02713793e+01, 8.40305700e+01, 3.29957160e-01, 1.68446306e-01,
#                         4.72506881e-01, 3.36895332e-01, 5.38893312e-01, 4.56918131e-01,
#                         1.71375122e+01, 2.07184428e+01, 2.37371029e+01, 1.64938960e+01,
#                         2.05451891e+01, 2.33365727e+01, 1.08223472e+02, 5.76045354e+01,
#                         1.85230903e+02, 1.09276157e+02, 5.89094758e+01, 1.86618171e+02,
#                         9.25301673e+01, 5.00009977e+01, 1.47302141e+02])



# A function that draws a line (or capsule) connecting the parent point and my point
# the rootMatrix enters None when drawing a root joint
# the parentMatrix enters identity(4) when drawing a root joint

# def drawJoint(parentMatrix, joint, characterMatrix=None):


#     newMatrix = np.identity(4)
#     cur_position = [0, 0, 0, 1]

#     # get current joint's offset from parent joint
#     curoffset = joint.get_offset()

#     # move transformation matrix's origin using offset data
#     temp = np.identity(4)
#     if len(joint.get_channel()) != 6:
#         temp[:3, 3] = curoffset
#     else: 
#         joint.bvh_matrix[:3, 3] = np.array(MyWindow.state.curFrame[:3])
#     newMatrix = newMatrix @ temp

#     # channel rotation
#     # ROOT
#     if len(joint.get_channel()) == 6:
#         bvh_current_orientation = np.identity(3)
#         bvh_to_real_rotation = None

#         for i in range(3, 6):
#             if joint.get_channel()[i].upper() == 'XROTATION':
#                 xr = MyWindow.state.curFrame[i]
#                 xr = np.radians(xr)
#                 Rx = np.array([[1., 0., 0.],
#                                [0, np.cos(xr), -np.sin(xr)],
#                                [0, np.sin(xr), np.cos(xr)]])
#                 bvh_current_orientation = bvh_current_orientation @ Rx
#             elif joint.get_channel()[i].upper() == 'YROTATION':
#                 yr = MyWindow.state.curFrame[i]
#                 yr = np.radians(yr)
#                 Ry = np.array([[np.cos(yr), 0, np.sin(yr)],
#                                [0, 1, 0],
#                                [-np.sin(yr), 0, np.cos(yr)]])
#                 bvh_current_orientation = bvh_current_orientation @ Ry
#             elif joint.get_channel()[i].upper() == 'ZROTATION':
#                 zr = MyWindow.state.curFrame[i]
#                 zr = np.radians(zr)
#                 Rz = np.array([[np.cos(zr), -np.sin(zr), 0],
#                                [np.sin(zr), np.cos(zr), 0],
#                                [0, 0, 1]])
#                 bvh_current_orientation = bvh_current_orientation @ Rz
#         joint.bvh_matrix[:3, :3] = bvh_current_orientation

#         # calculate real global orientation
#         # A-B about global frame : A @ B.T
#         if len(state.bvh_past_orientation) != 0: #Continuous motion playback received via the QnA function
#             state.real_global_orientation = state.bvh_to_real_yrotation @ bvh_current_orientation

#         else:   # if QnA is newly called
#             bvh_current_orientation_direction = bvh_current_orientation[:3, 1].copy()
#             bvh_current_orientation_direction[1] = 0
#             bvh_current_orientation_direction = utils.normalized(bvh_current_orientation_direction)

#             real_global_orientation_direction = state.real_global_orientation[:3, 1].copy()
#             real_global_orientation_direction[1] = 0
#             real_global_orientation_direction = utils.normalized(real_global_orientation_direction)

#             th = np.arccos(np.dot(bvh_current_orientation_direction, real_global_orientation_direction))
#             crossing = np.cross(bvh_current_orientation_direction, real_global_orientation_direction)

#             if crossing[1] < 0:
#                 th *= -1
            
#             state.bvh_to_real_yrotation = np.array([[np.cos(th), 0, np.sin(th)],
#                                              [0, 1, 0],
#                                              [-np.sin(th), 0, np.cos(th)]]) # about global frame

#             state.real_global_orientation = state.bvh_to_real_yrotation @ bvh_current_orientation

#         state.bvh_past_orientation = bvh_current_orientation

#         # calculate real global position
#         if len(state.bvh_past_position) != 0: # Continuous motion playback received via the QnA function
#             movement_vector = (np.array(MyWindow.state.curFrame[:3]) - np.array(state.bvh_past_position))
#             state.real_global_position += state.bvh_to_real_yrotation @ movement_vector    
#         else:   # if QnA is newly called
#             state.real_global_position[1] = MyWindow.state.curFrame[1]
            
#         state.bvh_past_position = MyWindow.state.curFrame[:3]

#         ROOTPOSITION = np.array(state.real_global_position, dtype='float32')

#         # move root's transformation matrix's origin using translation data
#         newMatrix[:3, 3] = ROOTPOSITION
#         newMatrix[:3, :3] = state.real_global_orientation

#     # JOINT
#     else:
#         index = joint.get_index()
#         for i in range(3):
#             if joint.get_channel()[i].upper() == 'XROTATION':
#                 xr = MyWindow.state.curFrame[(index + 1) * 3 + i]
#                 xr = np.radians(xr)
#                 Rx = np.array([[1., 0., 0., 0.],
#                                [0, np.cos(xr), -np.sin(xr), 0],
#                                [0, np.sin(xr), np.cos(xr), 0],
#                                [0., 0., 0., 1.]])
#                 newMatrix = newMatrix @ Rx

#             elif joint.get_channel()[i].upper() == 'YROTATION':
#                 yr = MyWindow.state.curFrame[(index + 1) * 3 + i]
#                 yr = np.radians(yr)
#                 Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
#                                [0, 1, 0, 0],
#                                [-np.sin(yr), 0, np.cos(yr), 0],
#                                [0., 0., 0., 1.]])
#                 newMatrix = newMatrix @ Ry

#             elif joint.get_channel()[i].upper() == 'ZROTATION':
#                 zr = MyWindow.state.curFrame[(index + 1) * 3 + i]
#                 zr = np.radians(zr)
#                 Rz = np.array([[np.cos(zr), -np.sin(zr), 0, 0],
#                                [np.sin(zr), np.cos(zr), 0, 0],
#                                [0, 0, 1, 0],
#                                [0, 0, 0, 1]])
#                 newMatrix = newMatrix @ Rz

#     joint.set_transform_matrix(parentMatrix @ newMatrix)
#     transform_matrix = joint.get_transform_matrix()
#     global_position = transform_matrix @ np.array([0., 0., 0., 1.])
    

#     # set parent's global position (if it is root joint, parent_position is current_position)
#     if joint.get_is_root():
#         parent_position = global_position
#     else:
#         parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

#     cur_position = global_position      # global position of this joint

#     # Check if it's Root joint, otherwise update Joint class's data
#     # velocity, rotation velocity update 

#     if joint.get_is_root():
#         characterMatrix = joint.getCharacterLocalFrame().copy()
#         # drawLocalFrame(characterMatrix)

        
#     # get root local position and root local velocity
#     new_global_position = global_position.copy()
#     past_global_position = joint.get_global_position()
#     global_velocity = (new_global_position - past_global_position) * 30
#     character_local_velocity = (np.linalg.inv(characterMatrix) @ global_velocity)[:3]
#     character_local_position = (np.linalg.inv(characterMatrix) @ global_position)[:3]
    

#     # set joint class's value
#     joint.set_global_position(new_global_position)
#     joint.set_character_local_velocity(character_local_velocity)
#     joint.set_character_local_position(character_local_position)

#     v = cur_position/Joint.resize - parent_position/Joint.resize
#     box_length = utils.l2norm(v)
#     v = utils.normalized(v)
#     rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
#     check = np.dot(np.array([0, 1, 0]), v[:3])
#     # under 90
#     if check >= 0:
#         rotate_angle = np.arcsin(utils.l2norm(rotation_vector))
#         rotation_vector = utils.normalized(rotation_vector) * rotate_angle
#     # over 90
#     else:
#         rotate_angle = np.arcsin(utils.l2norm(rotation_vector)) + np.pi
#         rotate_angle *= -1

#         # not 180
#         if utils.l2norm(rotation_vector) != 0:
#             rotation_vector = utils.normalized(rotation_vector) * rotate_angle
#         # if 180, rotate_vector becomes (0, 0, 0)
#         else:
#             rotation_vector = np.array([0., 0., 1.]) * rotate_angle

#     rotation_matrix = np.identity(4)
#     rotation_matrix[:3, :3] = utils.exp(rotation_vector[:3])

#     glPushMatrix()
#     glTranslatef(parent_position[0]/Joint.resize, parent_position[1]/Joint.resize, parent_position[2]/Joint.resize)
#     glMultMatrixf(rotation_matrix.T)

#     glTranslatef(0., box_length / 2, 0.)
#     glScalef(.05, box_length, .05)
#     # drawCapsule
#     quadric = gluNewQuadric()
#     gluSphere(quadric, 0.5, 20, 20)
#     glPopMatrix()

#     # draw end effector
#     if joint.get_end_site() is not None:
#         end_offset = joint.get_end_site()
#         endMatrix = np.identity(4)
#         endMatrix[:3, 3] = end_offset
#         end_position = parentMatrix @ newMatrix @ endMatrix @ np.array([0., 0., 0., 1.])

#         v = end_position/Joint.resize - cur_position/Joint.resize
#         box_length = utils.l2norm(v)
#         v = utils.normalized(v)
#         rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
#         check = np.dot(np.array([0, 1, 0]), v[:3])
#         if check >= 0:
#             rotate_angle = np.arcsin(utils.l2norm(rotation_vector))
#         else:
#             rotate_angle = np.arcsin(utils.l2norm(rotation_vector)) + np.pi
#             rotate_angle *= -1
#         rotation_vector = utils.normalized(rotation_vector) * rotate_angle
#         rotation_matrix = np.identity(4)
#         rotation_matrix[:3, :3] = utils.exp(rotation_vector[:3])

#         glPushMatrix()
#         glTranslatef(cur_position[0]/Joint.resize, cur_position[1]/Joint.resize, cur_position[2]/Joint.resize)
#         glMultMatrixf(rotation_matrix.T)
#         glTranslatef(0., box_length / 2, 0.)
#         glScalef(.05, box_length, .05)
#         # drawCapsule

#         quadric = gluNewQuadric()
#         gluSphere(quadric, 0.5, 20, 20)
#         glPopMatrix()

#     # draw child joints
#     else:
#         for j in joint.get_child():
#             drawJoint(joint.get_transform_matrix(), j, characterMatrix = characterMatrix)



# def set_query_vector(key_input = None):

#     two_foot_position = []
#     two_foot_velocity = []
#     hip_velocity = []

#     for joint in state.joint_list:
#         if joint.get_is_root():
#             hip_velocity.append(joint.get_character_local_velocity())
#             # print("root position", joint.get_character_local_position())

#         elif joint.get_is_foot():
        
#             two_foot_position.append(joint.get_character_local_position())
#             two_foot_velocity.append(joint.get_character_local_velocity())

    

#     # future direction setting
#     local_future_direction = None
#     global_future_direction = None

#     if key_input == None:
#         global_future_direction = state.target_orientation
#         local_future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
#     elif key_input == "LEFT":
#         state.target_orientation = np.array([1., 0., 0.])
#         global_future_direction = state.target_orientation
#         local_future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
#     elif key_input == "RIGHT":
#         state.target_orientation = np.array([-1., 0., 0.])
#         global_future_direction = state.target_orientation
#         local_future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
#     elif key_input == "UP":
#         state.target_orientation = np.array([0., 0., 1.])
#         global_future_direction = state.target_orientation
#         local_future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
#     elif key_input == "DOWN":
#         state.target_orientation = np.array([0., 0., -1.])
#         global_future_direction = state.target_orientation
#         local_future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction


    
#     # abs_global_velocity = 45
#     # local_3Dposition_future = np.zeros((3, 3))

#     # for i in range(3):
#     #     local_3Dposition_future[i] = local_future_direction * (abs_global_velocity * (i+1))

#     # local_2Dposition_future = local_3Dposition_future[:, 0::2]
    
#     # local_future_direction[1] = 0
#     # local_future_direction = utils.normalized(local_future_direction)
#     # local_3Ddirection_future = np.array([local_future_direction, local_future_direction, local_future_direction])
   
    
#     # local_2Ddirection_future = local_3Ddirection_future[:, 0::2]

#     local_3Dposition_future, local_3Ddirection_future = setRealFutureInfo()
#     local_2Dposition_future = local_3Dposition_future[:, 0::2]
#     local_2Ddirection_future = local_3Ddirection_future[:, 0::2]

    
#     # global position setting
#     # global_3Dposition_future = np.zeros((3, 3))

#     # for i in range(3):
#     #     global_3Dposition_future[i] = state.real_global_position + global_future_direction * (abs_global_velocity * (i+1))
#     # global_3Dposition_future[:, 1] = 0
#     # for i in range(3):
#     #     temp = np.array([0., 0., 0., 1])
#     #     temp[:3] = global_3Dposition_future[i]

#     global_3Dposition_future = []
#     for i in range(3):
#         temp = np.array([0., 0., 0., 1])
#         temp[:3] = local_3Dposition_future[i]
#         global_3Dposition_future.append((state.joint_list[0].getCharacterLocalFrame() @ temp)[:3])
    
#     global_3Dposition_future = np.array(global_3Dposition_future)
    
#     # global direction setting
#     global_3Ddirection_future = []
#     for i in range(3):
#         global_3Ddirection_future.append(state.joint_list[0].getCharacterLocalFrame()[:3, :3] @ local_3Ddirection_future[i])

#     global_3Ddirection_future = np.array(global_3Ddirection_future)


#     state.query_vector.set_global_future_position(global_3Dposition_future)
#     state.query_vector.set_global_future_direction(global_3Ddirection_future)

#     state.query_vector.set_future_position(np.array(local_2Dposition_future).reshape(6, ))
#     state.query_vector.set_future_direction(np.array(local_2Ddirection_future).reshape(6, ))
#     # state.query_vector.set_future_position(np.zeros_like(np.array(local_2Dposition_future).reshape(6, )))
#     # state.query_vector.set_future_direction(np.zeros_like(np.array(local_2Ddirection_future).reshape(6, )))
#     state.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
#     state.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
#     state.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))
#     feature_vector = state.query_vector.get_feature_list().copy()

#     #normalization
#     for i in range(0, 27):
#         feature_vector[i] = (feature_vector[i] - state.mean_array[i]) / state.std_array[i]
        
#     return feature_vector

 
# def draw_future_info():
#     # global 3d info
#     future_position = state.query_vector.get_global_future_position().reshape(3, 3)
#     future_direction = state.query_vector.get_global_future_direction().reshape(3, 3)

#     future_position[:, 1] = 0.
#     future_direction[:, 1] = 0.

#     glPointSize(20.)
#     glBegin(GL_POINTS)
#     glVertex3fv(future_position[0]/Joint.resize)
#     glVertex3fv(future_position[1]/Joint.resize)
#     glVertex3fv(future_position[2]/Joint.resize)
#     glEnd()

#     glLineWidth(5.)
#     glBegin(GL_LINES)
#     glVertex3fv(future_position[0]/Joint.resize)
#     glVertex3fv(future_position[0]/Joint.resize+future_direction[0])
#     glEnd()

#     glBegin(GL_LINES)
#     glVertex3fv(future_position[1]/Joint.resize)
#     glVertex3fv(future_position[1]/Joint.resize+future_direction[1])
#     glEnd()

#     glBegin(GL_LINES)
#     glVertex3fv(future_position[2]/Joint.resize)
#     glVertex3fv(future_position[2]/Joint.resize+future_direction[2])
#     glEnd()


# def reset_bvh_past_postion():
#     state.bvh_past_position = np.array([])

# def reset_bvh_past_orientation():
#     state.bvh_past_orientation = np.array([])

# def drawLocalFrame(M):
#     glPushMatrix()
#     glTranslatef(M[0][3]/Joint.resize, M[1][3]/Joint.resize, M[2][3]/Joint.resize)
#     rotationMatrix = np.identity(4)
#     rotationMatrix[:3, :3] = M[:3, :3]
#     glMultMatrixf(rotationMatrix.T)
#     glBegin(GL_LINE_STRIP)
#     glColor3ub(255, 0, 0)
#     glVertex3fv(np.array([0., 0., 0.]))
#     glVertex3fv(np.array([1., 0., 0.]))
#     glColor3ub(0, 255, 0)
#     glVertex3fv(np.array([0., 0., 0.]))
#     glVertex3fv(np.array([0., 3., 0.]))
#     glColor3ub(0, 0, 255)
#     glVertex3fv(np.array([0., 0., 0]))
#     glVertex3fv(np.array([0., 0., 5.]))
#     glEnd()
#     glPopMatrix()    

# def setRealFutureInfo():
#     future_matrices = [np.identity(4), np.identity(4), np.identity(4)]
#     for j in range (0, 3):
#         frame = np.array(MyWindow.state.future_frames[j])
#         bvh_future_orientation = np.identity(3)

#         for i in range(3, 6):
#             if state.joint_list[0].get_channel()[i].upper() == 'XROTATION':
#                 xr = frame[i]
#                 xr = np.radians(xr)
#                 Rx = np.array([[1., 0., 0.],
#                                [0, np.cos(xr), -np.sin(xr)],
#                                [0, np.sin(xr), np.cos(xr)]])
#                 bvh_future_orientation = bvh_future_orientation @ Rx
#             elif state.joint_list[0].get_channel()[i].upper() == 'YROTATION':
#                 yr = frame[i]
#                 yr = np.radians(yr)
#                 Ry = np.array([[np.cos(yr), 0, np.sin(yr)],
#                                [0, 1, 0],
#                                [-np.sin(yr), 0, np.cos(yr)]])
#                 bvh_future_orientation = bvh_future_orientation @ Ry
#             elif state.joint_list[0].get_channel()[i].upper() == 'ZROTATION':
#                 zr = frame[i]
#                 zr = np.radians(zr)
#                 Rz = np.array([[np.cos(zr), -np.sin(zr), 0],
#                                [np.sin(zr), np.cos(zr), 0],
#                                [0, 0, 1]])
#                 bvh_future_orientation = bvh_future_orientation @ Rz
#         future_matrices[j][:3, :3] = bvh_future_orientation
#         future_matrices[j][:3, 3] = np.array(frame[:3])

#     future_positions = [None, None, None]
#     for i in range(0, 3):
#         future_matrix = future_matrices[i]
#         root_position = [0., 0., 0., 1]
#         root_position[:3] = future_matrix[:3, 3]
#         root_position[1] = 0
#         future_positions[i] = root_position
    
#     future_directions = [None, None, None]
#     for i in range(0, 3):
#         future_matrix = future_matrices[i]
#         future_direction = future_matrix[:3, 1]  
#         future_direction[1] = 0
#         future_direction = utils.normalized(future_direction)
#         future_directions[i] = future_direction

#     bvh_curframe_character_local_frame = state.joint_list[0].getBvhCharacterLocalFrame()
    
#     character_local_future_positions = [None, None, None]
#     for i in range(0, 3):
#         future_position = future_positions[i]
#         character_local_future_position = np.linalg.inv(bvh_curframe_character_local_frame) @ np.array(future_position)
#         character_local_future_positions[i] = character_local_future_position[:3]

#     character_local_future_directions = [None, None, None]
#     for i in range(0, 3):
#         future_direction = future_directions[i]
#         character_local_future_direction = np.linalg.inv(bvh_curframe_character_local_frame)[:3, :3] @ np.array(future_direction)
#         character_local_future_directions[i] = character_local_future_direction


#     return np.array(character_local_future_positions), np.array(character_local_future_directions)