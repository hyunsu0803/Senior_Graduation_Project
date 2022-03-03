import numpy as np
from Joint import Joint
from Feature import Feature
from utils import 12norm, normalized, exp

from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R

joint_list = []
num_of_frames = 0
frame_list = []
features = []
feature_vector = Feature()


def parsing_bvh(bvh):
	global num_of_frames, frame_list, joint_list
	frameTime = 0
	frame_list = []

	line = bvh.readline().split()

	if line[0] == 'HIERARCHY':
		line = bvh.readline().split()
		if line[0] == 'ROOT':
			joint_list = []
			buildJoint(bvh, line[1])  # build ROOT and other joints

	line = bvh.readline().split()

	if line[0] == 'MOTION':
		line = bvh.readline().split()
		if line[0] == 'Frames:':
			num_of_frames = int(line[1])
		line = bvh.readline().split()
		if line[0] == 'Frame' and line[1] == 'Time:':
			frameTime = float(line[2])

		for i in range(num_of_frames):
			line = bvh.readline().split()
			line = list(map(float, line))
			frame_list.append(line)
		last = [0] * len(frame_list[0])
		frame_list.append(last)

	FPS = int(1 / frameTime)
	return FPS


def buildJoint(bvh, joint_name):
	global joint_list
	# joint_list = []

	line = bvh.readline().split()  # remove '{'
	newJoint = Joint(joint_name)

	# check if it's foot joint
	# 이건 사용하는 data set에 따라 달라질 수 있음(관절 이름이 달라질 수 있으니까)
	if "Foot" in joint_name:
		newJoint.set_is_foot(True)

	newJoint.set_index(len(joint_list))

	joint_list.append(newJoint)

	line = bvh.readline().split()
	if line[0] == 'OFFSET':
		offset = np.array(list(map(float, line[1:])), dtype='float32')
		if np.sqrt(np.dot(offset, offset)) > Joint.resize:
			Joint.resize = np.sqrt(np.dot(offset, offset))
		newJoint.set_offset(offset)

	line = bvh.readline().split()
	if line[0] == 'CHANNELS':
		newJoint.set_channel(line[2:])

	while True:
		line = bvh.readline().split()
		if line[0] == 'JOINT':
			newJoint.append_child_joint(buildJoint(bvh, line[1]))

		elif line[0] == 'End' and line[1] == 'Site':
			line = bvh.readline().split()  # remove '{'
			line = bvh.readline().split()
			if line[0] == 'OFFSET':
				offset = np.array(list(map(float, line[1:])), dtype='float32')
				if np.sqrt(np.dot(offset, offset)) > Joint.resize:
					Joint.resize = np.sqrt(np.dot(offset, offset))
				newJoint.set_end_site(offset)
			line = bvh.readline().split()  # remove '}'

		elif line[0] == '}':
			return newJoint

def createFeatures(joint, parentMatrix, rootMatrix = None, feature):
	
	newMatrix = np.identity(4)
	cur_position = [0, 0, 0, 1]

	# get current joint's offset from parent joint
	curoffset = joint.get_offset()/Joint.resize

	# temp = np.identity(4)
	temp[:3, 3] = curoffset
	newMatrix = newMatrix @ temp

	# channel rotation
	# ROOT
	if len(joint.get_channel()) == 6:
		ROOTPOSITION = np.array(curFrame[:3], dtype='float32')
		ROOTPOSITION /= Joint.resize

		# move root's transformation matrix's origin using translation data
		temp = np.identity(4)
		temp[:3, 3] = ROOTPOSITION
		newMatrix = newMatrix @ temp

		for i in range(3, 6):
			if joint.get_channel()[i].upper() == 'XROTATION':
				xr = curFrame[i]
				xr = np.radians(xr)
				Rx = np.array([[1., 0., 0., 0.],
							   [0, np.cos(xr), -np.sin(xr), 0],
							   [0, np.sin(xr), np.cos(xr), 0],
							   [0., 0., 0., 1.]])
				newMatrix = newMatrix @ Rx
			elif joint.get_channel()[i].upper() == 'YROTATION':
				yr = curFrame[i]
				yr = np.radians(yr)
				Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
							   [0, 1, 0, 0],
							   [-np.sin(yr), 0, np.cos(yr), 0],
							   [0., 0., 0., 1.]])
				newMatrix = newMatrix @ Ry
			elif joint.get_channel()[i].upper() == 'ZROTATION':
				zr = curFrame[i]
				zr = np.radians(zr)
				Rz = np.array([[np.cos(zr), -np.sin(zr), 0, 0],
							   [np.sin(zr), np.cos(zr), 0, 0],
							   [0, 0, 1, 0],
							   [0, 0, 0, 1]])
				newMatrix = newMatrix @ Rz

	# JOINT
	else:
		index = joint.get_index()
		for i in range(3):
			if joint.get_channel()[i].upper() == 'XROTATION':
				xr = curFrame[(index + 1) * 3 + i]
				xr = np.radians(xr)
				Rx = np.array([[1., 0., 0., 0.],
							   [0, np.cos(xr), -np.sin(xr), 0],
							   [0, np.sin(xr), np.cos(xr), 0],
							   [0., 0., 0., 1.]])
				newMatrix = newMatrix @ Rx

			elif joint.get_channel()[i].upper() == 'YROTATION':
				yr = curFrame[(index + 1) * 3 + i]
				yr = np.radians(yr)
				Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
							   [0, 1, 0, 0],
							   [-np.sin(yr), 0, np.cos(yr), 0],
							   [0., 0., 0., 1.]])
				newMatrix = newMatrix @ Ry

			elif joint.get_channel()[i].upper() == 'ZROTATION':
				zr = curFrame[(index + 1) * 3 + i]
				zr = np.radians(zr)
				Rz = np.array([[np.cos(zr), -np.sin(zr), 0, 0],
							   [np.sin(zr), np.cos(zr), 0, 0],
							   [0, 0, 1, 0],
							   [0, 0, 0, 1]])
				newMatrix = newMatrix @ Rz


	joint.set_transform_matrix(parentMatrix @ newMatrix)
	transform_matrix = joint.get_transform_matrix()
	global_position = transform_matrix @ np.array([0., 0., 0., 1.])

	# set parent's global position (if it is root joint, parent_position is current_position)
	if joint.get_is_root() is not None:
		parent_position = global_position
	else: parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

	cur_position = global_position

	# Check if it's Root joint, otherwise update Joint class's data
	# velocity, rotation velocity update 시키기

	if joint.get_is_root() is not None:
		rootMatrix = joint.get_transform_matrix()

	else:
		# get root local position and root local velocity
		new_root_local_position = (rootMatrix.T @ global_position)[:3]	  #local to root joint
		past_root_local_position = joint.get_root_local_position()	#local to root joint
		root_local_velocity = ((new_root_local_position - past_root_local_position)/timeStep)

		# get root local rotation and root local angular velocity
		new_root_local_rotation_matrix = (rootMatrix.T @ transform_matrix)[:3,:3]
		r = R.from_matrix(new_root_local_rotation_matrix)
		new_root_local_rotation = np.array(r.as_quat())
		past_root_local_rotation = joint.get_root_local_rotation()

		# set joint class's value
		joint.set_global_position(global_position[:3])
		joint.set_root_local_velocity(root_local_velocity)
		joint.set_root_local_position(new_root_local_position[:3])
		joint.set_root_local_rotation(new_root_local_rotation)

def set_feature_vector():
	global joint_list, feature_vector

	two_foot_position = []
	two_foot_velocity = []
	hip_velocity = []

	for joint in joint_list:
		if joint.get_is_root() is not None:
			hip_velocity.append(joint.get_root_local_velocity())
		elif joint.get_is_foot() is not None:
			two_foot_position.append(joint.get_root_local_position())
			two_foot_velocity.append(joint.get_root_local_velocity())

	feature_vector.set_foot_position(np.array(two_foot_position).reshape(1, 6))
	feature_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(1, 6))
	feature_vector.set_hip_velocity(np.array(hip_velocity))


def main():
	global curFrame
	curFrame = []
	Joint.resize = 1

if __name__ == "__main__":
	main()
