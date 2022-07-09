import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GLU import *
import ctypes

at = np.array([0., 0., 0.])
w = np.array([3., 4., 5.])
perspective = True
click = False
left = True

oldx = 0
oldy = 0
A = np.radians(30)
E = np.radians(36)

# modes
animation = False
box = False

joint_list = []
num_of_frames = 0
frame_list = []
curFrame = []
f = -1
cubeVarr = None


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


def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if perspective:
        gluPerspective(45, 1, .1, 40)
    else:
        glOrtho(-10, 10, -10, 10, -10, 20)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    cam = at + w
    if np.cos(E) > 0:
        gluLookAt(cam[0], cam[1], cam[2], at[0], at[1], at[2], 0, 1, 0)
    else:
        gluLookAt(cam[0], cam[1], cam[2], at[0], at[1], at[2], 0, -1, 0)

    drawGrid()
    drawFrame()

    glEnable(GL_LIGHTING)  # try to uncomment: no lighting
    glEnable(GL_LIGHT0)

    glEnable(GL_NORMALIZE)  # try to uncomment: lighting will be incorrect if you scale the object

    # light0 position
    glPushMatrix()
    lightPos = (3., 4., 5., 1.)  # try to change 4th element to 0. or 1.
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
    glPopMatrix()

    # light0 intensity for each color channel
    lightColor = (1., 1., 1., 1.)
    ambientLightColor = (.5, .5, .5, 1.)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor)
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLightColor)

    # material reflectance for each color channel
    specularObjectColor = (1., 1., 1., 1.)
    glMaterialfv(GL_FRONT, GL_SHININESS, 10)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specularObjectColor)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, specularObjectColor)

    if len(joint_list) > 0:
        drawJoint(np.identity(4), joint_list[0])

    glDisable(GL_LIGHTING)


def button_callback(window, button, action, mod):
    global click, left
    global oldx, oldy

    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            oldx, oldy = glfw.get_cursor_pos(window)
            click = True
            left = True

        elif action == glfw.RELEASE:
            click = False

    elif button == glfw.MOUSE_BUTTON_RIGHT:
        if action == glfw.PRESS:
            oldx, oldy = glfw.get_cursor_pos(window)
            click = True
            left = False

        elif action == glfw.RELEASE:
            click = False


def cursor_callback(window, xpos, ypos):
    global oldx, oldy

    if click:
        newx = xpos
        newy = ypos

        dx = newx - oldx
        dy = newy - oldy

        if left:
            orbit(dx, dy)
        else:
            panning(dx, dy)

        oldx = newx
        oldy = newy


def key_callback(window, key, scancode, action, mods):
    global perspective, animation, wireframe, smooth, animation
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_V:
            perspective = not perspective
        elif key == glfw.KEY_H:
            animation = True
        elif key == glfw.KEY_Z:
            wireframe = not wireframe
        elif key == glfw.KEY_S:
            smooth = not smooth
        elif key == glfw.KEY_SPACE:
            animation = not animation


# orbit
def orbit(dx, dy):
    global w, A, E

    A -= np.radians(dx) / 5
    E += np.radians(dy) / 5

    distance = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    w = distance * np.array([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)])


def panning(dx, dy):
    global at

    up = np.array([0, 1, 0])
    w_ = w / np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
    u = np.cross(up, w_)
    u = u / np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    v = np.cross(w_, u)

    at += (-1 * dx * u + dy * v) / 30


# zoom
def scroll_callback(window, xoffset, yoffset):
    zoom(yoffset)


def zoom(yoffset):
    global w
    w -= w * yoffset / 5


def drop_callback(window, paths):
    global animation, curFrame, frame_list, joint_list, f, box

    if (paths[0].split('\\'))[-1].split('.')[-1] != 'bvh':
        return

    frame_list = []
    joint_list = []
    curFrame = []
    Joint.resize = 1

    with open(paths[0], 'r') as file:

        animation = False
        FPS = parsing_bvh(file)
        file_name = (paths[0].split('\\'))[-1].strip(".bvh")
        box = False

        print("1. File name : " + file_name)
        print("2. Number of frames : " + str(num_of_frames))
        print("3. FPS : " + str(FPS))
        print("4. Number of joints : " + str(len(joint_list)))
        print("5. List of all joint names : ")
        for j in joint_list:
            print(j.get_joint_name(), end=' ')
        print('\n')

        f = -1
        curFrame = frame_list[f]


def parsing_bvh(bvh):
    global num_of_frames, frame_list
    frameTime = 0

    line = bvh.readline().split()

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        if line[0] == 'ROOT':
            buildJoint(bvh, line[1])     # build ROOT and other joints

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

    line = bvh.readline().split()   # remove '{'
    newJoint = Joint(joint_name)
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
            line = bvh.readline().split()   # remove '{'
            line = bvh.readline().split()
            if line[0] == 'OFFSET':
                offset = np.array(list(map(float, line[1:])), dtype='float32')
                if np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))
                newJoint.set_end_site(offset)
            line = bvh.readline().split()   # remove '}'

        elif line[0] == '}':
            return newJoint


def drawJoint(parentMatrix, joint):
    glPushMatrix()
    newMatrix = np.identity(4)

    parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    # offset translation
    curoffset = joint.get_offset() / Joint.resize

    temp = np.identity(4)
    temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    cur_position = parentMatrix @ newMatrix @ np.array([0., 0., 0., 1.])

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        ROOTPOSITION = np.array(curFrame[:3], dtype='float32')

        ROOTPOSITION /= Joint.resize
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

    if not box:
        # Let's draw
        glBegin(GL_LINES)
        glColor3ub(255, 255, 255)
        glVertex3fv(parent_position[:3])
        glVertex3fv(cur_position[:3])
        glEnd()

        # draw end effector
        if joint.get_end_site() is not None:
            end_offset = joint.get_end_site() / Joint.resize
            endMatrix = np.identity(4)
            endMatrix[:3, 3] = end_offset
            end_position = parentMatrix @ newMatrix @ endMatrix @ np.array([0., 0., 0., 1.])

            glBegin(GL_LINES)
            glColor3ub(255, 255, 255)
            glVertex3fv(cur_position[:3])
            glVertex3fv(end_position[:3])
            glEnd()

        # draw child joints
        else:
            for j in joint.get_child():
                drawJoint(joint.get_transform_matrix(), j)

    # if box
    else:
        v = cur_position - parent_position
        box_length = l2norm(v)
        v = normalized(v)
        rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
        check = np.dot(np.array([0, 1, 0]), v[:3])
        # under 90
        if check >= 0:
            rotate_angle = np.arcsin(l2norm(rotation_vector))
            rotation_vector = normalized(rotation_vector) * rotate_angle
        # over 90
        else:
            rotate_angle = np.arcsin(l2norm(rotation_vector)) + np.pi
            rotate_angle *= -1

            # not 180
            if l2norm(rotation_vector) != 0:
                rotation_vector = normalized(rotation_vector) * rotate_angle
            # if 180, rotate_vector becomes (0, 0, 0)
            else:
                rotation_vector = np.array([0., 0., 1.]) * rotate_angle

        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = exp(rotation_vector[:3])

        glPushMatrix()
        glTranslatef(parent_position[0], parent_position[1], parent_position[2])
        glMultMatrixf(rotation_matrix.T)
        glTranslatef(0., box_length / 2, 0.)
        glScalef(.05, box_length, .05)
        drawUnitCube()
        glPopMatrix()

        # draw end effector
        if joint.get_end_site() is not None:
            end_offset = joint.get_end_site() / joint.resize
            endMatrix = np.identity(4)
            endMatrix[:3, 3] = end_offset
            end_position = parentMatrix @ newMatrix @ endMatrix @ np.array([0., 0., 0., 1.])

            v = end_position - cur_position
            box_length = l2norm(v)
            v = normalized(v)
            rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
            check = np.dot(np.array([0, 1, 0]), v[:3])
            if check >= 0:
                rotate_angle = np.arcsin(l2norm(rotation_vector))
            else:
                rotate_angle = np.arcsin(l2norm(rotation_vector)) + np.pi
                rotate_angle *= -1
            rotation_vector = normalized(rotation_vector) * rotate_angle
            rotation_matrix = np.identity(4)
            rotation_matrix[:3, :3] = exp(rotation_vector[:3])

            glPushMatrix()
            glTranslatef(cur_position[0], cur_position[1], cur_position[2])
            glMultMatrixf(rotation_matrix.T)
            glTranslatef(0., box_length / 2, 0.)
            glScalef(.05, box_length, .05)
            drawUnitCube()
            glPopMatrix()

        # draw child joints
        else:
            for j in joint.get_child():
                drawJoint(joint.get_transform_matrix(), j)

    glPopMatrix()


def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([-30., 0., 0.]))
    glVertex3fv(np.array([30., 0., 0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([0., 1., 0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0., 0., -30]))
    glVertex3fv(np.array([0., 0., 30.]))
    glEnd()


def drawGrid():
    glBegin(GL_LINES)
    glColor3ub(60, 60, 60)
    for i in np.linspace(-20, 20, 50):
        glVertex3fv(np.array([-20, 0, i]))
        glVertex3fv(np.array([20, 0, i]))
        glVertex3fv(np.array([i, 0, -20]))
        glVertex3fv(np.array([i, 0, 20]))
    glEnd()


def createVertexArraySeparate():
    varr = np.array([
        [0, 1, 0],  # v0 normal
        [0.5, 0.5, -0.5],  # v0 position
        [0, 1, 0],  # v1 normal
        [-0.5, 0.5, -0.5],  # v1 position
        [0, 1, 0],  # v2 normal
        [-0.5, 0.5, 0.5],  # v2 position

        [0, 1, 0],  # v3 normal
        [0.5, 0.5, -0.5],  # v3 position
        [0, 1, 0],  # v4 normal
        [-0.5, 0.5, 0.5],  # v4 position
        [0, 1, 0],  # v5 normal
        [0.5, 0.5, 0.5],  # v5 position

        [0, -1, 0],  # v6 normal
        [0.5, -0.5, 0.5],  # v6 position
        [0, -1, 0],  # v7 normal
        [-0.5, -0.5, 0.5],  # v7 position
        [0, -1, 0],  # v8 normal
        [-0.5, -0.5, -0.5],  # v8 position

        [0, -1, 0],
        [0.5, -0.5, 0.5],
        [0, -1, 0],
        [-0.5, -0.5, -0.5],
        [0, -1, 0],
        [0.5, -0.5, -0.5],

        [0, 0, 1],
        [0.5, 0.5, 0.5],
        [0, 0, 1],
        [-0.5, 0.5, 0.5],
        [0, 0, 1],
        [-0.5, -0.5, 0.5],

        [0, 0, 1],
        [0.5, 0.5, 0.5],
        [0, 0, 1],
        [-0.5, -0.5, 0.5],
        [0, 0, 1],
        [0.5, -0.5, 0.5],

        [0, 0, -1],
        [0.5, -0.5, -0.5],
        [0, 0, -1],
        [-0.5, -0.5, -0.5],
        [0, 0, -1],
        [-0.5, 0.5, -0.5],

        [0, 0, -1],
        [0.5, -0.5, -0.5],
        [0, 0, -1],
        [-0.5, 0.5, -0.5],
        [0, 0, -1],
        [0.5, 0.5, -0.5],

        [-1, 0, 0],
        [-0.5, 0.5, 0.5],
        [-1, 0, 0],
        [-0.5, 0.5, -0.5],
        [-1, 0, 0],
        [-0.5, -0.5, -0.5],

        [-1, 0, 0],
        [-0.5, 0.5, 0.5],
        [-1, 0, 0],
        [-0.5, -0.5, -0.5],
        [-1, 0, 0],
        [-0.5, -0.5, 0.5],

        [1, 0, 0],
        [0.5, 0.5, -0.5],
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [1, 0, 0],
        [0.5, -0.5, 0.5],

        [1, 0, 0],
        [0.5, 0.5, -0.5],
        [1, 0, 0],
        [0.5, -0.5, 0.5],
        [1, 0, 0],
        [0.5, -0.5, -0.5],
        # ...
    ], 'float32')
    return varr


def drawUnitCube():
    global cubeVarr
    varr = cubeVarr
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 6 * varr.itemsize, varr)
    glVertexPointer(3, GL_FLOAT, 6 * varr.itemsize, ctypes.c_void_p(varr.ctypes.data + 3 * varr.itemsize))
    glDrawArrays(GL_TRIANGLES, 0, int(varr.size / 6))


def l2norm(v):
    return np.sqrt(np.dot(v, v))


def normalized(v):
    l = l2norm(v)
    if l == 0:
        l = 1
    return 1 / l * np.array(v)


def exp(rv):
    th = l2norm(rv)
    costh = np.cos(th)
    sinth = np.sin(th)

    rv = normalized(rv)
    ux = rv[0]
    uy = rv[1]
    uz = rv[2]

    R = np.array(
        [[costh + ux * ux * (1 - costh), ux * uy * (1 - costh) - uz * sinth, ux * uz * (1 - costh) + uy * sinth],
         [uy * ux * (1 - costh) + uz * sinth, costh + uy * uy * (1 - costh), uy * uz * (1 - costh) - ux * sinth],
         [uz * ux * (1 - costh) - uy * sinth, uz * uy * (1 - costh) + ux * sinth, costh + uz * uz * (1 - costh)]])
    return R


def main():
    global curFrame, f, cubeVarr

    if not glfw.init():
        return
    window = glfw.create_window(900, 900, 'my BVH viewer', None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_drop_callback(window, drop_callback)

    cubeVarr = createVertexArraySeparate()

    f = num_of_frames
    while not glfw.window_should_close(window):
        if animation:
            f += 1
            f %= num_of_frames

        if len(frame_list) > 0:
            curFrame = frame_list[f]

        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()