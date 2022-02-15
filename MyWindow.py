from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from Joint import Joint
from bvh_handler import drawJoint, parsing_bvh

curFrame = []
timeStep = 0.2


class MyWindow(QOpenGLWidget):

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        # timer
        self.timer = QTimer(self)
        self.timer.setInterval(1000 * timeStep)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        # initalize value
        self.at = np.array([0., 0., 0.])
        self.w = np.array([3., 4., 5.])
        self.perspective = True
        self.click = False
        self.left = True

        self.oldx = 0
        self.oldy = 0
        self.A = np.radians(30)
        self.E = np.radians(36)

        # modes
        self.animation = False

        self.frame_num = -1

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

    def paintGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspective:
            gluPerspective(45, 1, .1, 40)

        else:
            glOrtho(-10, 10, -10, 10, -10, 20)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cam = self.at + self.w
        if np.cos(self.E) > 0:
            gluLookAt(cam[0], cam[1], cam[2], self.at[0], self.at[1], self.at[2], 0, 1, 0)
        else:
            gluLookAt(cam[0], cam[1], cam[2], self.at[0], self.at[1], self.at[2], 0, -1, 0)

        self.drawGrid()
        self.drawFrame()

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

        from bvh_handler import joint_list
        if len(joint_list) > 0:
            drawJoint(np.identity(4), joint_list[0])

        glDisable(GL_LIGHTING)

    # ===draw grid and frame===
    def drawFrame(self):
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

    def drawGrid(self):
        glBegin(GL_LINES)
        glColor3ub(60, 60, 60)
        for i in np.linspace(-20, 20, 50):
            glVertex3fv(np.array([-20, 0, i]))
            glVertex3fv(np.array([20, 0, i]))
            glVertex3fv(np.array([i, 0, -20]))
            glVertex3fv(np.array([i, 0, 20]))
        glEnd()

    # ===event handler===
    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_V:
            self.perspective = not self.perspective
        elif e.key() == Qt.Key_Space:
            self.animation = not self.animation
        self.update()

    def mousePressEvent(self, e):

        self.oldx = e.x()
        self.oldy = e.y()
        if e.buttons() == Qt.LeftButton:
            self.click = True
            self.left = True

        elif e.buttons() == Qt.RightButton:
            self.click = True
            self.left = False

        self.update()

    def mouseReleaseEvent(self, e):

        # if e.buttons() and (Qt.RightButton or Qt.LeftButton):
        self.click = False
        self.update()

    def wheelEvent(self, e):
        self.zoom(e.angleDelta().y() / 2880)
        self.update()

    def mouseMoveEvent(self, e):

        if self.click:
            newx = e.x()
            newy = e.y()

            dx = newx - self.oldx
            dy = newy - self.oldy

            if self.left:
                self.orbit(dx, dy)
            else:
                self.panning(dx, dy)

            self.oldx = newx
            self.oldy = newy
            self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        global curFrame
        curFrame = []
        Joint.resize = 1
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            paths = []
            for url in e.mimeData().urls():
                paths.append(str(url.toLocalFile()))

            if (paths[0].split('/'))[-1].split('.')[-1] != 'bvh':
                return

            with open(paths[0], 'r') as file:
                self.animation = False
                FPS = parsing_bvh(file)
                file_name = (paths[0].split('/'))[-1].strip(".bvh")

                from bvh_handler import num_of_frames, joint_list, frame_list
                print("1. File name : " + file_name)
                print("2. Number of frames : " + str(num_of_frames))
                print("3. FPS : " + str(FPS))
                print("4. Number of joints : " + str(len(joint_list)))
                print("5. List of all joint names : ")
                for j in joint_list:
                    print(j.get_joint_name(), end=' ')
                print('\n')

                self.frame_num = -1
                curFrame = frame_list[self.frame_num]

        else:
            e.ignore()

    def orbit(self, dx, dy):

        self.A -= np.radians(dx) / 5
        self.E += np.radians(dy) / 5

        distance = np.sqrt(self.w[0] ** 2 + self.w[1] ** 2 + self.w[2] ** 2)
        self.w = distance * np.array([np.cos(self.E) * np.sin(self.A), np.sin(self.E), np.cos(self.E) * np.cos(self.A)])

    def panning(self, dx, dy):

        up = np.array([0, 1, 0])
        w_ = self.w / np.sqrt(self.w[0] ** 2 + self.w[1] ** 2 + self.w[2] ** 2)
        u = np.cross(up, w_)
        u = u / np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        v = np.cross(w_, u)

        self.at += (-1 * dx * u + dy * v) / 30

    def zoom(self, yoffset):
        self.w -= self.w * yoffset / 5

    # ===update frame===
    def update_frame(self):
        global curFrame
        from bvh_handler import num_of_frames, frame_list

        if self.animation:
            self.frame_num += 1
            self.frame_num %= num_of_frames

        if len(frame_list) > 0:
            curFrame = frame_list[self.frame_num]
        self.update()
