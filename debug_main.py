from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
# from PyQt5. QtWidgets import QApplication
from drawTpose import drawTpose
import numpy as np

# from Joint import Joint
import bvh_handler
from bvh_handler import drawJoint, reset_bvh_past_postion
from motion_matching_test import QnA


class state:
    curFrame = []
    curFrame_index = 0
	# timeStep = 0.2


class MyWindow(QOpenGLWidget):

	def __init__(self, parent=None):
		super(MyWindow, self).__init__(parent)
		self.setAcceptDrops(True)
		self.setMouseTracking(True)

		# update frame
		self.matching_num = -1
		self.coming_soon_10frames = []
		self.FPS = 1

		# timer
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update_frame)
		self.timer.setInterval(1000 / self.FPS)
		self.timer.start()

		# initialize value
		self.at = np.array([0., 0., 0.])
		self.w = np.array([4., 4., 4.])
		self.perspective = True
		self.click = False
		self.left = True

		self.oldx = 0
		self.oldy = 0
		self.A = np.radians(30)
		self.E = np.radians(36)

		# modes
		self.animation = True

		self.frame_num = -1

		# draw T pose
		drawTpose()

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
			gluLookAt(cam[0], cam[1], cam[2], self.at[0],
			          self.at[1], self.at[2], 0, 1, 0)
		else:
			gluLookAt(cam[0], cam[1], cam[2], self.at[0],
			          self.at[1], self.at[2], 0, -1, 0)

		self.drawGrid()
		self.drawFrame()

		glEnable(GL_LIGHTING)  # try to uncomment: no lighting
		glEnable(GL_LIGHT0)

		# try to uncomment: lighting will be incorrect if you scale the object
		glEnable(GL_NORMALIZE)

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

		if len(state.curFrame) > 0:
			drawJoint(np.identity(4), bvh_handler.state.joint_list[0])

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

		elif e.key() == Qt.Key_Up:
			self.coming_soon_10frames, self.FPS = QnA(key_input="UP")
			self.matching_num = 0
		elif e.key() == Qt.Key_Down:
			self.coming_soon_10frames, self.FPS = QnA(key_input="DOWN")
			self.matching_num = 0
		elif e.key() == Qt.Key_Left:
			self.coming_soon_10frames, self.FPS = QnA(key_input="LEFT")
			self.matching_num = 0
		elif e.key() == Qt.Key_Right:
			self.coming_soon_10frames, self.FPS = QnA(key_input="RIGHT")
			self.matching_num = 0

		self.timer.setInterval(1000 / self.FPS)

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
		self.zoom(e.angleDelta().y() / 2000)
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

	def orbit(self, dx, dy):

		self.A -= np.radians(dx) / 5
		self.E += np.radians(dy) / 5

		distance = np.sqrt(self.w[0] ** 2 + self.w[1] ** 2 + self.w[2] ** 2)
		self.w = distance * np.array([np.cos(self.E) * np.sin(self.A),
		                             np.sin(self.E), np.cos(self.E) * np.cos(self.A)])

	def panning(self, dx, dy):

		up = np.array([0, 1, 0])
		w_ = self.w / np.sqrt(self.w[0] ** 2 + self.w[1] ** 2 + self.w[2] ** 2)
		u = np.cross(up, w_)
		u = u / np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
		v = np.cross(w_, u)

		self.at += (-1 * dx * u + dy * v) / 30

	def zoom(self, yoffset):
		self.w -= self.w * yoffset / 5


    def update_frame(self):
        if len(bvh_handler.state.frame_list) != 0:
            state.curFrame = bvh_handler.state.frame_list[state.curFrame_index]
            state.curFrame_index = (state.curFrame_index+1)%len(bvh_handler.state.frame_list)

		# if state.curFrame == []:
		# 	self.coming_soon_10frames, self.FPS = QnA(key_input="init")
		# 	reset_bvh_past_postion()
		# 	self.timer.setInterval(1000 / self.FPS)

		# elif self.matching_num % 10 == 9:
		# 	self.coming_soon_10frames, self.FPS = QnA()
		# 	reset_bvh_past_postion()
		# 	self.timer.setInterval(1000 / self.FPS)
			

		# self.matching_num = (self.matching_num + 1) % 10

		# state.curFrame = self.coming_soon_10frames[self.matching_num]
		self.update()