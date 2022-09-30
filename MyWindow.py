from sre_parse import State
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QOpenGLWidget, QLabel
from OpenGL.GL import *
from OpenGL.GLU import *
# from PyQt5. QtWidgets import QApplication
import numpy as np

from MotionMatching_keyboard import MotionMatching_keyboard
from MotionMatching_task import MotionMatching_task
from MotionMatching_training import MotionMatching_training


class MyWindow(QOpenGLWidget):

	def __init__(self, parent=None):
		super(MyWindow, self).__init__(parent)
		self.setAcceptDrops(True)
		self.setMouseTracking(True)

		# update frame
		self.matching_num = -1
		self.FPS = 30

		# timer
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update_frame)
		self.timer.setInterval(1000 / self.FPS)
		self.timer.start()

		# initialize value
		self.at = np.array([0., 0., 0.])
		self.w = np.array([0., 25., -20.])
		self.perspective = True
		self.click = False
		self.left = True

		self.oldx = 0
		self.oldy = 0
		self.A = np.radians(30)
		self.E = np.radians(36)

		# self.MODE = "KEYBOARD"
		self.MODE = "TASK"

		if self.MODE == "KEYBOARD":
			self.motion_matching_system = MotionMatching_keyboard()
		elif self.MODE == "TASK":
			self.motion_matching_system = MotionMatching_task()

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
			gluPerspective(60, 1, .1, 50)

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

		if self.motion_matching_system.is_character_drawable():
			glPushMatrix()
			self.motion_matching_system.calculate_and_draw_character()
			self.motion_matching_system.draw_future_info()
			glPopMatrix()

		if self.motion_matching_system.is_character_drawable():
			if self.MODE == "TASK":
				glPushMatrix()
				self.motion_matching_system.draw_goal_position()
				glPopMatrix()


	# ===draw grid and frame===
	def drawFrame(self):
		glBegin(GL_LINES)
		glColor3ub(255, 0, 0)
		glVertex3fv(np.array([0., 0., 0.]))
		glVertex3fv(np.array([30., 0., 0.]))
		glColor3ub(0, 255, 0)
		glVertex3fv(np.array([0., 0., 0.]))
		glVertex3fv(np.array([0., 1., 0.]))
		glColor3ub(0, 0, 255)
		glVertex3fv(np.array([0., 0., 0]))
		glVertex3fv(np.array([0., 0., 30.]))
		glEnd()

	def drawGrid(self):
		glBegin(GL_LINES)
		glColor3ub(60, 60, 60)
		for i in np.linspace(-30, 30, 50):
			glVertex3fv(np.array([-30, 0, i]))
			glVertex3fv(np.array([30, 0, i]))
			glVertex3fv(np.array([i, 0, -30]))
			glVertex3fv(np.array([i, 0, 30]))
		glEnd()

	# ===event handler===
	def keyPressEvent(self, e):

		if e.key() == Qt.Key_Escape:
			self.close()
		elif e.key() == Qt.Key_V:
			self.perspective = not self.perspective

		if self.MODE == "KEYBOARD":
			if e.key() == Qt.Key_Up:
				print("--------up---------")
				self.motion_matching_system.change_curFrame("UP")

			elif e.key() == Qt.Key_Down:
				print("--------down---------")
				self.motion_matching_system.change_curFrame("DOWN")

			elif e.key() == Qt.Key_Left:
				print("--------left---------")
				self.motion_matching_system.change_curFrame("LEFT")
				
			elif e.key() == Qt.Key_Right:
				print("--------right---------")
				self.motion_matching_system.change_curFrame("RIGHT")

			elif e.key() == Qt.Key_R:
				self.motion_matching_system.reset_motion_matching()

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

	def update_frame(self):
		self.motion_matching_system.change_curFrame()
		self.update()

		#QApplication.processEvents()
