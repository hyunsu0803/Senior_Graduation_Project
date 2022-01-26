import sys
from PyQt5.QtWidgets import QApplication
from MyWindow import MyWindow


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setWindowTitle("my BVH viewer")
    window.setFixedSize(900, 900)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
