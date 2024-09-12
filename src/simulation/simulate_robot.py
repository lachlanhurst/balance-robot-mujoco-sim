from collections import deque
import time
import mujoco
import numpy as np
import pathlib
from OpenGL import GL
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QCheckBox, QGroupBox, QHBoxLayout
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtGui import (
    QOpenGLFunctions, QGuiApplication, QSurfaceFormat
)
import time

format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2,0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


class Viewport(QOpenGLWindow):

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()

        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        """ Sets a scale factor that is used to scale the OpenGL window to accommodate
        the high DPI scaling Qt does.
        """
        self.scale = scaleFactor

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):

    def __init__(self, model, data, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True
        self.reset()

    @property
    def real_time(self):
        return time.time() - self.real_time_start

    def run(self) -> None:
        while self.running:
            if self.data.time < self.real_time:
                mujoco.mj_step(self.model, self.data)
            else:
                time.sleep(0.0001)

    def stop(self):
        self.running = False
        self.wait()
    
    def reset(self):
        self.real_time_start = time.time()


class Window(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('scene.xml')))
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QVBoxLayout()
        layout_top = QHBoxLayout()
        reset_button = QPushButton("Reset")
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        layout_top.addWidget(reset_button)
        layout_top.addWidget(self.create_top())
        layout_top.setContentsMargins(8,0,8,0)
        layout.addLayout(layout_top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,0,0,0)
        layout.setStretch(1,1)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(640, 480)

        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

    def create_top(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        collision_checkbox = QCheckBox("Reflection")
        collision_checkbox.stateChanged.connect(self.toggle_reflection)
        layout.addWidget(collision_checkbox)
        stereo_checkbox = QCheckBox("Shadow")
        stereo_checkbox.stateChanged.connect(self.toggle_shadow)
        layout.addWidget(stereo_checkbox)
        layout.addStretch()
        w = QGroupBox("Robot Control")
        w.setLayout(layout)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w.setFixedHeight(60)
        return w

    def toggle_shadow(self, state):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(state)

    def toggle_reflection(self, state):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = bool(state)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        # Reset state and time.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
