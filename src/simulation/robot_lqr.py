import mujoco
from scipy.spatial.transform import Rotation

# obtained from running `calculate_lqr_gains.py`
LQR_K = [-2.1402165848237837, -0.03501370844016172, 5.9748026764525894e-18, 2.236067977499789]
WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 500.0 # rad/s


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


"""
Basic LQR controller implementation
"""
class RobotLqr:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

    def set_velocity_linear_set_point(self, vel: float) -> None:
        """
        Sets the target velocity of the robot
        """
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw: float) -> None:
        """
        The yaw value is added to one wheel, and subtracted from the other
        to produce a yaw motion (turn)
        """
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        # print(angles)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        # print(angular)
        return angular[0]

    def get_wheel_velocity(self) -> float:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0 * -1 + vel_m_1) / 2.0

    def calculate_lqr_velocity(self) -> float:
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        # apply a filter to pitch dot, and velocity
        # without these filters the controller seems to lack necessary dampening
        # would like to know why!
        self.pitch_dot_filtered = (self.pitch_dot_filtered * .975) + (pitch_dot * .025)
        self.velocity_angular_filtered = (self.velocity_angular_filtered * .975) + (self.get_wheel_velocity() * .025)

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS

        lqr_v = LQR_K[0] * (0 - pitch) + LQR_K[1] * self.pitch_dot_filtered + LQR_K[2] * 0 + LQR_K[3] * velocity_linear_error 
        return -lqr_v / WHEEL_RADIUS
    
    def update_motor_speed(self, ) -> None:
        vel = self.calculate_lqr_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl = [-vel + self.yaw]
        self.data.actuator('motor_r_wheel').ctrl = [vel + self.yaw]

    def reset(self):
        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

        # face a random direction
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        # rotate and pitch slightly
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        self.data.qpos[3:7] = rotation.as_quat()

        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]
