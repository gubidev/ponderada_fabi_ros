#!/usr/bin/env python3
"""
turtle_controller.py — ROS 2 node that draws image contours with turtlesim.

Architecture
------------
1. At startup the node runs the CV pipeline (cv_pipeline.py) on the input
   image, then asks the path planner (path_planner.py) for an ordered list
   of turtlesim waypoints and a parallel pen-state list.
2. A 20 Hz timer fires _control_loop(), which implements a simple two-stage
   proportional controller:
     a. Rotate in place until the heading error is small (< ~9°).
     b. Drive forward while applying a proportional angular correction.
3. When the turtle reaches a waypoint it advances to the next one and
   updates the pen state via the /turtle1/set_pen service.
4. Large gaps between successive waypoints are traversed with the pen up,
   so only true contour strokes appear on the canvas.

Proportional controller
-----------------------
Desired heading:  θ_d  = atan2(dy, dx)
Heading error:    e_θ  = θ_d − θ_current  (normalised to [−π, π])
Angular command:  ω    = Kp_ang · e_θ     (clamped to ±4 rad/s)
Linear command:   v    = Kp_lin · dist     (clamped to max_speed)

When |e_θ| > 0.15 rad (~9°) the turtle rotates in place (v = 0) before
moving forward.  This prevents the turtle from cutting corners and drawing
extra lines where the pen is down.
"""

import math
import os
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import SetPen, TeleportAbsolute

from turtle_draw.cv_pipeline import CVPipeline
from turtle_draw.path_planner import PathPlanner


class TurtleController(Node):

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__('turtle_controller')

        # ---- ROS parameters -----------------------------------------
        self.declare_parameter('image_path',    '')
        self.declare_parameter('max_points',    500)
        self.declare_parameter('jump_threshold', 0.8)
        self.declare_parameter('sigma',          1.5)
        self.declare_parameter('ksize',          5)
        self.declare_parameter('low_ratio',      0.15)
        self.declare_parameter('high_ratio',     0.35)
        self.declare_parameter('max_dim',        400)
        self.declare_parameter('visualize',      False)

        img_path = self.get_parameter('image_path').get_parameter_value().string_value
        if not img_path:
            img_path = self._default_image_path()
        self.get_logger().info(f'Image path: {img_path}')

        # ---- CV pipeline --------------------------------------------
        pipeline = CVPipeline(img_path)
        edges = pipeline.run(
            sigma=self.get_parameter('sigma').get_parameter_value().double_value,
            ksize=self.get_parameter('ksize').get_parameter_value().integer_value,
            low_ratio=self.get_parameter('low_ratio').get_parameter_value().double_value,
            high_ratio=self.get_parameter('high_ratio').get_parameter_value().double_value,
            max_dim=self.get_parameter('max_dim').get_parameter_value().integer_value,
            visualize=self.get_parameter('visualize').get_parameter_value().bool_value,
        )
        edge_count = int((edges > 0).sum())
        self.get_logger().info(f'Edge pixels detected: {edge_count}')

        # ---- Path planning ------------------------------------------
        planner = PathPlanner(
            edges,
            max_points=self.get_parameter('max_points').get_parameter_value().integer_value,
            jump_threshold=self.get_parameter('jump_threshold').get_parameter_value().double_value,
        )
        self.waypoints: List[Tuple[float, float]]
        self.pen_flags: List[bool]
        self.waypoints, self.pen_flags = planner.plan()
        self.get_logger().info(f'Waypoints planned: {len(self.waypoints)}')

        self.wp_idx: int = 0

        # ---- ROS interfaces -----------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub    = self.create_subscription(
            Pose, '/turtle1/pose', self._pose_cb, 10
        )
        self.set_pen_cli   = self.create_client(SetPen,             '/turtle1/set_pen')
        self.teleport_cli  = self.create_client(TeleportAbsolute,   '/turtle1/teleport_absolute')

        self.pose: Optional[Pose] = None
        self._pen_is_down: bool   = False
        self._skip_ticks: int     = 0   # ticks to pause after a teleport

        # Wait for services to be available
        for cli, name in [
            (self.set_pen_cli,  '/turtle1/set_pen'),
            (self.teleport_cli, '/turtle1/teleport_absolute'),
        ]:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'Waiting for service {name} …')

        # ---- Initial position ---------------------------------------
        # Lift pen, teleport to the first waypoint, then let the loop draw.
        self._set_pen(down=False)
        x0, y0 = self.waypoints[0]
        self._teleport(x0, y0, 0.0)
        self.wp_idx     = 1          # first waypoint already reached via teleport
        self._skip_ticks = 10        # wait ~500 ms for turtlesim to process

        # ---- Proportional controller gains --------------------------
        self.Kp_linear  = 1.0
        self.Kp_angular = 4.0
        self.goal_tol   = 0.18       # waypoint considered reached within this radius
        self.max_speed  = 1.0        # m/s cap
        self.max_omega  = 2.0        # rad/s cap

        # ---- Control timer (20 Hz) ----------------------------------
        self.timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info('TurtleController ready — drawing starts …')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_image_path(self) -> str:
        """Resolve dog.jpg from the ROS 2 share directory (after colcon build)."""
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(
                get_package_share_directory('turtle_draw'), 'images', 'dog.jpg'
            )
        except Exception:
            # Fallback for running directly from the source tree
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, '..', 'images', 'dog.jpg')

    def _set_pen(self, down: bool) -> None:
        """Call /turtle1/set_pen — skip if the state is already correct."""
        if down == self._pen_is_down:
            return
        req = SetPen.Request()
        if down:
            req.r, req.g, req.b = 255, 255, 255   # cor padrão turtlesim (branco)
            req.width = 2
            req.off   = 0
        else:
            req.off = 1
        self.set_pen_cli.call_async(req)
        self._pen_is_down = down

    def _teleport(self, x: float, y: float, theta: float) -> None:
        """Instant position jump — used only for the initial placement."""
        req = TeleportAbsolute.Request()
        req.x, req.y, req.theta = float(x), float(y), float(theta)
        self.teleport_cli.call_async(req)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pose_cb(self, msg: Pose) -> None:
        self.pose = msg

    def _control_loop(self) -> None:
        """
        Called at 20 Hz.  Drives the turtle toward the current waypoint using
        a proportional heading-then-forward controller.
        """
        # Wait out any post-teleport settling ticks
        if self._skip_ticks > 0:
            self._skip_ticks -= 1
            return

        # Wait until the first pose message arrives
        if self.pose is None:
            return

        # ---- Done? --------------------------------------------------
        if self.wp_idx >= len(self.waypoints):
            self._set_pen(down=False)
            self.get_logger().info('Drawing complete — all waypoints visited.')
            self.timer.cancel()
            return

        # ---- Set pen state for this segment -------------------------
        self._set_pen(down=self.pen_flags[self.wp_idx])

        # ---- Compute error to current waypoint ----------------------
        tx, ty  = self.waypoints[self.wp_idx]
        dx      = tx - self.pose.x
        dy      = ty - self.pose.y
        dist    = math.hypot(dx, dy)

        # ---- Waypoint reached? advance ------------------------------
        if dist < self.goal_tol:
            self.wp_idx += 1
            return

        # ---- Proportional controller --------------------------------
        desired_angle = math.atan2(dy, dx)
        angle_err     = desired_angle - self.pose.theta

        # Normalise heading error to (−π, π]
        angle_err = (angle_err + math.pi) % (2.0 * math.pi) - math.pi

        cmd = Twist()

        if abs(angle_err) > 0.15:
            # Phase 1: rotate in place until roughly aligned
            cmd.linear.x  = 0.0
            cmd.angular.z = max(-self.max_omega,
                                min(self.Kp_angular * angle_err, self.max_omega))
        else:
            # Phase 2: drive forward with proportional heading correction
            speed = self.max_speed
            cmd.linear.x  = min(self.Kp_linear * dist, speed)
            cmd.angular.z = max(-self.max_omega,
                                min(self.Kp_angular * angle_err, self.max_omega))

        self.cmd_vel_pub.publish(cmd)


# ----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()