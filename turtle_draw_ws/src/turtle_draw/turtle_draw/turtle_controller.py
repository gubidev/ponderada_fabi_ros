#!/usr/bin/env python3
"""
turtle_controller.py — ROS 2 node that draws image contours with turtlesim.

Architecture
------------
1. At startup the node runs the CV pipeline (cv_pipeline.py) on the input
   image, then asks the path planner (path_planner.py) for a scan-line list
   of turtlesim waypoints and a parallel pen-state list.

2. A 20 Hz timer fires _control_loop(), which applies one of two strategies
   depending on the pen state of the current waypoint:

     Pen UP  (pen_flags[i] = False)
       → TeleportAbsolute to the waypoint instantly, with theta = 0 so the
         turtle is already facing right for the next horizontal stroke.
         No movement on canvas.

     Pen DOWN (pen_flags[i] = True)
       → Proportional controller drives the turtle to the waypoint while
         drawing.  Two phases:
           a. Rotate in place until heading error < ~9°.
           b. Drive forward with proportional angular correction.

3. Because the scan-line planner always places a pen-up waypoint at the
   START of each run and a pen-down waypoint at the END, the canvas receives
   only horizontal strokes corresponding to real edge-pixel runs.  No
   spurious lines appear between rows or between discontinuous runs.

Proportional controller (pen-down phase)
-----------------------------------------
Desired heading:  θ_d  = atan2(dy, dx)
Heading error:    e_θ  = θ_d − θ_current  (normalised to [−π, π])
Angular command:  ω    = Kp_ang · e_θ     (clamped to ±max_omega rad/s)
Linear command:   v    = max(Kp_lin · dist, min_draw_speed)
                         clamped to draw_speed

Using a minimum draw speed ensures that even very short strokes (single
pixels) complete within one or two control ticks rather than crawling.
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
        self.declare_parameter('image_path',   '')
        self.declare_parameter('row_step',     1)       # rows to skip (1=all)
        self.declare_parameter('draw_speed',   3.0)     # m/s while drawing
        self.declare_parameter('sigma',        1.5)
        self.declare_parameter('ksize',        5)
        self.declare_parameter('low_ratio',    0.15)
        self.declare_parameter('high_ratio',   0.35)
        self.declare_parameter('max_dim',      400)
        self.declare_parameter('visualize',    False)

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

        # ---- Path planning (scan-line) ------------------------------
        row_step = self.get_parameter('row_step').get_parameter_value().integer_value
        planner = PathPlanner(edges, row_step=row_step)
        self.waypoints: List[Tuple[float, float]]
        self.pen_flags: List[bool]
        self.waypoints, self.pen_flags = planner.plan()
        self.get_logger().info(
            f'Scan-line waypoints planned: {len(self.waypoints)} '
            f'({sum(self.pen_flags)} draw strokes)'
        )

        self.wp_idx: int = 0

        # ---- ROS interfaces -----------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub    = self.create_subscription(
            Pose, '/turtle1/pose', self._pose_cb, 10
        )
        self.set_pen_cli  = self.create_client(SetPen,           '/turtle1/set_pen')
        self.teleport_cli = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')

        self.pose: Optional[Pose] = None
        self._pen_is_down: bool   = False
        self._skip_ticks: int     = 5   # brief pause for turtlesim to be ready

        for cli, name in [
            (self.set_pen_cli,  '/turtle1/set_pen'),
            (self.teleport_cli, '/turtle1/teleport_absolute'),
        ]:
            while not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'Waiting for service {name} …')

        # Pen up at start — control loop handles the first teleport
        self._set_pen(down=False)

        # ---- Controller parameters ----------------------------------
        self.draw_speed    = self.get_parameter('draw_speed').get_parameter_value().double_value
        self.min_draw_speed = 1.5      # m/s floor so short strokes complete quickly
        self.Kp_linear     = 2.0
        self.Kp_angular    = 4.0
        self.goal_tol      = 0.08      # waypoint reached within this radius (turtlesim units)
        self.max_omega     = 3.0       # rad/s cap

        # ---- Control timer (20 Hz) ----------------------------------
        self.timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info('TurtleController ready — scan-line drawing starts …')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_image_path(self) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(
                get_package_share_directory('turtle_draw'), 'images', 'dog.jpg'
            )
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, '..', 'images', 'dog.jpg')

    def _set_pen(self, down: bool) -> None:
        """Call /turtle1/set_pen — skip if the state is already correct."""
        if down == self._pen_is_down:
            return
        req = SetPen.Request()
        if down:
            req.r, req.g, req.b = 255, 255, 255
            req.width = 2
            req.off   = 0
        else:
            req.off = 1
        self.set_pen_cli.call_async(req)
        self._pen_is_down = down

    def _teleport(self, x: float, y: float, theta: float) -> None:
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
        20 Hz control loop.

        For pen-up waypoints (start of a horizontal run):
          → Set pen off, teleport instantly to the run's start with theta=0
            so the turtle is already facing right for the drawing phase.
            Pause 1 tick (50 ms) for turtlesim to process the teleport.

        For pen-down waypoints (end of a horizontal run):
          → Set pen on, drive forward with the proportional controller.
            Because the turtle was teleported with theta=0 and the target
            is directly to its right (same Y), the rotation phase is
            skipped and the turtle immediately draws the stroke.
        """
        if self._skip_ticks > 0:
            self._skip_ticks -= 1
            return

        if self.pose is None:
            return

        # ---- Done? --------------------------------------------------
        if self.wp_idx >= len(self.waypoints):
            self._set_pen(down=False)
            self.get_logger().info('Drawing complete — all scan lines drawn.')
            self.timer.cancel()
            return

        tx, ty  = self.waypoints[self.wp_idx]
        is_draw = self.pen_flags[self.wp_idx]

        # ---- Pen UP: teleport to start of next run ------------------
        if not is_draw:
            self._set_pen(down=False)
            # theta=0 → turtle faces right, ready to draw the horizontal stroke
            self._teleport(tx, ty, 0.0)
            self.wp_idx     += 1
            self._skip_ticks = 1   # 50 ms for turtlesim to process the teleport
            return

        # ---- Pen DOWN: proportional controller draws the stroke -----
        self._set_pen(down=True)

        dx   = tx - self.pose.x
        dy   = ty - self.pose.y
        dist = math.hypot(dx, dy)

        if dist < self.goal_tol:
            self.wp_idx += 1
            return

        desired_angle = math.atan2(dy, dx)
        angle_err     = desired_angle - self.pose.theta
        angle_err     = (angle_err + math.pi) % (2.0 * math.pi) - math.pi

        cmd = Twist()

        if abs(angle_err) > 0.15:
            # Rotate in place until roughly aligned
            cmd.linear.x  = 0.0
            cmd.angular.z = max(-self.max_omega,
                                min(self.Kp_angular * angle_err, self.max_omega))
        else:
            # Drive forward; enforce minimum speed so short strokes complete fast
            speed = max(self.min_draw_speed,
                        min(self.Kp_linear * dist, self.draw_speed))
            cmd.linear.x  = speed
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
