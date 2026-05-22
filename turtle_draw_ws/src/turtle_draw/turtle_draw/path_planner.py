"""
path_planner.py — Map edge pixels to an ordered list of turtlesim waypoints.

Coordinate systems
------------------
Image space   : origin top-left, row axis points DOWN, col axis points RIGHT.
                np.argwhere returns [row, col] pairs.
Turtlesim     : origin bottom-left, X axis points RIGHT, Y axis points UP.
                Valid range ≈ [0.5, 10.5] × [0.5, 10.5].

Mapping
-------
  tx = TMIN + (col / img_w) * TRANGE        # col → x  (same direction)
  ty = TMAX - (row / img_h) * TRANGE        # row → y  (flip: ↓ becomes ↑)

Path ordering
-------------
Raw edge pixels are scattered across the image in raster order.  Connecting
them naively would cause the turtle to zig-zag randomly, lifting the pen
thousands of times and producing an incoherent drawing.

A Greedy Nearest-Neighbour (NN) tour minimises total travel distance with
O(N²) time complexity — acceptable for N ≤ 1 000 points with NumPy's
vectorised operations (no Python loops over individual distances).

At each step we compute the squared Euclidean distance from the current
point to ALL remaining candidates using a vectorised subtraction, then pick
the minimum.  Using squared distance avoids the sqrt and is monotonically
equivalent for comparison purposes.
"""

import numpy as np

TURTLESIM_MIN = 0.5
TURTLESIM_MAX = 10.5
TURTLESIM_RANGE = TURTLESIM_MAX - TURTLESIM_MIN   # 10.0 units


class PathPlanner:

    def __init__(self, edge_map: np.ndarray,
                 max_points: int = 500,
                 jump_threshold: float = 0.8):
        """
        Parameters
        ----------
        edge_map        : binary uint8 array — 255 where an edge pixel exists.
        max_points      : cap on waypoints to keep drawing time under ~2 min.
        jump_threshold  : turtlesim-space distance above which the pen is
                          lifted to avoid drawing phantom connecting lines.
        """
        self.edge_map        = edge_map
        self.max_points      = max_points
        self.jump_threshold  = jump_threshold
        self.img_h, self.img_w = edge_map.shape

    # ------------------------------------------------------------------ #

    def extract_points(self) -> np.ndarray:
        """
        Return all edge pixels as an (N, 2) integer array of [row, col] pairs.
        np.argwhere scans in raster order (top-left to bottom-right).
        """
        return np.argwhere(self.edge_map > 0)

    def subsample(self, points: np.ndarray) -> np.ndarray:
        """
        Reduce the point cloud to at most max_points using uniform stride.

        Stride-based selection rather than random sampling preserves the
        spatial distribution of the original edge map: consecutive indices
        from np.argwhere are nearby in raster order, so taking every k-th
        point retains roughly one representative per small neighbourhood.
        """
        if len(points) <= self.max_points:
            return points
        step = max(1, len(points) // self.max_points)
        return points[::step][: self.max_points]

    def nearest_neighbor_sort(self, points: np.ndarray) -> np.ndarray:
        """
        Greedy nearest-neighbour reordering of the waypoint set.

        Algorithm
        ---------
        1. Start at index 0.
        2. For each remaining step, compute squared L2 distance from the
           current point to every unvisited point (vectorised NumPy subtract
           + element-wise multiply + sum along axis 1).
        3. Select the minimum → mark visited → advance.

        Complexity: O(N²) NumPy operations — for N = 500 this is ~250 k
        element-level ops, completing in milliseconds.

        Using squared distance (no sqrt) is valid because sqrt is monotone:
        argmin(d²) == argmin(d).
        """
        n = len(points)
        if n == 0:
            return points

        pts = points.astype(np.float64)

        visited  = np.zeros(n, dtype=bool)
        order    = np.empty(n, dtype=np.int64)
        current  = 0
        visited[0] = True
        order[0]   = 0

        for step in range(1, n):
            p     = pts[current]
            diffs = pts - p                            # (N, 2)
            dists = (diffs * diffs).sum(axis=1)        # squared L2, shape (N,)
            dists[visited] = np.inf                    # exclude visited
            nxt   = int(np.argmin(dists))
            order[step]  = nxt
            visited[nxt] = True
            current      = nxt

        return points[order]

    def image_to_turtlesim(self, row: float, col: float):
        """
        Convert a pixel coordinate (row, col) to turtlesim (x, y).

        The image row axis points downward while turtlesim's Y axis points
        upward, so we subtract the normalised row from TMAX to flip it.
        Both axes are scaled linearly to fill [TMIN, TMAX].
        """
        tx = TURTLESIM_MIN + (col / self.img_w) * TURTLESIM_RANGE
        ty = TURTLESIM_MAX - (row / self.img_h) * TURTLESIM_RANGE
        return float(tx), float(ty)

    # ------------------------------------------------------------------ #

    def plan(self):
        """
        Run the full planning pipeline.

        Returns
        -------
        waypoints : list[tuple[float, float]]   (x, y) turtlesim coordinates
        pen_down  : list[bool]                  True  → draw; False → jump
        """
        raw    = self.extract_points()
        if len(raw) == 0:
            raise RuntimeError(
                "No edge pixels found — verify the CV pipeline parameters."
            )

        sampled  = self.subsample(raw)
        ordered  = self.nearest_neighbor_sort(sampled)
        waypoints = [self.image_to_turtlesim(r, c) for r, c in ordered]

        # Mark pen-up segments: large spatial jumps between consecutive points
        pen_down = [False] * len(waypoints)   # first move always pen-up
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i - 1][0]
            dy = waypoints[i][1] - waypoints[i - 1][1]
            dist = (dx * dx + dy * dy) ** 0.5
            pen_down[i] = dist <= self.jump_threshold

        return waypoints, pen_down