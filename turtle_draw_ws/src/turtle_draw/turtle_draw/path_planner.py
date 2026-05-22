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

Scan-line path ordering
-----------------------
The edge map is traversed row by row (top to bottom), left to right within
each row — exactly like a printer rastering a page.

For each row we identify "runs": groups of consecutive edge-pixel columns.
Each run becomes two waypoints:

  1. pen_down = False  →  start of the run
     The controller teleports the turtle here instantly (no trail left).

  2. pen_down = True   →  end of the run
     The controller drives the turtle forward with the pen down, drawing
     the horizontal stroke.

Gaps between runs (non-edge pixels) and the vertical jumps between rows are
handled by the pen-up teleport, so no phantom lines appear on the canvas.

The row_step parameter lets the caller skip rows to trade detail for speed:
  row_step=1  → every row (maximum detail)
  row_step=2  → every other row (half the waypoints, twice as fast)
"""

import numpy as np

TURTLESIM_MIN = 0.5
TURTLESIM_MAX = 10.5
TURTLESIM_RANGE = TURTLESIM_MAX - TURTLESIM_MIN   # 10.0 units


class PathPlanner:

    def __init__(self, edge_map: np.ndarray, row_step: int = 1):
        """
        Parameters
        ----------
        edge_map  : binary uint8 array — 255 where an edge pixel exists.
        row_step  : process every N-th row (1 = all rows, 2 = every other, …).
                    Increasing this value reduces total waypoints and
                    drawing time at the cost of vertical resolution.
        """
        self.edge_map = edge_map
        self.row_step = max(1, int(row_step))
        self.img_h, self.img_w = edge_map.shape

    # ------------------------------------------------------------------ #

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

    @staticmethod
    def _group_runs(cols: np.ndarray):
        """
        Partition a sorted array of column indices into contiguous runs.

        A new run begins whenever the gap between consecutive columns > 1.
        Returns a list of (start_col, end_col) pairs (both inclusive).

        Example
        -------
          cols = [2, 3, 4, 7, 8, 12]  →  [(2, 4), (7, 8), (12, 12)]

        A single-pixel run has start_col == end_col; the draw waypoint
        coincides with the teleport destination and the controller advances
        immediately (dist = 0 < goal_tol).
        """
        runs = []
        start = int(cols[0])
        prev  = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c - prev > 1:          # gap detected → close current run
                runs.append((start, prev))
                start = c
            prev = c
        runs.append((start, prev))    # close the last run
        return runs

    # ------------------------------------------------------------------ #

    def plan(self):
        """
        Build a scan-line waypoint list for the full edge map.

        Traverses every row_step-th row top to bottom.  Within each row,
        contiguous runs of edge pixels become horizontal draw strokes.
        Each stroke yields one pen-up waypoint (run start, teleport) and
        one pen-down waypoint (run end, drawn by the proportional
        controller).

        Returns
        -------
        waypoints : list[tuple[float, float]]
            (x, y) positions in turtlesim coordinate space.
        pen_down  : list[bool]
            Parallel list of pen states.
            False → pen up   (controller teleports to this point)
            True  → pen down (controller drives forward, drawing)
        """
        waypoints = []
        pen_down  = []

        for row in range(0, self.img_h, self.row_step):
            # Columns where an edge pixel exists in this row
            edge_cols = np.where(self.edge_map[row] > 0)[0]
            if len(edge_cols) == 0:
                continue

            for run_start, run_end in self._group_runs(edge_cols):
                # --- pen up: teleport to start of run ---
                x, y = self.image_to_turtlesim(row, run_start)
                waypoints.append((x, y))
                pen_down.append(False)

                # --- pen down: drive to end of run (draws the stroke) ---
                x, y = self.image_to_turtlesim(row, run_end)
                waypoints.append((x, y))
                pen_down.append(True)

        if not waypoints:
            raise RuntimeError(
                "No edge pixels found — verify the CV pipeline parameters."
            )

        return waypoints, pen_down
