"""Perspective correction for chessboard images."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .aruco import BoardCorners


@dataclass
class BoardCalibration:
    """Calibration data for perspective correction."""

    corners: np.ndarray  # 4x2 array of source corner points
    homography: np.ndarray  # 3x3 homography matrix
    inverse_homography: np.ndarray  # 3x3 inverse homography
    output_size: Tuple[int, int]  # (width, height) of output


class PerspectiveCorrector:
    """Apply perspective correction to warp chessboard to bird's-eye view."""

    def __init__(
        self,
        output_size: Tuple[int, int] = (640, 640),
        margin: int = 0,
    ):
        """Initialize perspective corrector.

        Args:
            output_size: Size of warped output image (width, height)
            margin: Inset margin in pixels (positive = grid inset from edges,
                    negative = grid extends beyond edges). Use this to align
                    the grid with the actual playing area when markers are
                    placed at board corners but slightly outside the squares.
        """
        self.output_size = output_size
        self.margin = margin
        self.calibration: Optional[BoardCalibration] = None

    def calibrate(self, corners: BoardCorners) -> BoardCalibration:
        """Compute homography from detected corners.

        Args:
            corners: Detected board corners [TL, TR, BR, BL]

        Returns:
            BoardCalibration with computed homography
        """
        return self.calibrate_from_points(corners.corners)

    def calibrate_from_points(self, src_points: np.ndarray) -> BoardCalibration:
        """Compute homography from corner points.

        Args:
            src_points: 4x2 array of corner points [TL, TR, BR, BL]

        Returns:
            BoardCalibration with computed homography
        """
        w, h = self.output_size

        # Destination points - perfect square aligned with output
        dst_points = np.array([
            [0, 0],       # Top-left
            [w, 0],       # Top-right
            [w, h],       # Bottom-right
            [0, h],       # Bottom-left
        ], dtype=np.float32)

        src_points = src_points.astype(np.float32)

        # Compute homography
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to compute homography")

        # Compute inverse for mapping back
        H_inv = np.linalg.inv(H)

        self.calibration = BoardCalibration(
            corners=src_points,
            homography=H,
            inverse_homography=H_inv,
            output_size=self.output_size,
        )

        return self.calibration

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp frame to bird's-eye view.

        Args:
            frame: Input BGR image

        Returns:
            Warped image of size output_size
        """
        if self.calibration is None:
            raise ValueError("Must calibrate before warping. Call calibrate() first.")

        return cv2.warpPerspective(
            frame,
            self.calibration.homography,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),  # Gray border
        )

    def warp_point(
        self,
        point: Tuple[float, float],
        inverse: bool = False,
    ) -> Tuple[float, float]:
        """Transform a point between original and warped coordinates.

        Args:
            point: (x, y) coordinates
            inverse: If True, transform from warped to original

        Returns:
            Transformed (x, y) coordinates
        """
        if self.calibration is None:
            raise ValueError("Must calibrate before transforming points.")

        H = (
            self.calibration.inverse_homography if inverse
            else self.calibration.homography
        )

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        return tuple(transformed[0, 0])

    def warp_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        inverse: bool = False,
    ) -> Tuple[float, float, float, float]:
        """Transform a bounding box between coordinate systems.

        Args:
            bbox: (x1, y1, x2, y2) coordinates
            inverse: If True, transform from warped to original

        Returns:
            Transformed (x1, y1, x2, y2) coordinates
        """
        x1, y1, x2, y2 = bbox

        # Transform all four corners
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32).reshape(1, 4, 2)

        H = (
            self.calibration.inverse_homography if inverse
            else self.calibration.homography
        )

        transformed = cv2.perspectiveTransform(corners, H)[0]

        # Get bounding box of transformed corners
        x_coords = transformed[:, 0]
        y_coords = transformed[:, 1]

        return (
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max()),
        )

    def draw_grid(
        self,
        frame: np.ndarray,
        warped: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw 8x8 grid on frame.

        Args:
            frame: Image to draw on
            warped: If True, draw on warped image; if False, draw on original
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Frame with grid drawn
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        if warped:
            # Simple grid on warped image
            cell_w = w / 8
            cell_h = h / 8

            for i in range(9):
                # Vertical lines
                x = int(i * cell_w)
                cv2.line(frame, (x, 0), (x, h), color, thickness)
                # Horizontal lines
                y = int(i * cell_h)
                cv2.line(frame, (0, y), (w, y), color, thickness)
        else:
            # Draw grid on original (perspective) image
            if self.calibration is None:
                return frame

            w_out, h_out = self.output_size
            cell_w = w_out / 8
            cell_h = h_out / 8

            for i in range(9):
                # Vertical lines
                x = i * cell_w
                pt1 = self.warp_point((x, 0), inverse=True)
                pt2 = self.warp_point((x, h_out), inverse=True)
                cv2.line(
                    frame,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    thickness,
                )
                # Horizontal lines
                y = i * cell_h
                pt1 = self.warp_point((0, y), inverse=True)
                pt2 = self.warp_point((w_out, y), inverse=True)
                cv2.line(
                    frame,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    thickness,
                )

        return frame

    def add_square_labels(
        self,
        frame: np.ndarray,
        font_scale: float = 0.4,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Add algebraic notation labels to squares.

        Args:
            frame: Warped image to label
            font_scale: Font size
            color: Text color

        Returns:
            Frame with labels
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        cell_w = w / 8
        cell_h = h / 8

        files = "abcdefgh"
        ranks = "87654321"  # Top to bottom

        for row, rank in enumerate(ranks):
            for col, file in enumerate(files):
                x = int((col + 0.5) * cell_w) - 10
                y = int((row + 0.5) * cell_h) + 5
                label = f"{file}{rank}"
                cv2.putText(
                    frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1
                )

        return frame
