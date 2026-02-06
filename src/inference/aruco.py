"""ArUco marker detection for chessboard corner localization."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ArUco dictionary to use - 4x4 markers are small and work well for corners
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Expected marker IDs for each corner (user should print these specific markers)
# Order: top-left (a8), top-right (h8), bottom-right (h1), bottom-left (a1)
CORNER_MARKER_IDS = [0, 1, 2, 3]


@dataclass
class BoardCorners:
    """Detected board corners with confidence."""

    corners: np.ndarray  # 4x2 array of corner points
    marker_ids: list[int]  # IDs of detected markers
    confidence: float  # Detection confidence (0-1)
    all_found: bool  # Whether all 4 corners were found

    @property
    def top_left(self) -> np.ndarray:
        return self.corners[0]

    @property
    def top_right(self) -> np.ndarray:
        return self.corners[1]

    @property
    def bottom_right(self) -> np.ndarray:
        return self.corners[2]

    @property
    def bottom_left(self) -> np.ndarray:
        return self.corners[3]


class ArUcoDetector:
    """Detect ArUco markers for board corner localization."""

    def __init__(
        self,
        dictionary: int = ARUCO_DICT,
        corner_marker_ids: list[int] = CORNER_MARKER_IDS,
    ):
        """Initialize ArUco detector.

        Args:
            dictionary: ArUco dictionary type
            corner_marker_ids: Expected marker IDs for corners [TL, TR, BR, BL]
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.corner_marker_ids = corner_marker_ids

        # Cache for temporal smoothing
        self._last_corners: Optional[BoardCorners] = None
        self._smoothing_alpha = 0.3  # Exponential smoothing factor

    def detect(self, frame: np.ndarray) -> Optional[BoardCorners]:
        """Detect board corners from ArUco markers in frame.

        Args:
            frame: BGR image

        Returns:
            BoardCorners if at least 3 corners found, None otherwise
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Map detected markers to corners
        detected_corners = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.corner_marker_ids:
                # Get center of marker
                marker_corners = corners[i][0]
                center = marker_corners.mean(axis=0)
                corner_idx = self.corner_marker_ids.index(marker_id)
                detected_corners[corner_idx] = center

        if len(detected_corners) < 3:
            return None

        # Build corners array, extrapolating missing corner if needed
        all_corners = np.zeros((4, 2), dtype=np.float32)
        found_indices = list(detected_corners.keys())

        for idx in range(4):
            if idx in detected_corners:
                all_corners[idx] = detected_corners[idx]

        # Extrapolate missing corner if exactly 3 found
        all_found = len(detected_corners) == 4
        if len(detected_corners) == 3:
            missing_idx = [i for i in range(4) if i not in detected_corners][0]
            all_corners[missing_idx] = self._extrapolate_corner(
                all_corners, found_indices, missing_idx
            )

        # Calculate confidence based on detection quality
        confidence = len(detected_corners) / 4.0

        result = BoardCorners(
            corners=all_corners,
            marker_ids=list(detected_corners.keys()),
            confidence=confidence,
            all_found=all_found,
        )

        # Apply temporal smoothing
        if self._last_corners is not None:
            result = self._smooth_corners(result)

        self._last_corners = result
        return result

    def _extrapolate_corner(
        self,
        corners: np.ndarray,
        found_indices: list[int],
        missing_idx: int,
    ) -> np.ndarray:
        """Extrapolate a missing corner from 3 known corners.

        Uses the property that opposite corners of a parallelogram
        sum to the same value: A + C = B + D
        """
        # Corner pairs: (0,2) and (1,3) are opposite
        if missing_idx == 0:
            # A = B + D - C
            return corners[1] + corners[3] - corners[2]
        elif missing_idx == 1:
            # B = A + C - D
            return corners[0] + corners[2] - corners[3]
        elif missing_idx == 2:
            # C = A + B - D... wait, that's wrong
            # C = B + D - A
            return corners[1] + corners[3] - corners[0]
        else:  # missing_idx == 3
            # D = A + C - B
            return corners[0] + corners[2] - corners[1]

    def _smooth_corners(self, current: BoardCorners) -> BoardCorners:
        """Apply exponential smoothing to corners for stability."""
        if self._last_corners is None:
            return current

        alpha = self._smoothing_alpha
        smoothed = (
            alpha * current.corners + (1 - alpha) * self._last_corners.corners
        )

        return BoardCorners(
            corners=smoothed,
            marker_ids=current.marker_ids,
            confidence=current.confidence,
            all_found=current.all_found,
        )

    def draw_markers(
        self,
        frame: np.ndarray,
        corners: Optional[BoardCorners] = None,
    ) -> np.ndarray:
        """Draw detected markers and board outline on frame.

        Args:
            frame: BGR image to draw on
            corners: Detected corners (or use last detection)

        Returns:
            Frame with markers drawn
        """
        if corners is None:
            corners = self._last_corners

        if corners is None:
            return frame

        frame = frame.copy()

        # Draw corner points
        colors = [
            (0, 255, 0),    # TL - green
            (255, 0, 0),    # TR - blue
            (0, 0, 255),    # BR - red
            (255, 255, 0),  # BL - cyan
        ]
        labels = ["a8", "h8", "h1", "a1"]

        for i, (pt, color, label) in enumerate(zip(corners.corners, colors, labels)):
            pt = tuple(pt.astype(int))
            cv2.circle(frame, pt, 10, color, -1)
            cv2.putText(
                frame, label, (pt[0] + 15, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # Draw board outline
        pts = corners.corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Draw confidence
        cv2.putText(
            frame,
            f"Board conf: {corners.confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame


def generate_aruco_markers(
    output_dir: str,
    marker_size: int = 200,
    dictionary: int = ARUCO_DICT,
    marker_ids: list[int] = CORNER_MARKER_IDS,
) -> None:
    """Generate ArUco marker images for printing.

    Args:
        output_dir: Directory to save marker images
        marker_size: Size of marker in pixels
        dictionary: ArUco dictionary type
        marker_ids: Marker IDs to generate
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    corner_names = ["top_left_a8", "top_right_h8", "bottom_right_h1", "bottom_left_a1"]

    for marker_id, name in zip(marker_ids, corner_names):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # Add white border for easier cutting
        border_size = 20
        bordered = cv2.copyMakeBorder(
            marker_img,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        # Add label
        cv2.putText(
            bordered,
            f"ID: {marker_id} ({name})",
            (10, bordered.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            0,
            1,
        )

        output_path = os.path.join(output_dir, f"marker_{marker_id}_{name}.png")
        cv2.imwrite(output_path, bordered)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Generate markers for printing
    generate_aruco_markers("aruco_markers")
