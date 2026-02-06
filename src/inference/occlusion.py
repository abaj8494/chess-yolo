"""Occlusion detection and handling for chess piece detection."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .detector import Detection


@dataclass
class OcclusionRegion:
    """A detected occlusion region (e.g., player's hand)."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    area: float


class OcclusionDetector:
    """Detect occlusions (hands) that may interfere with piece detection."""

    def __init__(
        self,
        min_skin_area: float = 5000,
        skin_lower_hsv: tuple[int, int, int] = (0, 20, 70),
        skin_upper_hsv: tuple[int, int, int] = (20, 255, 255),
    ):
        """Initialize occlusion detector.

        Args:
            min_skin_area: Minimum contour area to consider as hand
            skin_lower_hsv: Lower HSV bound for skin detection
            skin_upper_hsv: Upper HSV bound for skin detection
        """
        self.min_skin_area = min_skin_area
        self.skin_lower = np.array(skin_lower_hsv, dtype=np.uint8)
        self.skin_upper = np.array(skin_upper_hsv, dtype=np.uint8)

    def detect_hands(self, frame: np.ndarray) -> list[OcclusionRegion]:
        """Detect hands/skin regions in frame.

        Args:
            frame: BGR image

        Returns:
            List of detected occlusion regions
        """
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create skin mask
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_skin_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append(OcclusionRegion(
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=min(1.0, area / (self.min_skin_area * 10)),
                    area=area,
                ))

        return regions

    def filter_occluded_detections(
        self,
        detections: list[Detection],
        occlusions: list[OcclusionRegion],
        iou_threshold: float = 0.3,
    ) -> tuple[list[Detection], list[Detection]]:
        """Filter out detections that overlap significantly with occlusions.

        Args:
            detections: Piece detections
            occlusions: Detected occlusion regions
            iou_threshold: IoU threshold for filtering

        Returns:
            (clean_detections, occluded_detections) tuple
        """
        if not occlusions:
            return detections, []

        clean = []
        occluded = []

        for det in detections:
            max_iou = 0.0
            for occ in occlusions:
                iou = self._compute_iou(det.bbox, occ.bbox)
                max_iou = max(max_iou, iou)

            if max_iou < iou_threshold:
                clean.append(det)
            else:
                occluded.append(det)

        return clean, occluded

    def _compute_iou(
        self,
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def draw_occlusions(
        self,
        frame: np.ndarray,
        occlusions: list[OcclusionRegion],
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """Draw occlusion regions on frame.

        Args:
            frame: Image to draw on
            occlusions: Detected occlusions
            color: Box color (BGR)

        Returns:
            Frame with occlusions drawn
        """
        frame = frame.copy()

        for occ in occlusions:
            x1, y1, x2, y2 = [int(v) for v in occ.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"HAND {occ.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return frame


class MotionDetector:
    """Detect motion to identify when moves are being made."""

    def __init__(self, threshold: float = 25.0, min_area: float = 500):
        """Initialize motion detector.

        Args:
            threshold: Difference threshold for motion detection
            min_area: Minimum contour area to consider as motion
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame: Optional[np.ndarray] = None

    def detect_motion(self, frame: np.ndarray) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """Detect motion between current and previous frame.

        Args:
            frame: Current BGR frame

        Returns:
            (has_motion, motion_regions) tuple
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, []

        # Compute difference
        diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, x + w, y + h))

        self.prev_frame = gray

        return len(motion_regions) > 0, motion_regions

    def reset(self):
        """Reset motion detector state."""
        self.prev_frame = None
