"""YOLOv8 chess piece detector."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

# Class name mapping
CLASS_NAMES = [
    "white-king", "white-queen", "white-rook", "white-bishop", "white-knight", "white-pawn",
    "black-king", "black-queen", "black-rook", "black-bishop", "black-knight", "black-pawn",
]


@dataclass
class Detection:
    """A single piece detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    center: tuple[float, float]

    @property
    def is_white(self) -> bool:
        return self.class_name.startswith("white")

    @property
    def piece_type(self) -> str:
        """Get piece type (king, queen, etc.) without color."""
        return self.class_name.split("-")[1]


class ChessPieceDetector:
    """YOLOv8-based chess piece detector."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        """Initialize detector.

        Args:
            model_path: Path to trained YOLOv8 model (.pt or .mlpackage)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ("auto", "cpu", "cuda", "mps")
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = YOLO(str(model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect chess pieces in frame.

        Args:
            frame: BGR image (should be warped to bird's-eye view)

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes
        for i in range(len(boxes)):
            # Get box data
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # Calculate center
            x1, y1, x2, y2 = xyxy
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # Get class name
            if cls_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[cls_id]
            else:
                class_name = f"unknown-{cls_id}"

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=center,
            ))

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw detections on frame.

        Args:
            frame: Image to draw on
            detections: List of detections
            show_labels: Show class labels
            show_confidence: Show confidence scores

        Returns:
            Frame with detections drawn
        """
        import cv2

        frame = frame.copy()

        # Colors for white and black pieces
        white_color = (255, 200, 100)  # Light blue
        black_color = (100, 100, 255)  # Light red

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = white_color if det.is_white else black_color

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels:
                label = det.piece_type[0].upper()  # K, Q, R, B, N, P
                if det.piece_type == "knight":
                    label = "N"

                if show_confidence:
                    label = f"{label} {det.confidence:.2f}"

                # Label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_h - 4),
                    (x1 + text_w + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        return frame

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "path": str(self.model_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_names": CLASS_NAMES,
        }


class MockDetector:
    """Mock detector for testing without a trained model."""

    def __init__(self):
        self.confidence_threshold = 0.5

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return empty detections (for pipeline testing)."""
        return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        **kwargs,
    ) -> np.ndarray:
        return frame
