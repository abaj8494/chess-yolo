#!/usr/bin/env python3
"""Generate printable ArUco markers for chess board corner detection."""

import cv2
import numpy as np
from pathlib import Path


def generate_marker_sheet(output_path: str = "aruco_markers.png", marker_size: int = 200):
    """
    Generate a printable sheet with 4 ArUco markers for board corners.

    Args:
        output_path: Where to save the marker sheet
        marker_size: Size of each marker in pixels
    """
    # Use DICT_4X4_50 - same as in src/inference/aruco.py
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Marker IDs for each corner (matching aruco.py)
    # 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    marker_ids = [0, 1, 2, 3]
    corner_names = ["TOP-LEFT (A8)", "TOP-RIGHT (H8)", "BOTTOM-RIGHT (H1)", "BOTTOM-LEFT (A1)"]

    # Generate individual markers
    markers = []
    for marker_id in marker_ids:
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        markers.append(marker)

    # Create a sheet layout (2x2 grid with labels and spacing)
    padding = 60
    label_height = 80
    cell_size = marker_size + padding * 2
    sheet_width = cell_size * 2 + 100
    sheet_height = cell_size * 2 + label_height * 2 + 150

    # White background
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255

    # Convert to BGR for colored text
    sheet_color = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)

    # Title
    cv2.putText(sheet_color, "Chess Board ArUco Markers", (sheet_width // 2 - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(sheet_color, "Cut out and place at board corners", (sheet_width // 2 - 200, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    # Place markers in 2x2 grid
    positions = [
        (50, 120),  # Top-left marker
        (cell_size + 50, 120),  # Top-right marker
        (cell_size + 50, cell_size + label_height + 120),  # Bottom-right marker
        (50, cell_size + label_height + 120),  # Bottom-left marker
    ]

    for i, (marker, pos, name) in enumerate(zip(markers, positions, corner_names)):
        x, y = pos

        # Draw marker border
        cv2.rectangle(sheet_color, (x - 5, y - 5),
                     (x + marker_size + 5, y + marker_size + 5), (0, 0, 0), 2)

        # Place marker
        sheet_color[y:y + marker_size, x:x + marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        # Add label below marker
        label = f"ID {marker_ids[i]}: {name}"
        cv2.putText(sheet_color, label, (x, y + marker_size + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Instructions at bottom
    instructions = [
        "INSTRUCTIONS:",
        "1. Print this page at 100% scale (no scaling)",
        "2. Cut out each marker with ~1cm white border",
        "3. Place markers at board corners as labeled",
        "4. Markers should be flat and visible to camera",
        "5. TOP-LEFT = A8 corner (black's left rook)",
        "6. BOTTOM-RIGHT = H1 corner (white's right rook)",
    ]

    y_offset = sheet_height - 140
    for instruction in instructions:
        cv2.putText(sheet_color, instruction, (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20

    # Save
    cv2.imwrite(output_path, sheet_color)
    print(f"Marker sheet saved to: {output_path}")

    return output_path


def generate_individual_markers(output_dir: str = "markers", marker_size: int = 300):
    """Generate individual marker images (larger, for separate printing)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corner_names = ["top_left_A8", "top_right_H8", "bottom_right_H1", "bottom_left_A1"]

    for marker_id, name in enumerate(corner_names):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # Add white border
        bordered = np.ones((marker_size + 100, marker_size + 100), dtype=np.uint8) * 255
        bordered[50:50 + marker_size, 50:50 + marker_size] = marker

        filepath = output_path / f"marker_{marker_id}_{name}.png"
        cv2.imwrite(str(filepath), bordered)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ArUco markers for chess board")
    parser.add_argument("--output", "-o", default="aruco_markers.png", help="Output file path")
    parser.add_argument("--size", "-s", type=int, default=200, help="Marker size in pixels")
    parser.add_argument("--individual", "-i", action="store_true", help="Generate individual marker files")

    args = parser.parse_args()

    if args.individual:
        generate_individual_markers(marker_size=args.size)
    else:
        generate_marker_sheet(args.output, args.size)
