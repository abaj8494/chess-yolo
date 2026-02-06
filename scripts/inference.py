#!/usr/bin/env python3
"""Inference CLI for Chess YOLO."""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def run_video(args):
    """Process a pre-recorded video."""
    from src.inference.detector import ChessPieceDetector
    from src.pipeline.video_pipeline import VideoPipeline, ProcessingConfig

    console.print(f"\n[blue]Processing video: {args.input}")

    # Load detector
    detector = ChessPieceDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Configure processing
    config = ProcessingConfig(
        skip_frames=args.skip_frames,
        stability_threshold=args.stability,
        save_visualization=args.save_viz,
        visualization_path=args.viz_output,
    )

    # Create pipeline
    pipeline = VideoPipeline(detector=detector, config=config)

    # Process
    result = pipeline.process(
        video_path=args.input,
        output_pgn=args.output,
    )

    # Print PGN
    console.print("\n[bold]Generated PGN:[/bold]")
    console.print(result.pgn)


def run_live(args):
    """Run live camera processing."""
    from src.inference.detector import ChessPieceDetector
    from src.pipeline.live_pipeline import LivePipeline, LiveConfig

    console.print("\n[blue]Starting live capture...")

    # Load detector
    detector = ChessPieceDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Configure
    config = LiveConfig(
        camera_index=args.camera,
        output_pgn_path=args.output,
        white_player=args.white,
        black_player=args.black,
        event_name=args.event,
    )

    # Create and run pipeline
    pipeline = LivePipeline(detector=detector, config=config)

    try:
        pipeline.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user")


def run_test(args):
    """Test detector on a single image."""
    import cv2
    from src.inference.detector import ChessPieceDetector
    from src.inference.aruco import ArUcoDetector
    from src.inference.perspective import PerspectiveCorrector

    console.print(f"\n[blue]Testing on image: {args.input}")

    # Load image
    img = cv2.imread(str(args.input))
    if img is None:
        console.print(f"[red]Could not load image: {args.input}")
        return

    # Detect ArUco markers
    aruco = ArUcoDetector()
    corners = aruco.detect(img)

    if corners is None:
        console.print("[yellow]No ArUco markers detected")
        console.print("[yellow]Showing original image...")
        cv2.imshow("Test", img)
        cv2.waitKey(0)
        return

    console.print(f"[green]Detected {len(corners.marker_ids)} markers")

    # Warp
    perspective = PerspectiveCorrector()
    perspective.calibrate(corners)
    warped = perspective.warp_frame(img)

    # Detect pieces
    detector = ChessPieceDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    detections = detector.detect(warped)
    console.print(f"[green]Detected {len(detections)} pieces")

    for det in detections:
        console.print(f"  {det.class_name}: {det.confidence:.2f}")

    # Visualize
    viz = detector.draw_detections(warped, detections)
    viz = perspective.draw_grid(viz, warped=True)

    # Show
    cv2.imshow("Detections", viz)
    console.print("\n[yellow]Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if args.output:
        cv2.imwrite(str(args.output), viz)
        console.print(f"[green]Saved to: {args.output}")


def generate_markers(args):
    """Generate ArUco markers for printing."""
    from src.inference.aruco import generate_aruco_markers

    output_dir = args.output or "aruco_markers"
    generate_aruco_markers(output_dir, marker_size=args.size)
    console.print(f"[green]Markers saved to: {output_dir}/")
    console.print("[yellow]Print these markers and place at board corners:")
    console.print("  - marker_0_top_left_a8.png → Top-left corner (a8)")
    console.print("  - marker_1_top_right_h8.png → Top-right corner (h8)")
    console.print("  - marker_2_bottom_right_h1.png → Bottom-right corner (h1)")
    console.print("  - marker_3_bottom_left_a1.png → Bottom-left corner (a1)")


def main():
    parser = argparse.ArgumentParser(
        description="Chess YOLO Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Video command
    video_parser = subparsers.add_parser("video", help="Process a video file")
    video_parser.add_argument("input", type=Path, help="Input video path")
    video_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output PGN path"
    )
    video_parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to trained model"
    )
    video_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    video_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )
    video_parser.add_argument(
        "--skip-frames", type=int, default=1, help="Process every Nth frame"
    )
    video_parser.add_argument(
        "--stability", type=int, default=5, help="Frames for stable state"
    )
    video_parser.add_argument(
        "--save-viz", action="store_true", help="Save visualization video"
    )
    video_parser.add_argument(
        "--viz-output", type=str, default=None, help="Visualization output path"
    )
    video_parser.set_defaults(func=run_video)

    # Live command
    live_parser = subparsers.add_parser("live", help="Live camera processing")
    live_parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to trained model"
    )
    live_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output PGN path"
    )
    live_parser.add_argument(
        "--camera", type=int, default=0, help="Camera index"
    )
    live_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    live_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )
    live_parser.add_argument(
        "--white", type=str, default="White", help="White player name"
    )
    live_parser.add_argument(
        "--black", type=str, default="Black", help="Black player name"
    )
    live_parser.add_argument(
        "--event", type=str, default="Live Game", help="Event name"
    )
    live_parser.set_defaults(func=run_live)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test on single image")
    test_parser.add_argument("input", type=Path, help="Input image path")
    test_parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to trained model"
    )
    test_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output image path"
    )
    test_parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    test_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )
    test_parser.set_defaults(func=run_test)

    # Markers command
    markers_parser = subparsers.add_parser("markers", help="Generate ArUco markers")
    markers_parser.add_argument(
        "-o", "--output", type=str, default="aruco_markers", help="Output directory"
    )
    markers_parser.add_argument(
        "--size", type=int, default=200, help="Marker size in pixels"
    )
    markers_parser.set_defaults(func=generate_markers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
