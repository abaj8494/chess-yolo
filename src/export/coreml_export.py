"""Export YOLOv8 models to CoreML for Apple Silicon inference."""

from pathlib import Path
from typing import Optional

from rich.console import Console
from ultralytics import YOLO

console = Console()


def export_to_coreml(
    model_path: str | Path,
    output_dir: Optional[str | Path] = None,
    half: bool = True,
    int8: bool = False,
    nms: bool = True,
    imgsz: int = 640,
) -> Path:
    """Export YOLOv8 model to CoreML format.

    Args:
        model_path: Path to trained YOLOv8 model (.pt)
        output_dir: Output directory (defaults to same as model)
        half: Use FP16 quantization (recommended for Apple Silicon)
        int8: Use INT8 quantization (more aggressive, may reduce accuracy)
        nms: Include NMS in the model
        imgsz: Input image size

    Returns:
        Path to exported CoreML model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    console.print(f"[blue]Loading model: {model_path}")
    model = YOLO(str(model_path))

    console.print("[blue]Exporting to CoreML...")
    console.print(f"  FP16: {half}")
    console.print(f"  INT8: {int8}")
    console.print(f"  NMS: {nms}")
    console.print(f"  Size: {imgsz}x{imgsz}")

    # Export
    export_path = model.export(
        format="coreml",
        half=half,
        int8=int8,
        nms=nms,
        imgsz=imgsz,
    )

    export_path = Path(export_path)

    # Move to output dir if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / export_path.name
        export_path.rename(new_path)
        export_path = new_path

    console.print(f"[green]Exported to: {export_path}")
    console.print(f"[green]Size: {export_path.stat().st_size / 1024 / 1024:.1f} MB")

    return export_path


def export_to_onnx(
    model_path: str | Path,
    output_dir: Optional[str | Path] = None,
    opset: int = 17,
    half: bool = False,
    simplify: bool = True,
    imgsz: int = 640,
) -> Path:
    """Export YOLOv8 model to ONNX format.

    Args:
        model_path: Path to trained YOLOv8 model (.pt)
        output_dir: Output directory
        opset: ONNX opset version
        half: Use FP16
        simplify: Simplify ONNX model
        imgsz: Input image size

    Returns:
        Path to exported ONNX model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    console.print(f"[blue]Loading model: {model_path}")
    model = YOLO(str(model_path))

    console.print("[blue]Exporting to ONNX...")

    export_path = model.export(
        format="onnx",
        opset=opset,
        half=half,
        simplify=simplify,
        imgsz=imgsz,
    )

    export_path = Path(export_path)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / export_path.name
        export_path.rename(new_path)
        export_path = new_path

    console.print(f"[green]Exported to: {export_path}")

    return export_path


def verify_coreml_export(model_path: str | Path) -> bool:
    """Verify CoreML export by running a test inference.

    Args:
        model_path: Path to CoreML model (.mlpackage or .mlmodel)

    Returns:
        True if verification successful
    """
    import numpy as np

    try:
        import coremltools as ct

        console.print(f"[blue]Loading CoreML model: {model_path}")
        model = ct.models.MLModel(str(model_path))

        # Get input spec
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        console.print(f"[blue]Input name: {input_name}")

        # Create dummy input
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        console.print("[blue]Running test inference...")
        _ = model.predict({input_name: dummy_input})

        console.print("[green]Verification successful!")
        return True

    except Exception as e:
        console.print(f"[red]Verification failed: {e}")
        return False


def benchmark_coreml(
    model_path: str | Path,
    num_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark CoreML model inference speed.

    Args:
        model_path: Path to CoreML model
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    import time
    import numpy as np

    try:
        import coremltools as ct

        console.print(f"[blue]Loading model for benchmark...")
        model = ct.models.MLModel(str(model_path))

        spec = model.get_spec()
        input_name = spec.description.input[0].name

        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        # Warmup
        console.print(f"[blue]Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            model.predict({input_name: dummy_input})

        # Benchmark
        console.print(f"[blue]Benchmarking ({num_iterations} iterations)...")
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            model.predict({input_name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)

        results = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times),
        }

        console.print(f"\n[green]Benchmark Results:")
        console.print(f"  Mean: {results['mean_ms']:.2f} ms")
        console.print(f"  Std: {results['std_ms']:.2f} ms")
        console.print(f"  Min: {results['min_ms']:.2f} ms")
        console.print(f"  Max: {results['max_ms']:.2f} ms")
        console.print(f"  FPS: {results['fps']:.1f}")

        return results

    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}")
        return {}
