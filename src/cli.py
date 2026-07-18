"""Cortex-Vision CLI — command-line interface for object detection.

Usage:
    cortex-vision detect <image> [options]
    cortex-vision process <images...> [options]
    cortex-vision video <file> [options]
    cortex-vision serve [options]
"""

import argparse
import json
import sys
from pathlib import Path

from .detector import Detector
from .pipeline import BatchPipeline
from .utils import process_video_frames


# Shared arguments for detection subcommands
_SHARED_ARGS = [
    (["--model"], {"default": "yolov8n.pt", "help": "YOLO model variant"}),
    (["--threshold"], {"type": float, "default": 0.0, "help": "Min confidence (default: 0.0)"}),
    (["--output"], {"choices": ["json", "table"], "default": "json", "help": "Output format"}),
    (["--save"], {"default": None, "help": "Directory for annotated images"}),
]


def _add_shared(sub):
    for args, kwargs in _SHARED_ARGS:
        sub.add_argument(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        prog="cortex-vision",
        description="Object recognition powered by YOLOv8",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect
    p = subparsers.add_parser("detect", help="Detect in a single image")
    _add_shared(p)
    p.add_argument("image", help="Path to image file")

    # process
    p = subparsers.add_parser("process", help="Batch process multiple images")
    _add_shared(p)
    p.add_argument("images", nargs="+", help="Image file paths")

    # video
    p = subparsers.add_parser("video", help="Analyze a video frame by frame")
    _add_shared(p)
    p.add_argument("file", help="Path to video file")
    p.add_argument("--fps", type=int, default=1, help="Frames per second to sample")

    # serve
    p = subparsers.add_parser("serve", help="Start FastAPI web server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "serve":
        _run_server(args)
    else:
        _run_detection(args)


def _run_detection(args):
    detector = Detector(model_name=args.model)

    if args.command == "detect":
        results = detector.detect(args.image)
        _output_results(results, args, source=args.image)
        if args.save:
            _save_annotated(args.image, results, args.save)

    elif args.command == "process":
        pipeline = BatchPipeline(detector)
        results = pipeline.process(args.images)
        aggregated = pipeline.aggregate(results)
        _output_results(aggregated, args, source=f"{len(args.images)} images")
        if args.save:
            for r in results:
                _save_annotated(r["image"], r["detections"], args.save)

    elif args.command == "video":
        print(f"Processing video at {args.fps} FPS...", file=sys.stderr)
        frames = process_video_frames(args.file, detector, fps_sample=args.fps)
        _output_video_results(frames, args)
        if args.save:
            _save_video_frames(frames, args.save)


def _output_results(detections, args, source=""):
    filtered = [d for d in detections if d["confidence"] >= args.threshold]
    name = Path(source).name if source else "results"

    if args.output == "json":
        print(json.dumps(filtered, indent=2))
    else:
        print(f"\n{'─' * 50}")
        print(f"  {name}: {len(filtered)} object(s)")
        print(f"{'─' * 50}")
        for d in filtered:
            print(f"  {d['class']:20s} {d['confidence']:.1%}")


def _output_video_results(frames, args):
    total = sum(f["object_count"] for f in frames)
    hits = sum(1 for f in frames if f["object_count"] > 0)

    if args.output == "json":
        out = [{k: v for k, v in f.items() if k != "annotated_frame"} for f in frames]
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{'─' * 50}")
        print(f"  {len(frames)} frames, {hits} with objects, {total} total")
        print(f"{'─' * 50}")
        for f in frames:
            icon = "+" if f["object_count"] > 0 else " "
            top = ", ".join(d["class"] for d in f["detections"][:4])
            if len(f["detections"]) > 4:
                top += " ..."
            print(f"  [{icon}] @{f['timestamp']:4.1f}s  ({f['object_count']:2d})  {top}")


def _save_annotated(image_path, detections, save_dir):
    import cv2
    from .utils import draw_boxes_on_frame
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is not None:
        annotated = draw_boxes_on_frame(img, detections)
        out = save_path / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(out), annotated)
        print(f"  Saved: {out}", file=sys.stderr)


def _save_video_frames(frames, save_dir):
    import base64
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in frames:
        b64 = f.get("annotated_frame", "")
        if b64:
            data = base64.b64decode(b64.split(",", 1)[-1])
            (save_path / f"frame_{f['frame_number']:04d}.jpg").write_bytes(data)
            n += 1
    print(f"  Saved {n} frame(s) to {save_path}", file=sys.stderr)


def _run_server(args):
    import uvicorn
    print(f"Starting Cortex-Vision at http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run("src.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
