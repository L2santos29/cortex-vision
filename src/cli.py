"""Command-line interface for Cortex-Vision."""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Build the argument parser with detect and serve subcommands."""
    parser = argparse.ArgumentParser(
        prog="cortex-vision",
        description="Object recognition platform powered by YOLOv8",
    )
    subparsers = parser.add_subparsers(dest="command")

    # detect subcommand
    detect_parser = subparsers.add_parser("detect", help="Run detection on an image")
    detect_parser.add_argument("image", help="Path to image file")
    detect_parser.add_argument(
        "--model", default="yolov8n.pt", help="YOLO model name"
    )
    detect_parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold"
    )

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind"
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "detect":
        run_detect(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


def run_detect(args: argparse.Namespace) -> None:
    """Run detection on an image."""
    print(f"Detection on {args.image} with model {args.model}")


def run_serve(args: argparse.Namespace) -> None:
    """Start the API server."""
    print(f"Starting server at {args.host}:{args.port}")


if __name__ == "__main__":
    main()
