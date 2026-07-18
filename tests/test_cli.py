"""Tests for the CLI argument parser."""

from src.cli import create_parser


def test_cli_parser_detect():
    """detect subcommand parses image path and optional flags."""
    parser = create_parser()

    args = parser.parse_args(["detect", "photo.jpg"])
    assert args.command == "detect"
    assert args.image == "photo.jpg"
    assert args.model == "yolov8n.pt"
    assert args.conf == 0.5

    args = parser.parse_args(["detect", "photo.jpg", "--model", "yolov8m.pt", "--conf", "0.8"])
    assert args.model == "yolov8m.pt"
    assert args.conf == 0.8


def test_cli_parser_serve():
    """serve subcommand parses host and port flags."""
    parser = create_parser()

    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8000

    args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert args.host == "127.0.0.1"
    assert args.port == 9000
