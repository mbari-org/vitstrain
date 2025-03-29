# vitstrain
# Filename: src/args.py
# Description: Argument parser for training a Vision Transformer model
import argparse
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data"

def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification Vision Transformer (ViTS) model.")
    parser.add_argument(
        "--remove-long-tail",
        type=bool,
        default=False,
        help="Set to true to truncate the long-tail classes.",
    )
    parser.add_argument(
        "--remap",
        type=str,
        help="Path to a JSON file that maps the original class names to new class names.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="catsdogs-vit-b-16",
        help="Name of the model you want to train.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/vit-base-patch16-224",
        help="Name of the base model to fine-tune from, e.g. google/vit-base-patch16-224 or facebook/dino-vitb8.",
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        nargs="+",
        required=False,
        help="Paths to the raw dataset (space-separated if multiple paths).",
    )
    parser.add_argument(
        "--filter-data",
        type=str,
        default=str(Path(__file__).parent.parent / "data_filter"),
        help="Path to store the filtered dataset.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--add-rotations",
        type=bool,
        default=True,
        help="Set to true to add 90, 180, 270 rotations to the training images.",
    )
    parser.add_argument(
        "--early-stopping-epochs",
        type=int,
        default=2,
        help="Number of epochs to wait for early stopping.",
    )
    return parser.parse_args()
