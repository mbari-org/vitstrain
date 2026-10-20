# vitstrain
# Filename: src/args.py
# Description: Argument parser for training a Vision Transformer model
import argparse
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data"

# Short aliases for base models known to work with this pipeline. Pass either
# an alias (e.g. "dinov3-vits16") or the full Hugging Face model id
# (e.g. "facebook/dino-vits8") to --base-model.
MODEL_PRESETS = {
    "vit-b16": "google/vit-base-patch16-224",
    "vit-b16-in21k": "google/vit-base-patch16-224-in21k",
    "clip-vit-b32": "openai/clip-vit-base-patch32",
    "dino-vits8": "facebook/dino-vits8",
    "dino-vits16": "facebook/dino-vits16",
    "dino-vitb8": "facebook/dino-vitb8",
    "dino-vitb16": "facebook/dino-vitb16",
    "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "dinov3-convnext-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnextv2-tiny": "facebook/convnextv2-tiny-22k-224",
    "convnextv2-base": "facebook/convnextv2-base-22k-224",
    "efficientnet-b0": "google/efficientnet-b0",
    "efficientnet-b1": "google/efficientnet-b1",
    "efficientnet-b2": "google/efficientnet-b2",
    "efficientnet-b3": "google/efficientnet-b3",
    "efficientnet-b4": "google/efficientnet-b4",
    "deit-base": "facebook/deit-base-patch16-224",
    "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
}


def resolve_base_model(name):
    """Resolves a --base-model value, expanding known aliases to their full Hugging Face model id."""
    return MODEL_PRESETS.get(name, name)


def str2bool(value):
    """argparse type= helper for proper boolean parsing (unlike bare `type=bool`, which treats any non-empty string, including "False", as True)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value (True/False), got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification Vision Transformer (ViTS) model.")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip data copy/filtering and train using the existing prepared dataset in --filter-data (and split.json if present).",
    )
    parser.add_argument(
        "--remove-long-tail",
        type=bool,
        default=False,
        help="Set to true to truncate the long-tail classes.",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=10,
        help="The minimum number of images per class. If a class has fewer than this, it will be removed if --remove-long-tail is set, or augmented if it is not.",
    )
    parser.add_argument(
        "--remap",
        type=str,
        help="Path to a JSON file that maps the original class names to new class names.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="catsdogs-vit-b16",
        help="Name of the model you want to train.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available --base-model aliases and exit.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/vit-base-patch16-224",
        help=(
            "Base model to fine-tune from. Accepts either a short alias "
            f"({', '.join(sorted(MODEL_PRESETS))}) or any Hugging Face model id, "
            "e.g. google/vit-base-patch16-224 or facebook/dino-vitb8. "
            "Run with --list-models to see aliases and their full ids."
        ),
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        nargs="+",
        default=[(DATA_PATH / "crops").as_posix()],
        help="Paths to the raw dataset (space-separated if multiple paths).",
    )
    parser.add_argument(
        "--filter-data",
        type=str,
        default=(DATA_PATH/ "data_filter").as_posix(),
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
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the trained model to ONNX in the model output directory.",
    )
    parser.add_argument(
        "--depth-csv",
        type=str,
        default=None,
        help=(
            "Path to a CSV with columns 'roi' and 'depth' (meters) giving the depth each ROI was imaged at. "
            "'roi' is the crop filename or its stem (e.g. the localization UUID). When set, depth is fed to the "
            "classification head alongside the image features; ROIs missing from the CSV are treated as unknown depth. "
            "Forces the backbone-only model path (AutoModel + custom head)."
        ),
    )
    parser.add_argument(
        "--depth-dropout",
        type=float,
        default=0.1,
        help=(
            "Probability of hiding an ROI's depth during training, so the model still works when depth is unknown "
            "and doesn't lean on depth alone. Only used with --depth-csv."
        ),
    )
    parser.add_argument(
        "--exclude-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels to exclude from training (space-separated if multiple labels).",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available --base-model aliases:")
        for alias, model_id in sorted(MODEL_PRESETS.items()):
            print(f"  {alias:<24} {model_id}")
        raise SystemExit(0)

    args.base_model = resolve_base_model(args.base_model)
    return args
