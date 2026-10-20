# vitstrain
# Filename: src/model_factory.py
# Description: Build and load image classification models, shared by training and inference
import re
from logging import Logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, MODEL_MAPPING, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

DEFAULT_CLASSIFIER_DROPOUT = 0.1
DEFAULT_DEPTH_EMBED_DIM = 16

# ViT/DeiT classifiers use the CLS token directly; their checkpoints have no trained pooler,
# so building one would add randomly initialized weights in front of the head
NO_POOLER_MODEL_TYPES = ("vit", "deit")


def get_hidden_size(logger: Logger, model):
    """Returns the hidden size dimension from the backbone's configuration. Falls back to None."""

    config = model.config

    d = getattr(config, "hidden_size", None)
    if d is not None:
        logger.debug(f"Using config.hidden_size={d}")
        return d

    d = getattr(config, "embed_dim", None)
    if d is not None:
        logger.debug(f"Using config.embed_dim={d}")
        return d

    d = getattr(config, "hidden_sizes", None)
    if d is not None and isinstance(config.hidden_sizes, (list, tuple)):
        d = config.hidden_sizes[-1]
        logger.debug(f"Using config.hidden_sizes[-1]={d}")
        return d

    d = getattr(model, "num_features", None)
    if d is not None:
        logger.debug(f"Using model.num_features={d}")
        return d

    raise RuntimeError(
        "Unable to determine hidden size for "
        f"{model.__class__.__name__}. "
        "You may want to extend get_hidden_size() to support this backbone."
    )


def extract_features(outputs):
    """Reduce a backbone's output to a single feature vector per image."""

    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output

    if getattr(outputs, "last_hidden_state", None) is not None:
        hidden = outputs.last_hidden_state

        if hidden.ndim == 3:
            return hidden[:, 0]  # e.g., ViT / DINO

        if hidden.ndim == 4:
            return hidden.mean(dim=(-2, -1))  # e.g., ConvNeXt

    raise RuntimeError(f"Unsupported output type: {type(outputs)}")


class DepthEncoder(nn.Module):
    """
    Embeds a per-ROI depth (meters) as a small feature vector for the classification head.

    Input is [z-scored depth, missing flag]; NaN depth means unknown. The normalization stats are
    buffers, so they are saved with the model and reused at inference.
    """

    def __init__(self, depth_mean, depth_std, embed_dim=DEFAULT_DEPTH_EMBED_DIM, depth_dropout=0.1):
        super().__init__()
        self.register_buffer("depth_mean", torch.tensor(float(depth_mean)))
        self.register_buffer("depth_std", torch.tensor(float(depth_std)))
        self.depth_dropout = depth_dropout
        self.mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, depth):
        missing = torch.isnan(depth)
        if self.training and self.depth_dropout > 0:
            # Hide some known depths so the head also learns the image-only case
            missing = missing | (torch.rand_like(depth) < self.depth_dropout)
        z = torch.where(missing, torch.zeros_like(depth), (depth - self.depth_mean) / self.depth_std)
        return self.mlp(torch.stack([z, missing.float()], dim=-1))


def backbone_kwargs(config):
    """Extra constructor arguments for a backbone, so it is built the same way at training and load time."""
    return {"add_pooling_layer": False} if config.model_type in NO_POOLER_MODEL_TYPES else {}


def backbone_classifier_class(backbone_cls):
    """Builds a PreTrainedModel classification wrapper around a backbone-only model class.

    Used for backbones that transformers ships without a classification head (e.g. DINOv3).
    The backbone is stored under its own base_model_prefix rather than a generic name, so the
    saved checkpoint looks like any other transformers classifier: config.json carries the
    backbone's model_type and the label maps, and the weight keys are prefixed such that
    AutoModel.from_pretrained() loads the fine-tuned backbone and ignores the head. Naming the
    submodule anything else makes AutoModel silently reinitialize every backbone weight.
    """

    prefix = backbone_cls.base_model_prefix
    # Inherit the backbone's own PreTrainedModel base so config_class and the attention /
    # gradient checkpointing support flags carry over.
    base_pretrained = next(c for c in backbone_cls.__mro__[1:] if issubclass(c, PreTrainedModel))

    class BackboneClassifier(base_pretrained):
        base_model_prefix = prefix

        def __init__(self, config, backbone=None):
            super().__init__(config)
            setattr(self, prefix, backbone if backbone is not None else backbone_cls(config, **backbone_kwargs(config)))
            self.dropout = nn.Dropout(getattr(config, "classifier_dropout", DEFAULT_CLASSIFIER_DROPOUT))

            # Optional ROI-depth conditioning; its settings live in the config so load_model() rebuilds it
            self.depth_encoder = None
            depth_dim = getattr(config, "depth_embed_dim", None) or 0
            if depth_dim:
                self.depth_encoder = DepthEncoder(
                    config.depth_mean, config.depth_std, embed_dim=depth_dim, depth_dropout=config.depth_dropout
                )

            self.classifier = nn.Linear(config.classifier_input_size + depth_dim, config.num_labels)

        def forward(self, pixel_values, labels=None, depth=None):
            outputs = self.base_model(pixel_values=pixel_values)
            features = self.dropout(extract_features(outputs))
            if self.depth_encoder is not None:
                if depth is None:
                    depth = torch.full(features.shape[:1], float("nan"), device=features.device)
                features = torch.cat([features, self.depth_encoder(depth.to(features.dtype))], dim=-1)
            logits = self.classifier(features)
            loss = F.cross_entropy(logits, labels) if labels is not None else None

            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
            )

        def freeze_backbone(self):
            for p in self.base_model.parameters():
                p.requires_grad = False

        def unfreeze_backbone(self):
            for p in self.base_model.parameters():
                p.requires_grad = True

    BackboneClassifier.__name__ = re.sub(r"Model$", "", backbone_cls.__name__) + "ForImageClassification"
    BackboneClassifier.__qualname__ = BackboneClassifier.__name__

    return BackboneClassifier


def create_model(logger: Logger, model_name, id2label, freeze_backbone=False, dropout=DEFAULT_CLASSIFIER_DROPOUT, depth=None):
    """
    Creates a vision classifier with the specified backbone and number of labels.

    depth, if given, is {"mean", "std", "dropout"} for the training split's ROI depths and conditions the
    head on depth. It always uses the backbone-only path, since the transformers classification heads can't take depth.
    """

    num_classes = len(id2label)
    label2id = {v: k for k, v in id2label.items()}

    if depth is None:
        try:
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            return model

        except (ValueError, OSError) as e:
            logger.info(f"Falling back to backbone model: {e}")

    backbone = AutoModel.from_pretrained(model_name, **backbone_kwargs(AutoConfig.from_pretrained(model_name)))

    config = backbone.config
    config.id2label = id2label
    config.label2id = label2id
    config.classifier_dropout = dropout
    config.classifier_input_size = get_hidden_size(logger, backbone)
    if depth is not None:
        config.depth_embed_dim = DEFAULT_DEPTH_EMBED_DIM
        config.depth_mean = depth["mean"]
        config.depth_std = depth["std"]
        config.depth_dropout = depth["dropout"]

    model = backbone_classifier_class(type(backbone))(config, backbone=backbone)

    if freeze_backbone:
        model.freeze_backbone()
        logger.info("Backbone frozen.")

    return model


class LogitsOnly(nn.Module):
    """ONNX has no equivalent of ImageClassifierOutput, so expose the logits tensor directly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def export_onnx(logger: Logger, model, onnx_path, image_size, opset_version=17):
    """Exports a trained classifier to ONNX and checks it against the PyTorch outputs.

    eval() matters for more than dropout here: DINOv3 randomizes its rotary position
    embeddings while training, which would otherwise be traced into the graph.
    """

    import onnxruntime as ort

    was_training = model.training
    wrapped = LogitsOnly(model).eval()

    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    logger.info(f"Exporting to ONNX with input 1x3x{image_size}x{image_size}, opset {opset_version}")
    torch.onnx.export(
        wrapped,
        (dummy,),
        onnx_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset_version,
    )

    with torch.inference_mode():
        expected = wrapped(dummy).cpu().numpy()

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pixel_values": dummy.cpu().numpy()})[0]
    max_diff = float(np.abs(actual - expected).max())

    if max_diff > 1e-4:
        logger.warning(f"ONNX outputs differ from PyTorch by {max_diff:.2e}; check {onnx_path.name}")
    else:
        logger.info(f"ONNX model saved to {onnx_path.name} (max difference from PyTorch {max_diff:.2e})")

    # torch.onnx.export restores the mode of the module it was handed, which is the wrapper.
    model.train(was_training)

    return onnx_path


def load_model(logger: Logger, model_dir):
    """Loads a classifier previously saved by fine_tune_vits.py."""

    config = AutoConfig.from_pretrained(model_dir)

    # A depth-conditioned head only exists on the backbone classifier
    if not getattr(config, "depth_embed_dim", None):
        try:
            return AutoModelForImageClassification.from_pretrained(model_dir)

        except (ValueError, OSError) as e:
            logger.info(f"Falling back to backbone classifier: {e}")

    return backbone_classifier_class(MODEL_MAPPING[type(config)]).from_pretrained(model_dir)
