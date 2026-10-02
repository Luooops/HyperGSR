"""Construct a model from graph sizes and the model configuration only."""

from .direct_sr import DirectSR
from .hyper_gsr import HyperGSR
from .stp_gsr import STPGSR


def build_model(*, n_source_nodes: int, n_target_nodes: int, model_config):
    """Does not seed RNGs or move devices; callers control initialization timing."""
    model_classes = {
        "direct_sr": DirectSR,
        "stp_gsr": STPGSR,
        "hyper_gsr": HyperGSR,
    }
    if model_config.name not in model_classes:
        raise ValueError(f"Unsupported model type: {model_config.name}")
    return model_classes[model_config.name](
        n_source_nodes=n_source_nodes,
        n_target_nodes=n_target_nodes,
        model_config=model_config,
    )
