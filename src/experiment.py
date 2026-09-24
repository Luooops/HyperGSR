"""Shared experiment setup; no configuration or filesystem changes on import."""


def get_run_dir(config) -> str:
    """Return the legacy result path, including its trailing slash."""
    base = config.experiment.base_dir
    model = config.model.name
    dataset = config.dataset.name
    run = config.experiment.run_name
    if model == "hyper_gsr":
        return f"{base}/{model}/{dataset}/{config.model.hyper_dual_learner.mode}/{run}/"
    return f"{base}/{model}/{dataset}/{run}/"


def seed_experiment(random_seed: int) -> None:
    """Keep the original seed calls and their order (including CUDA calls)."""
    import numpy as np
    import torch

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
