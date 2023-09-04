
from ..types import DatasetSplits

def generate_experiment_id(name: str,
                           model: str|None = None,
                           dataset: str|None = None, split: DatasetSplits|None = None,
                           seed: int|None = None):
    """Creates a standardized experiment name.

    The format is
        {name}_m-{model}_d-{dataset}_s-{seed}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment.
        model (str, optional): the name of the model.
        dataset (str, optional): the name of the dataset.
        split (DatasetSplits, optional): the dataset split.
        seed (int, optional): the models initialization seed.
    Returns:
        str: the experiment identifier
    """
    experiment_id = f"{name.lower()}"
    if isinstance(model, str):
        experiment_id += f"_m-{model.lower()}"
    if isinstance(dataset, str):
        experiment_id += f"_d-{dataset.lower()}"
    if isinstance(split, DatasetSplits):
        experiment_id += f"_p-{str(split).lower()}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id
