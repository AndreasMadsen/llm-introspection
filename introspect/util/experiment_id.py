
from ..types import DatasetSplits, SystemMessage

def generate_experiment_id(name: str,
                           model: str|None = None, system_message: SystemMessage|None = None,
                           dataset: str|None = None, split: DatasetSplits|None = None):
    """Creates a standardized experiment name.

    The format is
        {name}_m-{model}_d-{dataset}_s-{seed}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment.
        model (str, optional): the name of the model.
        system_message (SystemMessage, optional): the system message mode.
        dataset (str, optional): the name of the dataset.
        split (DatasetSplits, optional): the dataset split.
    Returns:
        str: the experiment identifier
    """
    experiment_id = f"{name.lower()}"
    if isinstance(model, str):
        experiment_id += f"_m-{model.lower()}"
    if isinstance(system_message, SystemMessage):
        experiment_id += f"_s-{str(system_message).lower()}"
    if isinstance(dataset, str):
        experiment_id += f"_d-{dataset.lower()}"
    if isinstance(split, DatasetSplits):
        experiment_id += f"_p-{str(split).lower()}"

    return experiment_id
