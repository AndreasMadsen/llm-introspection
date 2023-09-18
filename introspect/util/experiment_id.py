
from typing import Optional

def generate_experiment_id(name: str,
                           model: Optional[str] = None, system_message: Optional[str] = None,
                           dataset: Optional[str] = None, split: Optional[str] = None,
                           seed: Optional[int] = None):
    """Creates a standardized experiment name.

    The format is
        {name}_m-{model}_d-{dataset}_s-{seed}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment.
        model (str, optional): the name of the model.
        system_message (str, optional): the system message mode.
        dataset (str, optional): the name of the dataset.
        split (str, optional): the dataset split.
        seed (int, optional): the generation seed.
    Returns:
        str: the experiment identifier
    """
    experiment_id = f"{name}"
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(system_message, str):
        experiment_id += f"_y-{system_message}"
    if isinstance(dataset, str):
        experiment_id += f"_d-{dataset}"
    if isinstance(split, str):
        experiment_id += f"_p-{split}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id.lower()
