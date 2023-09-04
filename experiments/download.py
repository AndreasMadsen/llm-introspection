import pathlib
import argparse

from tqdm import tqdm

from introspect.dataset import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    for name, Dataset in (pbar := tqdm(datasets.items())):
        pbar.set_description(f'Downloading dataset {name}')
        dataset = Dataset(persistent_dir=args.persistent_dir)
        dataset.download()
