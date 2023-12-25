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
parser.add_argument('--datasets',
                    nargs='+',
                    action='store',
                    default=list(datasets.keys()),
                    type=str,
                    choices=datasets.keys(),
                    help='The datasets to download')

if __name__ == "__main__":
    args = parser.parse_args()

    for name in (pbar := tqdm(args.datasets)):
        pbar.set_description(f'Downloading dataset {name}')
        dataset = datasets[name](persistent_dir=args.persistent_dir)
        dataset.download()
