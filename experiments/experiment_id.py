
import pathlib
import argparse
from introspect.util import generate_experiment_id

parser = argparse.ArgumentParser()
parser.add_argument('scriptpath',
                    action='store',
                    type=pathlib.Path,
                    help='The script path that the job will execute')
parser.add_argument('--model-name',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--system-message',
                    action='store',
                    default=None,
                    type=str,
                    help='Use a system message')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--split',
                    action='store',
                    default=None,
                    type=str,
                    help='The dataset split to evaluate on')
parser.add_argument('--seed',
                    action='store',
                    default=None,
                    type=int,
                    help='Seed used for generation')

def main():
    args, _ = parser.parse_known_args()

    experiment_id = generate_experiment_id(
        args.scriptpath.name.rstrip('.py'),
        args.model_name, args.system_message, args.dataset, args.split, args.seed)
    print(experiment_id)

if __name__ == '__main__':
    main()
