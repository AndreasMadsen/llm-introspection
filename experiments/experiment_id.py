
import pathlib
import argparse
from introspect.util import generate_experiment_id, default_model_id, default_model_type
from introspect.types import DatasetSplits, SystemMessage

parser = argparse.ArgumentParser()
parser.add_argument('scriptpath',
                    action='store',
                    type=pathlib.Path,
                    help='The script path that the job will execute')
parser.add_argument('--model-type',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--model-name',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--model-id',
                    action='store',
                    default=None,
                    type=str,
                    help='Model id')
parser.add_argument('--system-message',
                    action='store',
                    default=None,
                    type=SystemMessage,
                    choices=list(SystemMessage),
                    help='Use a system message')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--split',
                    action='store',
                    default=None,
                    type=DatasetSplits,
                    choices=list(DatasetSplits),
                    help='The dataset split to evaluate on')

def main():
    args, _ = parser.parse_known_args()
    args.model_id = default_model_id(args)
    args.model_type = default_model_type(args)
    experiment_id = generate_experiment_id('answerable', args.model_name, args.system_message, args.dataset, args.split)
    print(experiment_id)

if __name__ == '__main__':
    main()
