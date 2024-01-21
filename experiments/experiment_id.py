
import pathlib
import argparse
from introspect.util import generate_experiment_id, default_model_id, default_model_type, default_system_message

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
parser.add_argument('--model-type',
                    action='store',
                    default=None,
                    type=str,
                    help='Model type')
parser.add_argument('--model-id',
                    action='store',
                    default=None,
                    type=str,
                    help='Model id')
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
parser.add_argument('--task',
                    action='store',
                    default='answerable',
                    type=str,
                    help='Which task to run')
parser.add_argument('--task-config',
                    action='store',
                    nargs='*',
                    default=[],
                    type=str,
                    help='List of configuration options for selected task')
parser.add_argument('--seed',
                    action='store',
                    default=None,
                    type=int,
                    help='Seed used for generation')

def main():
    args, _ = parser.parse_known_args()
    args.model_id = default_model_id(args)
    args.model_type = default_model_type(args)
    args.system_message = default_system_message(args)

    experiment_id = generate_experiment_id(
        args.scriptpath.name.rstrip('.py'),
        model=args.model_name, system_message=args.system_message,
        dataset=args.dataset, split=args.split,
        task=args.task, task_config=args.task_config,
        seed=args.seed)
    print(experiment_id)

if __name__ == '__main__':
    main()
