
import asyncio
import argparse
import pathlib
import re
import os

from introspect.dataset import datasets
from introspect.types import SystemMessage, TaskCategories, DatasetSplits, Observation, FaithfulResult
from introspect.model import Llama2Model
from introspect.client import clients
from introspect.tasks import tasks
from introspect.util import generate_experiment_id
from introspect.database import GenerationCache
from introspect.plot import tag, annotation

parser = argparse.ArgumentParser(
    description = 'Plots the 0% masking test performance given different training masking ratios'
)
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--endpoint',
                    action='store',
                    default='http://127.0.0.1:20002',
                    type=str,
                    help='The TGI endpoint for this model')
parser.add_argument('--client',
                    action='store',
                    default='Offline' if 'RUN_OFFLINE' in os.environ else 'TGI',
                    type=str,
                    choices=clients.keys(),
                    help='Which client to use, either TGI or VLLM')
parser.add_argument('--system-message',
                    action='store',
                    default=SystemMessage.DEFAULT,
                    type=SystemMessage,
                    choices=list(SystemMessage),
                    help='Use a system message')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    type=str,
                    choices=datasets.keys(),
                    help='The dataset to fine-tune on')
parser.add_argument('--split',
                    action='store',
                    default=DatasetSplits.TRAIN,
                    type=DatasetSplits,
                    choices=list(DatasetSplits),
                    help='The dataset split to evaluate on')
parser.add_argument('--task',
                    action='store',
                    default=TaskCategories.ANSWERABLE,
                    type=TaskCategories,
                    choices=list(TaskCategories),
                    help='Which task to run')
parser.add_argument('--task-config',
                    action='store',
                    nargs='*',
                    default=[],
                    type=str,
                    help='List of configuration options for selected task')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed used for generation')
parser.add_argument('--idx',
                    action='store',
                    default=0,
                    type=int,
                    help='The observation index of the dataset split to print')

def latex_encode_newline(content: str) -> str:
    return content.replace('\n\n', r'\\[1em]').replace('\n', r'\\')

async def main():
    args = parser.parse_args()
    experiment_id = generate_experiment_id(
        'analysis',
        model='llama2-70b', system_message=args.system_message,
        dataset=args.dataset, split=args.split,
        task=args.task, task_config=args.task_config,
        seed=args.seed)

    # Check cache is valid
    if not GenerationCache.exists(experiment_id, args.persistent_dir / 'database'):
        raise ValueError(f'cache "{experiment_id}" does not exist')

    # setup task
    cache = GenerationCache(experiment_id, cache_dir=args.persistent_dir / 'database')
    client = clients[args.client](args.endpoint, cache, record=True)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    model = Llama2Model(client, system_message=args.system_message, config={'seed': args.seed})
    task = tasks[dataset.category, args.task](model, config=args.task_config)

    message_pairs = []
    obs: Observation|None = None
    evalulation: FaithfulResult|None = None
    async with cache:
        for scan_obs in dataset.split(args.split):
            if scan_obs['idx'] == args.idx:
                obs = scan_obs
                break
        if obs is None:
            raise ValueError(f'Could not find observation with idx {args.idx}')
        evalulation = await task(obs) # populates the test-client logs
        if evalulation is None:
            raise ValueError('Unreachable')
        if evalulation['faithful'] is None:
            raise ValueError('Something went wrong with the faithfulness evaluation')

        for prompt, response in client.record:
            p = re.match(r'<s>\[INST\] (?:<<SYS>>.+<</SYS>>' '\n\n' r')?(.+) \[/INST\]$', prompt, flags=re.DOTALL)
            if p is None:
                error = ValueError('Could not identify user prompt')
                error.add_note('Prompt: ' + prompt)
                raise error

            user_prompt = p.group(1).strip()
            model_response = response['response'].strip()
            message_pairs.append((user_prompt, model_response))

    config = { 'Persona instruction': tag.explain_persona(args.task_config) }
    match args.task:
        case 'counterfactual':
            config['Counterfactual target'] = annotation.persona.labeller(tag.explain_counterfactual_target(args.task_config))
        case 'redacted' | 'importance':
            config['Redaction instruction'] = annotation.redact.labeller(tag.explain_redact(args.task_config))

    config_str = ', '.join(f'{config_name}: {config_value}' for config_name, config_value in config.items())
    correct_str = 'correct' if evalulation["correct"] else 'not correct'
    faithful_str = 'faithful' if evalulation["faithful"] else 'not faithful'
    task_str = ({
        'counterfactual': 'Counterfactual',
        'importance': 'Feature attribution',
        'redacted': 'Redaction'
    })[args.task]
    session_str = [
        'Session 1: Classification',
        'Session 2: Explanation',
        'Session 3: Consistency check',
    ]

    caption = (f'{task_str} explanation and interpretability-faithfulness evaluation,'
               f' with the configuration ``{config_str}\'\'.'
               f' The true label is ``{obs["label"]}\'\'.'
               f' The initial prediction was ``{correct_str}\'\'.'
               f' The interpretability-faithfulness was evaluted to be ``{faithful_str}\'\'.')

    print('Explanation prompt figure:')
    print(f' - Task: {task_str}')
    print(f' - Config: {config_str}')
    print('')
    print(f' - Label: {obs["label"]}')
    print(f' - Correct: {correct_str}')
    print(f' - Faithful: {faithful_str}')
    print('')

    os.makedirs(args.persistent_dir / 'figure', exist_ok=True)
    with open((args.persistent_dir / 'figure' / experiment_id).with_suffix('.tex'), 'w') as fp:
        for session_i, (user_prompt, model_response) in enumerate(message_pairs):
            print(f'\\session{{{session_str[session_i]}}}', file=fp)
            print(f'\\user{{{latex_encode_newline(user_prompt)}}}', file=fp)
            print(f'\\model{{{latex_encode_newline(model_response)}}}', file=fp)
        print(f'\\caption{{{caption}}}', file=fp)

if __name__ == '__main__':
    asyncio.run(main())
