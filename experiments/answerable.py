
import pathlib
import asyncio
import argparse
import json
import os
import signal
import traceback
from timeit import default_timer as timer
from pprint import pprint

from tqdm.asyncio import tarange
from asyncstdlib import zip as azip

from introspect.client import clients
from introspect.dataset import datasets
from introspect.model import models
from introspect.tasks import tasks
from introspect.util import AsyncMap, generate_experiment_id, default_model_id, default_model_type, cancel_eventloop_on_signal
from introspect.database import Answerable, GenerationCache
from introspect.types import DatasetSplits, SystemMessage

parser = argparse.ArgumentParser()
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
                    default='TGI',
                    type=str,
                    choices=clients.keys(),
                    help='Which client to use, either TGI or VLLM')
parser.add_argument('--model-name',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--model-type',
                    action='store',
                    default=None,
                    type=str,
                    choices=models.keys(),
                    help='Model type')
parser.add_argument('--model-id',
                    action='store',
                    default=None,
                    type=str,
                    help='Model id')
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
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed used for generation')
parser.add_argument('--num-tasks',
                    action='store',
                    default=100,
                    type=int,
                    help='Max number of parallel async tasks')
parser.add_argument('--debug',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    type=bool,
                    help='Enable debug mode')
parser.add_argument('--clean-database',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    type=bool,
                    help='Remove result database')
parser.add_argument('--clean-cache',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    type=bool,
                    help='Remove cache')

async def main():
    durations = {}
    setup_time_start = timer()

    args = parser.parse_args()
    args.model_id = default_model_id(args)
    args.model_type = default_model_type(args)
    experiment_id = generate_experiment_id(
        'answerable', args.model_name, args.system_message, args.dataset, args.split, args.seed)

    # connect to inference server
    print('Answerable experiment:')
    print(f' - Endpoint: {args.endpoint}')
    print(f' - Client: {args.client}')
    print(f' - Number of tasks: {args.num_tasks}')
    print('')
    print(f' - Model name: {args.model_name}')
    print(f' - Model type: {args.model_type}')
    print(f' - Model id: {args.model_id}')
    print(f' - System message: {args.system_message}')
    print(f' - Dataset: {args.dataset}')
    print(f' - Split: {args.split}')
    print(f' - Seed: {args.seed}')
    print('')
    print(f' - Debug: {args.debug}')
    print(f' - Clean database: {args.clean_database}')
    print(f' - Clean cache: {args.clean_cache}')
    print('')

    # Create directories
    os.makedirs(args.persistent_dir / 'database', exist_ok=True)
    os.makedirs(args.persistent_dir / 'results' / 'answerable', exist_ok=True)

    # setup experiment
    client = clients[args.client](args.endpoint)
    database = Answerable(experiment_id, persistent_dir=args.persistent_dir)
    cache = GenerationCache(
        generate_experiment_id('cache', args.model_name, args.system_message, args.dataset, args.seed),
        persistent_dir=args.persistent_dir)

    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir)
    model = models[args.model_type](client, cache, system_message=args.system_message, debug=args.debug, config={'seed': args.seed})
    task = tasks[dataset.category](dataset, model).answerable
    durations['setup'] = timer() - setup_time_start

    # cleanup old database
    if args.clean_database:
        database.remove()
    if args.clean_cache:
        cache.remove()

    # connect to inference server
    print('Waiting for connection ...')
    await client.connect()
    print('Connection established')
    pprint(await client.info())

    # Set the signal handler
    cancel_eventloop_on_signal(signal.SIGTERM)

    # Process observations
    results = []
    async with cache, database as db:
        async def worker(obs):
            answer = await task(obs)
            await db.put(args.split, obs['idx'], answer)
            return answer

        # process train split
        train_time_start = timer()
        introspect_count, correct_count, error_count = (0, 0, 0)
        async for _, answer in azip(
            pbar := tarange(dataset.num_examples(args.split), desc='Processing[C=0, I=0, E=0]'),
            AsyncMap(worker, dataset.split(args.split), max_tasks=args.num_tasks)
        ):
            if answer['error'] is not None:
                traceback.print_exception(answer['error'])

            if answer['introspect'] is None or answer['correct'] is None:
                #print(f'{answer["answer_ability"]} -> {answer["answer_sentiment"]}')
                error_count += 1
            else:
                introspect_count += int(answer['introspect'])
                correct_count += int(answer['correct'])

            pbar.set_description(f'Processing[C={correct_count}, I={introspect_count}, E={error_count}]')

        # save accumulated results
        results.append({
            'split': args.split,
            'introspect': introspect_count,
            'correct': correct_count,
            'error': error_count,
            'total': dataset.num_examples(args.split)
        })
        durations[f'eval_{args.split}'] = timer() - train_time_start

    # save results
    with open((args.persistent_dir / 'results' / 'answerable' / experiment_id).with_suffix('.json'), 'w') as fp:
        json.dump({
            'args': { name: value for name, value in vars(args).items() if name != 'persistent_dir' },
            'results': results,
            'durations': durations
        }, fp)

if __name__ == '__main__':
    asyncio.run(main())
