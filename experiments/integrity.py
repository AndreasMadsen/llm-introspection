
import pathlib
import asyncio
import argparse
import os
from timeit import default_timer as timer

from tqdm.asyncio import tarange
from asyncstdlib import zip as azip

from introspect.client import OfflineClient
from introspect.dataset import datasets
from introspect.model import models
from introspect.tasks import tasks
from introspect.util import AsyncMap, generate_experiment_id, default_model_id, default_model_type
from introspect.database import result_databases, GenerationCache
from introspect.types import TaskCategories, DatasetSplits, SystemMessage, GenerateError

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
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
parser.add_argument('--task',
                    action='store',
                    default=TaskCategories.ANSWERABLE,
                    type=TaskCategories,
                    choices=list(TaskCategories),
                    help='Which task to run')
parser.add_argument('--task-config',
                    action='store',
                    nargs='+',
                    default=[],
                    type=str,
                    help='List of configuration options for selected task')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed used for generation')
parser.add_argument('--max-workers',
                    action='store',
                    default=100,
                    type=int,
                    help='Max number of parallel async tasks')

async def main():
    durations = {}
    setup_time_start = timer()

    args = parser.parse_args()
    args.model_id = default_model_id(args)
    args.model_type = default_model_type(args)
    experiment_id = generate_experiment_id(
        'analysis',
        model=args.model_name, system_message=args.system_message,
        dataset=args.dataset, split=args.split,
        task=args.task, task_config=args.task_config,
        seed=args.seed)

    # connect to inference server
    print('Answerable experiment:')
    print(f' - Maximum number of workers: {args.max_workers}')
    print('')
    print(f' - Model name: {args.model_name}')
    print(f' - Model type: {args.model_type}')
    print(f' - Model id: {args.model_id}')
    print(f' - System message: {args.system_message}')
    print(f' - Task: {args.task}')
    print(f' - Task config: [{", ".join(args.task_config)}]')
    print(f' - Dataset: {args.dataset}')
    print(f' - Split: {args.split}')
    print(f' - Seed: {args.seed}')
    print('')

    # Create directories
    os.makedirs(args.persistent_dir / 'database', exist_ok=True)
    os.makedirs(args.persistent_dir / 'results' / 'analysis', exist_ok=True)

    # setup database
    database = result_databases[args.task](experiment_id, persistent_dir=args.persistent_dir)
    cache = GenerationCache(
        generate_experiment_id('cache', args.model_name, args.system_message, args.dataset, args.seed),
        persistent_dir=args.persistent_dir)

    # setup task
    client = OfflineClient(cache=cache)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir)
    model = models[args.model_type](client, system_message=args.system_message, config={'seed': args.seed})
    task = tasks[dataset.category, args.task](dataset, model, config=args.task_config)

    # connect to inference server
    print('Waiting for connection ...')
    await client.connect()

    # Process observations
    async with cache, database as db:
        async def worker(obs):
            try:
                answer = await task(obs)
            except GenerateError as error:
                return { 'error': error, 'success': False, 'integrity': True }

            store = await db.get(args.split, obs['idx'])
            match store:
                case GenerateError():
                    return { 'error': store, 'success': False, 'integrity': False }
                case None:
                    return { 'error': None, 'success': False, 'integrity': False }
                case _:
                    return {
                        'error': None,
                        'success': True,
                        'integrity': all(store[key] == answer[key] for key in answer.keys() if key.endswith('_source'))
                    }

        # process train split
        error_count = 0
        integrity_count = 0
        success_count = 0
        async for _, match in azip(
            pbar := tarange(dataset.num_examples(args.split), desc='Integrity[I=0, S=0, E=0]'),
            AsyncMap(worker, dataset.split(args.split), max_tasks=args.max_workers)
        ):
            if isinstance(match['error'], GenerateError):
                error_count += 1
            if match['integrity'] is not None:
                integrity_count += int(match['integrity'])
            success_count += int(match['success'])

            pbar.set_description(f'Integrity[I={integrity_count}, S={success_count}, E={error_count}]')


if __name__ == '__main__':
    asyncio.run(main())
