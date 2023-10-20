
import pathlib
import asyncio
import argparse
import json
import os
import traceback
from timeit import default_timer as timer
from pprint import pprint

from tqdm.asyncio import tarange
from asyncstdlib import zip as azip

from introspect.client import clients
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
parser.add_argument('--debug',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    type=bool,
                    help='Enable debug mode')
parser.add_argument('--clean-database',
                    action=argparse.BooleanOptionalAction,
                    default=True,
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
        'analysis',
        model=args.model_name, system_message=args.system_message,
        dataset=args.dataset, split=args.split,
        task=args.task, task_config=args.task_config,
        seed=args.seed)

    # connect to inference server
    print('Answerable experiment:')
    print(f' - Endpoint: {args.endpoint}')
    print(f' - Client: {args.client}')
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
    print(f' - Debug: {args.debug}')
    print(f' - Clean database: {args.clean_database}')
    print(f' - Clean cache: {args.clean_cache}')
    print('')

    # Create directories
    os.makedirs(args.persistent_dir / 'database', exist_ok=True)
    os.makedirs(args.persistent_dir / 'results' / 'analysis', exist_ok=True)

    # setup database
    database = result_databases[args.task](
        (args.persistent_dir / 'results' / 'analysis' / experiment_id).with_suffix('.sqlite')
    )
    cache = GenerationCache(experiment_id, cache_dir=args.persistent_dir / 'database', deps=[
        generate_experiment_id('analysis',
                               model=args.model_name, system_message=args.system_message,
                               dataset=args.dataset, split=args.split,
                               task='classify', task_config=classify_task_config,
                               seed=args.seed)

        for classify_task_config in (['no-maybe-redacted'], [])
    ])

    # setup task
    client = clients[args.client](args.endpoint, cache)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir)
    model = models[args.model_type](client, system_message=args.system_message, debug=args.debug, config={'seed': args.seed})
    task = tasks[dataset.category, args.task](dataset, model, config=args.task_config)
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

    # Process observations
    results = {}
    async with cache, database as db:
        async def worker(obs):
            try:
                answer = await task(obs)
            except GenerateError as error:
                answer = error
            await db.put(args.split, obs['idx'], answer)
            return answer

        # process train split
        aggregator = task.make_aggregator()
        async for _, answer in azip(
            pbar := tarange(dataset.num_examples(args.split), desc=aggregator.progress_description),
            AsyncMap(worker, dataset.split(args.split), max_tasks=args.max_workers)
        ):
            if isinstance(answer, GenerateError):
                traceback.print_exception(answer)

            aggregator.add_answer(answer)
            pbar.set_description(aggregator.progress_description)

        # save accumulated results
        results = aggregator.results
        durations['eval'] = aggregator.total_duration

    # save results
    with open((args.persistent_dir / 'results' / 'analysis' / experiment_id).with_suffix('.json'), 'w') as fp:
        json.dump({
            'args': { name: value for name, value in vars(args).items() if name != 'persistent_dir' },
            'results': results,
            'durations': durations
        }, fp)

if __name__ == '__main__':
    asyncio.run(main())
