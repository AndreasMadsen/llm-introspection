
import json
import argparse
import os
import pathlib
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import plotnine as p9

from introspect.dataset import datasets
from introspect.types import DatasetSplits, SystemMessage, TaskCategories
from introspect.util import generate_experiment_id
from introspect.plot import annotation, tag

def tex_format_time(secs):
    hh, mm = divmod(secs // 60, 60)
    return f'{int(hh):02d}:{int(mm):02d}'

def tex_multirow(rows, content):
    return f'\\multirow{{{rows}}}{{*}}{{{content}}}'

parser = argparse.ArgumentParser(
    description = 'Plots the 0% masking test performance given different training masking ratios'
)
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--model-names',
                    nargs='+',
                    action='store',
                    default=['llama2-70b', 'llama2-7b', 'falcon-40b', 'falcon-7b', 'mistral-v1-7b'],
                    type=str,
                    help='Model names')
parser.add_argument('--system-message',
                    action='store',
                    default=SystemMessage.DEFAULT,
                    type=SystemMessage,
                    choices=list(SystemMessage),
                    help='Use a system message')
parser.add_argument('--datasets',
                    nargs='+',
                    action='store',
                    default=['IMDB'],
                    type=str,
                    choices=datasets.keys(),
                    help='The dataset to fine-tune on')
parser.add_argument('--split',
                    action='store',
                    default=DatasetSplits.TEST,
                    type=DatasetSplits,
                    choices=list(DatasetSplits),
                    help='The dataset split to evaluate on')
parser.add_argument('--tasks',
                    nargs='+',
                    action='store',
                    default=[TaskCategories.CLASSIFY, TaskCategories.COUNTERFACTUAL, TaskCategories.IMPORTANCE, TaskCategories.REDACTED],
                    type=TaskCategories,
                    choices=list(TaskCategories),
                    help='Which tasks to run')
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

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args = parser.parse_args()

    experiment_id = generate_experiment_id('walltime',
        model=''.join(d[0] for d in args.model_names), system_message=args.system_message,
        dataset=''.join(d[0] for d in args.datasets), split=args.split,
        task=''.join(t[0] for t in args.tasks),
        seed=args.seed)

    # Read JSON files into dataframe
    results = []
    files = sorted((args.persistent_dir / 'results' / 'analysis').glob('analysis_*.json'))
    for file in tqdm(files, desc='Loading answerable .json files'):
        try:
            with open(file, 'r') as fp:
                data = json.load(fp)
        except Exception as error:
            raise Exception(f'{file} caused an error') from error

        if data['args']['model_name'] in args.model_names and \
            data['args']['dataset'] in args.datasets and \
            data['args']['split'] == args.split and \
            data['args']['task'] in args.tasks and \
            set(data['args']['task_config']) == set(args.task_config):
            if data['results']['error'] > 0 or data['results']['missmatch'] > 0:
                tqdm.write(f'Detected error ({data["results"]["error"]}) or missmatch ({data["results"]["missmatch"]}) in {file.name}')

            data['plot'] = {
                'model-name': tag.model_name(data['args']['model_name']),
                'model-size': tag.model_size(data['args']['model_name'])
            }

            if None not in data['plot'].values():
                results.append(data)


    # Convert results into a flat DataFrame
    df = pd.json_normalize(results)
    df = df.loc[:, ['args.dataset', 'args.task', 'plot.model-size', 'plot.model-name', 'durations.eval']]
    df['plot.walltime'] = df.loc[:, 'durations.eval'] / (1000)
    os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
    df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    models = defaultdict(list)
    for model_name in args.model_names:
        models[tag.model_name(model_name)].append(tag.model_size(model_name))

    os.makedirs(args.persistent_dir / 'tables', exist_ok=True)
    with open(args.persistent_dir / 'tables' / f'{experiment_id}.tex', 'w') as fp:
        print(r'\begin{tabular}[t]{lllcccc}', file=fp)
        print(r'\toprule', file=fp)
        print(r'Dataset & Model & Size & \multicolumn{4}{c}{Inference time [hh:mm]} \\', file=fp)
        print(r'\cmidrule(r){4-7}', file=fp)
        print(r'& & & Classify & Counterfactual & Redacted & Importance \\', file=fp)
        print(r'\midrule', file=fp)

        for dataset_i, dataset in enumerate(args.datasets):
            df_dataset = df.query('`args.dataset` == @dataset')
            for model_i, (model_name, model_sizes) in enumerate(models.items()):
                for size_i, model_size in enumerate(model_sizes):
                    df_model = df_dataset.query('`plot.model-name` == @model_name & `plot.model-size` == @model_size')
                    df_classify = df_model.query('`args.task` == "classify"').loc[:, 'plot.walltime'].iat[0]
                    df_counterfactual = df_model.query('`args.task` == "counterfactual"').loc[:, 'plot.walltime'].iat[0]
                    df_importance = df_model.query('`args.task` == "importance"').loc[:, 'plot.walltime'].iat[0]
                    df_redacted = df_model.query('`args.task` == "redacted"').loc[:, 'plot.walltime'].iat[0]

                    time_tex = f'{tex_format_time(df_classify)} & ' \
                               f'{tex_format_time(df_counterfactual)} & ' \
                               f'{tex_format_time(df_importance)} & ' \
                               f'{tex_format_time(df_redacted)} \\\\'

                    dataset_tex = tex_multirow(len(args.model_names), dataset) if model_i == 0 and size_i == 0 else ''
                    model_tex = tex_multirow(len(model_sizes), annotation.model_type[model_name]) if size_i == 0 else ''
                    size_tex = f'{model_size}B'

                    if len(args.datasets) > dataset_i > 0 and model_i == 0 and size_i == 0:
                        print(r'\cmidrule{2-7}', file=fp)
                    if len(args.datasets) > dataset_i and model_i > 0 and size_i == 0:
                        print(r'\cmidrule{3-7}', file=fp)

                    print(f'{dataset_tex} & {model_tex} & {size_tex} & {time_tex}', file=fp)

        print(r'\bottomrule', file=fp)
        print(r'\end{tabular}', file=fp)
