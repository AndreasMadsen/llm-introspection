
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd
import plotnine as p9

from introspect.dataset import datasets
from introspect.types import DatasetSplits, TaskCategories
from introspect.util import generate_experiment_id
from introspect.plot import annotation, tag

parser = argparse.ArgumentParser(
    description = 'Plots the 0% masking test performance given different training masking ratios'
)
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--stage',
                    action='store',
                    default='both',
                    type=str,
                    choices=['preprocess', 'plot', 'both'],
                    help='Which export stage should be performed. Mostly just useful for debugging.')
parser.add_argument('--format',
                    action='store',
                    default='paper',
                    type=str,
                    choices=['paper', 'keynote', 'appendix'],
                    help='The dimentions and format of the plot.')
parser.add_argument('--model-names',
                    nargs='+',
                    action='store',
                    default=['llama2-70b', 'llama2-7b', 'falcon-40b', 'falcon-7b', 'mistral-v1-7b'],
                    choices=['llama2-70b', 'llama2-7b', 'falcon-40b', 'falcon-7b', 'mistral-v1-7b'],
                    type=str,
                    help='Model name')
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
parser.add_argument('--task',
                    action='store',
                    default=TaskCategories.CLASSIFY,
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

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args = parser.parse_args()

    experiment_id = generate_experiment_id('classify_accuracy',
        model='-'.join(args.model_names),
        dataset='-'.join(args.datasets), split=args.split,
        task=args.task, task_config=args.task_config,
        seed=args.seed)

    if args.stage in ['both', 'preprocess']:
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
               data['args']['task'] == args.task and \
               set(data['args']['task_config']) == set(args.task_config):
                if data['results']['error'] > 0 or data['results']['missmatch'] > 0:
                    tqdm.write(f'Detected error ({data["results"]["error"]}) or missmatch ({data["results"]["missmatch"]}) in {file.name}')

                data['plot'] = {
                    'model-name': tag.model_name(data['args']['model_name']),
                    'model-size': tag.model_size(data['args']['model_name']),
                }

                if None not in data['plot'].values():
                    results.append(data)

        # Convert results into a flat DataFrame
        df = pd.json_normalize(results)
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
        df = df.assign(**{
          'plot.accuracy': df.loc[:, 'results.correct'] / (df.loc[:, 'results.total'] - df.loc[:, 'results.missmatch'])
        })

        p = (
            p9.ggplot(df, p9.aes(x='plot.model-size')) +
            p9.geom_point(p9.aes(y='plot.accuracy', color='plot.model-name')) +
            p9.geom_line(p9.aes(y='plot.accuracy', color='plot.model-name')) +
            p9.facet_wrap('args.dataset', nrow=1) +
            p9.scale_y_continuous(
                name='Accuracy',
                limits=[0, 1]
            ) +
            p9.scale_x_continuous(
                name='Model size [B]',
                limits=[-10, 80],
                breaks=[7, 40, 70],
            ) +
            p9.scale_color_discrete(
                breaks=annotation.model_type.breaks,
                labels=annotation.model_type.labels,
                aesthetics=["color"],
                name='Model type'
            )
        )

        if args.format == 'paper':
            size = (3.03209, 2.0)
            p += p9.guides(color=p9.guide_legend(ncol=3))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                legend_box_margin=0,
                legend_position='bottom',
                legend_background=p9.element_rect(fill='#F2F2F2')
            )
        else:
            raise ValueError('unknown format')

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.pdf',
               width=size[0], height=size[1], units='in')
