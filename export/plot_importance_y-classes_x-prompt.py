
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd
import plotnine as p9

from introspect.dataset import datasets
from introspect.types import DatasetSplits, SystemMessage, TaskCategories
from introspect.util import generate_experiment_id
from introspect.plot import annotation

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
parser.add_argument('--model-name',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
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
                    default=DatasetSplits.TEST,
                    type=DatasetSplits,
                    choices=list(DatasetSplits),
                    help='The dataset split to evaluate on')
parser.add_argument('--task',
                    action='store',
                    default=TaskCategories.IMPORTANCE,
                    type=TaskCategories,
                    choices=list(TaskCategories),
                    help='Which task to run')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed used for generation')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args = parser.parse_args()

    experiment_id = generate_experiment_id('explain_classes',
        model=args.model_name, system_message=args.system_message,
        dataset=args.dataset, split=args.split,
        task=args.task,
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


            if data['args']['model_name'] == args.model_name and \
               data['args']['system_message'] == args.system_message and \
               data['args']['dataset'] == args.dataset and \
               data['args']['split'] == args.split and \
               data['args']['task'] == args.task:
                data['plot'] = {'persona': 'no-persona'}
                if 'e-persona-human' in data['args']['task_config']:
                    data['plot']['persona'] = 'human-persona'
                elif 'e-persona-you' in data['args']['task_config']:
                     data['plot']['persona'] = 'you-persona'
                results.append(data)

        # Convert results into a flat DataFrame
        df = pd.json_normalize(results).explode('results.answer', ignore_index=True)
        results_answer = pd.json_normalize(list(df.pop('results.answer'))).add_prefix('results.answer.')
        df = pd.concat([df, results_answer], axis=1)

        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
        df = df.groupby([
            'args.model_name', 'args.system_message',
            'args.task', 'plot.persona',
            'args.dataset', 'args.split',
            'args.seed',
            'results.answer.explain_predict', 'results.answer.predict'],
            as_index=False
        ).agg({
            'results.answer.count': 'sum'
        })

        p = (
            p9.ggplot(df, p9.aes(x='results.answer.predict')) +
            p9.geom_bar(p9.aes(y='results.answer.count', fill='results.answer.explain_predict'), stat="identity") +
            p9.facet_grid('plot.persona ~ .', labeller=(annotation.persona).labeller) + # type: ignore
            p9.scale_y_continuous(
                name='Count'
            ) +
            p9.scale_x_discrete(
                breaks=annotation.predicted_sentiment.breaks,
                labels=annotation.predicted_sentiment.labels,
                name='Predicted sentiment'
            ) +
            p9.scale_fill_discrete(
                breaks=annotation.predicted_sentiment.breaks,
                labels=annotation.predicted_sentiment.labels,
                aesthetics=["fill"],
                name='Predicted sentiment of explanation'
            ) +
            p9.ggtitle(f'{args.dataset} - Importance')
        )

        if args.format == 'paper':
            size = (3.03209, 4.5)
            p += p9.guides(fill=p9.guide_legend(ncol=2))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                legend_box_margin=0,
                legend_position='bottom',
                legend_background=p9.element_rect(fill='#F2F2F2'),
                axis_text_x=p9.element_text(angle = 60, hjust=1)
            )
        else:
            raise ValueError('unknown format')

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.pdf',
               width=size[0], height=size[1], units='in')
