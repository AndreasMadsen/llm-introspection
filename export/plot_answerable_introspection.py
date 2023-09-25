
import json
import argparse
import os
import pathlib

from tqdm import tqdm
import pandas as pd
import plotnine as p9
import numpy as np

from introspect.dataset import datasets
from introspect.types import DatasetSplits
from introspect.util import generate_experiment_id

def select_target_metric(df):
    idx, cols = pd.factorize('results.' + df.loc[:, 'target_metric'])
    return df.assign(
        metric = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    )

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
parser.add_argument('--datasets',
                    action='store',
                    nargs='+',
                    default=list(datasets.keys()),
                    choices=datasets.keys(),
                    type=str,
                    help='The datasets to plot')
parser.add_argument('--model-name',
                    action='store',
                    default=None,
                    type=str,
                    help='Model name')
parser.add_argument('--split',
                    action='store',
                    default=DatasetSplits.TRAIN,
                    type=DatasetSplits,
                    choices=list(DatasetSplits),
                    help='The dataset split to evaluate on')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args, unknown = parser.parse_known_args()

    experiment_id = generate_experiment_id('answerable_introspect', args.model_name, args.split)

    if args.stage in ['both', 'preprocess']:
        # Read JSON files into dataframe
        results = []
        files = sorted((args.persistent_dir / 'results' / 'answerable').glob('answerable_*.json'))
        for file in tqdm(files, desc='Loading answerable .json files'):
            try:
                with open(file, 'r') as fp:
                    data = json.load(fp)
            except Exception as error:
                raise Exception(f'{file} caused an error') from error

            if data['args']['split'] == args.split and \
                data['args']['model_name'] == args.model_name and \
                data['args']['dataset'] in args.datasets:
                results.append(data)

        # Convert results into a flat DataFrame
        df = pd.json_normalize(results).explode('results.answer', ignore_index=True)
        results_answer = pd.json_normalize(list(df.pop('results.answer'))).add_prefix('results.answer.')
        df = pd.concat([df, results_answer], axis=1)

        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

    if args.stage in ['both', 'plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

        p = (
            p9.ggplot(df, p9.aes(x='results.answer.sentiment')) +
            p9.geom_bar(p9.aes(y='results.answer.count', fill='results.answer.ability'), stat="identity")
        )

        if args.format == 'paper':
            size = (3.03209, 3.5)
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                legend_box_margin=0,
                legend_position='bottom',
                legend_background=p9.element_rect(fill='#F2F2F2'),
                axis_text_x=p9.element_text(angle = 60, hjust=1)
            )

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.pdf', width=size[0], height=size[1], units='in')
