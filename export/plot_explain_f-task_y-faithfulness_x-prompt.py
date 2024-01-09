
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
                    default=[TaskCategories.COUNTERFACTUAL, TaskCategories.IMPORTANCE, TaskCategories.REDACTED],
                    type=TaskCategories,
                    choices=list(TaskCategories),
                    help='Which tasks to run')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Seed used for generation')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args = parser.parse_args()

    experiment_id = generate_experiment_id('explain_f-task_y-faithfulness_x-model',
        model=args.model_name, system_message=args.system_message,
        dataset=''.join(d[0] for d in args.datasets), split=args.split,
        task=''.join(t[0] for t in args.tasks),
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
               data['args']['dataset'] in args.datasets and \
               data['args']['split'] == args.split and \
               data['args']['task'] in args.tasks:
                if data['results']['error'] > 0 or data['results']['missmatch'] > 0:
                    tqdm.write(f'Detected error ({data["results"]["error"]}) or missmatch ({data["results"]["missmatch"]}) in {file.name}')

                data['plot'] = {
                    'x-prompt': None,
                    'persona': tag.explain_persona(data['args']['task_config'])
                }

                match data['args']['task']:
                    case 'counterfactual':
                        data['plot']['x-prompt'] = tag.explain_counterfactual_target(data['args']['task_config'])
                    case 'importance' | 'redacted':
                        data['plot']['x-prompt'] = tag.explain_redact(data['args']['task_config'])

                if None not in data['plot'].values():
                    results.append(data)


        # Convert results into a flat DataFrame
        df = pd.json_normalize(results)
        os.makedirs(args.persistent_dir / 'pandas', exist_ok=True)
        df.to_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))

        print(df.loc[:, ['args.dataset', 'plot.x-prompt', 'plot.persona', 'results.faithful', 'results.faithful_and_correct']])

    if args.stage in ['both', 'plot']:
        df = pd.read_parquet((args.persistent_dir / 'pandas' / experiment_id).with_suffix('.parquet'))
        df = df.assign(**{
          'plot.faithfulness': df.loc[:, 'results.faithful_and_correct'] / df.loc[:, 'results.correct']
        })

        def fn(breaks):
            print(breaks)
            return breaks

        p = (
            p9.ggplot(df, p9.aes(x='plot.x-prompt')) +
            p9.geom_bar(p9.aes(y='plot.faithfulness', fill='plot.persona'), stat="identity", position="dodge") + # type: ignore
            p9.facet_grid('args.dataset ~ args.task', scales='free_x', labeller=annotation.explain_task.labeller) + # type: ignore
            p9.scale_y_continuous(
                name='Faithfulness',
                limits=[0, 1]
            ) +
            p9.scale_x_discrete(
                name='Explanation dependent prompt variation',
                labels=(annotation.redact_token | annotation.counterfactual_target).labels_callable
            ) +
            p9.scale_fill_discrete(
                breaks=annotation.persona.breaks,
                labels=annotation.persona.labels,
                aesthetics=["fill"],
                name='Persona instruction'
            )
        )

        if args.format == 'paper':
            size = (3.03209, 4.0)
            p += p9.guides(fill=p9.guide_legend(ncol=3))
            p += p9.theme(
                text=p9.element_text(size=10, fontname='Times New Roman'),
                legend_box_margin=0,
                legend_position='bottom',
                legend_background=p9.element_rect(fill='#F2F2F2'),
                axis_text_x=p9.element_text(angle = 60)
            )
        else:
            raise ValueError('unknown format')

        os.makedirs(args.persistent_dir / 'plots' / args.format, exist_ok=True)
        p.save(args.persistent_dir / 'plots'/ args.format / f'{experiment_id}.pdf',
               width=size[0], height=size[1], units='in')
