# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The bAbI tasks dataset."""

from dataclasses import dataclass
import datasets
import os.path as path

_CITATION = """\
@misc{weston2015aicomplete,
      title={Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks},
      author={Jason Weston and Antoine Bordes and Sumit Chopra and Alexander M. Rush and Bart van Merriënboer and Armand Joulin and Tomas Mikolov},
      year={2015},
      eprint={1502.05698},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
"""

_DESCRIPTION = """\
The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
"""

_PATHS = {
    "qa9": {
        "test": "tasks_1-20_v1-2/en-10k/qa9_simple-negation_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa9_simple-negation_train.txt",
    },
    "qa4": {
        "train": "tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_test.txt",
    },
    "qa6": {
        "train": "tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt",
    },
    "qa11": {
        "test": "tasks_1-20_v1-2/en-10k/qa11_basic-coreference_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa11_basic-coreference_train.txt",
    },
    "qa3": {
        "test": "tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt",
    },
    "qa15": {
        "test": "tasks_1-20_v1-2/en-10k/qa15_basic-deduction_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa15_basic-deduction_train.txt",
    },
    "qa17": {
        "test": "tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_train.txt",
    },
    "qa13": {
        "test": "tasks_1-20_v1-2/en-10k/qa13_compound-coreference_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa13_compound-coreference_train.txt",
    },
    "qa1": {
        "train": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt",
    },
    "qa14": {
        "train": "tasks_1-20_v1-2/en-10k/qa14_time-reasoning_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa14_time-reasoning_test.txt",
    },
    "qa16": {
        "test": "tasks_1-20_v1-2/en-10k/qa16_basic-induction_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa16_basic-induction_train.txt",
    },
    "qa19": {
        "test": "tasks_1-20_v1-2/en-10k/qa19_path-finding_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa19_path-finding_train.txt",
    },
    "qa18": {
        "test": "tasks_1-20_v1-2/en-10k/qa18_size-reasoning_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa18_size-reasoning_train.txt",
    },
    "qa10": {
        "train": "tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_test.txt",
    },
    "qa7": {
        "train": "tasks_1-20_v1-2/en-10k/qa7_counting_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa7_counting_test.txt",
    },
    "qa5": {
        "test": "tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_train.txt",
    },
    "qa12": {
        "test": "tasks_1-20_v1-2/en-10k/qa12_conjunction_test.txt",
        "train": "tasks_1-20_v1-2/en-10k/qa12_conjunction_train.txt",
    },
    "qa2": {
        "train": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_test.txt",
    },
    "qa20": {
        "train": "tasks_1-20_v1-2/en-10k/qa20_agents-motivations_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa20_agents-motivations_test.txt",
    },
    "qa8": {
        "train": "tasks_1-20_v1-2/en-10k/qa8_lists-sets_train.txt",
        "test": "tasks_1-20_v1-2/en-10k/qa8_lists-sets_test.txt",
    }
}

@dataclass
class LocalBabiConfig(datasets.BuilderConfig):
    task_no: int = -1


class LocalBabiDataset(datasets.GeneratorBasedBuilder):
    """The bAbI QA (20) tasks Dataset"""
    config: LocalBabiConfig

    VERSION = datasets.Version("1.2.0")

    BUILDER_CONFIG_CLASS = LocalBabiConfig

    BUILDER_CONFIGS = [
        LocalBabiConfig(
            task_no=task_no,
            name=f'en-10k-qa{task_no}',
            description=f"The 'qa{task_no}' task of the bAbI 'en-10k' dataset",
        )
        for task_no in range(1, 21)
    ]

    def _info(self):
        features = datasets.Features({
            "paragraph": datasets.Value("string"),
            "question": datasets.Value("string"),
            "choices": datasets.Sequence(datasets.Value("string")),
            "label": datasets.Value("string"),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://research.fb.com/downloads/babi/",
            license="Creative Commons Attribution 3.0 License",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archive: str = dl_manager.download_and_extract(
            "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
        ) # type:ignore

        return [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={
                    "filepath": path.join(archive, _PATHS[f'qa{self.config.task_no}']["train"]),
                },
            ),

            datasets.SplitGenerator(
                name='test',
                gen_kwargs={
                    "filepath": path.join(archive, _PATHS[f'qa{self.config.task_no}']["test"]),
                },
            )
        ]

    def _generate_examples(self, filepath):
        def push_observations(story_id, story_observaions):
            choices = list(set(obs['label'] for obs in story_observaions))

            for paragraph_id, story_observaion in enumerate(story_observaions):
                yield f'{story_id}-{paragraph_id}', {
                    'paragraph': ' Then, '.join(story_observaion['paragraph']),
                    'question': story_observaion['question'],
                    'choices': choices,
                    'label': story_observaion['label'],
                }

        with open(filepath) as fp:
            story = []
            story_id = 0
            story_observaions = []

            for line in fp:
                line = line.strip()
                tid, line_tid_striped = line.split(' ', 1)
                line_data = line_tid_striped.split('\t')

                # Start of a new paragraph construction
                if tid == '1':
                    yield from push_observations(story_id, story_observaions)

                    story = []
                    story_observaions = []
                    story_id += 1

                # paragraph component
                if len(line_data) == 1:
                    story.append(line_data[0].strip())
                # question
                else:
                    story_observaions.append({
                        'paragraph': story[:],
                        'question': line_data[0].strip(),
                        "label": line_data[1].strip()
                    })

            yield from push_observations(story_id, story_observaions)
