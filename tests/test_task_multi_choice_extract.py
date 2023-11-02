
import pathlib

import pytest

from introspect.model import Llama2Model
from introspect.client import OfflineClient
from introspect.dataset import MCTestDataset
from introspect.tasks import MultiChoiceClassifyTask

@pytest.fixture
def task() ->MultiChoiceClassifyTask:
    return MultiChoiceClassifyTask(
        MCTestDataset(persistent_dir=pathlib.Path('.')),
        Llama2Model(OfflineClient())
    )

def test_task_multi_choice_extract_choice(task: MultiChoiceClassifyTask):
    c = task._extract_choice

    # Choice b) with matching content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is miami.') == 'Miami'

    # Choice b) without matching content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The answer is b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The answer is (b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], '(b)') == 'Miami'

    # Choice b) with only content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The city where Alyssa is in is miami.') == 'Miami'

    # Choice b) without with wrong content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Atlanta.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Atlanta.') == 'Miami'

    # Choice b) on another line
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The answer to the quetion is:\nb) The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The answer to the quetion is:\nb) The city where Alyssa is in is Miami.\n') == 'Miami'
