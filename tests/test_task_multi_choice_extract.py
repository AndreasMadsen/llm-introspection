
import pathlib

import pytest

from introspect.model import Llama2Model
from introspect.client import OfflineClient
from introspect.dataset import MCTestDataset
from introspect.tasks import MultiChoiceClassifyTask

@pytest.fixture
def task() ->MultiChoiceClassifyTask:
    return MultiChoiceClassifyTask(
        Llama2Model(OfflineClient())
    )

def test_task_multi_choice_extract_choice(task: MultiChoiceClassifyTask):
    c = task._extract_choice

    """
    # Choice b) with matching content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is miami.') == 'Miami'

    # Choice b) without matching content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The answer is b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The answer is (b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], '(b)') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'Answer: b') == 'Miami'

    # Choice b) with only content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The city where Alyssa is in is miami.') == 'Miami'

    # Choice b) without with wrong content
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Atlanta.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'b) The city where Alyssa is in is Atlanta.') == 'Miami'

    # Choice b) on another line
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The answer to the quetion is:\nb) The city where Alyssa is in is Miami.') == 'Miami'
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The answer to the quetion is:\nb) The city where Alyssa is in is Miami.\n') == 'Miami'

    # Choice b) on another line with Answer: prefix
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'Where did Sandra go?\nAnswer: b') == 'Miami'

    # Choice b) but 'a' is a distraction
    assert c(['trip', 'Miami', 'Atlanta', 'beach'], 'The answer to a question is b)') == 'Miami'
    """

    # Falcon
    assert c(['trip', 'Miami', 'Atlanta', 'beach'],
             'Based on the given information, the correct response to the prompt is option C).') == 'Atlanta'
    assert c(['bird', 'friend', 'butterfly', 'dog'],
             'It seems like your granddaughter tried to catch a butterfly while playing in the park with her friend Mary.') == 'butterfly'
    assert c(['Carrots', 'Hamburgers', 'Hotdogs', 'Salad'],
             'Based on the given information, it seems that you ate two hamburgers.')