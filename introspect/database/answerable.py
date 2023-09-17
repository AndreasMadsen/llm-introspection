
import pickle

from ..types import AnswerableResult
from ._result_dataset import ResultDatabase

def to_bool(value: int|None) -> bool|None:
    if value is None:
        return None
    return bool(value)

class Answerable(ResultDatabase[AnswerableResult]):
    _setup_sql = f'''
        CREATE TABLE IF NOT EXISTS Answerable (
            id INTEGER NOT NULL PRIMARY KEY,
            idx INTEGER NOT NULL,
            split INTEGER NOT NULL,
            answer_ability TEXT,
            answer_sentiment TEXT,
            introspect INTEGER,
            correct INTEGER,
            duration REAL,
            error BLOB,
            traceback TEXT
        )
    '''
    _put_error_sql = '''
        REPLACE INTO Answerable(id, idx, split, answer_ability, answer_sentiment, introspect, correct, duration, error, traceback)
        VALUES (:rowid, :idx, :split, NULL, NULL, NULL, NULL, NULL, :error, :traceback)
    '''
    _put_obs_sql = '''
        REPLACE INTO Answerable(id, idx, split, answer_ability, answer_sentiment, introspect, correct, duration, error, traceback)
        VALUES (:rowid, :idx, :split, :answer_ability, :answer_sentiment, :introspect, :correct, :duration, NULL, NULL)
    '''
    _has_sql = '''
        SELECT EXISTS(SELECT 1 FROM Answerable WHERE id = ?)
    '''
    _get_sql = '''
        SELECT answer_ability, answer_sentiment, introspect, correct, duration, error
        FROM Answerable
        WHERE id = ?
    '''

    def _get_unpack(self,
                    answer_ability: str|None, answer_sentiment: str|None,
                    introspect: int|None, correct: int|None,
                    duration: float) -> AnswerableResult:
        return {
            'answer_ability': answer_ability,
            'answer_sentiment': answer_sentiment,
            'introspect': to_bool(introspect),
            'correct': to_bool(correct),
            'duration': duration
        }
