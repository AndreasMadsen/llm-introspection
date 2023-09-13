
from ..types import AnswerableResult
from ._result_dataset import ResultDatabase

def to_bool(value: int|None) -> bool|None:
    if value is None:
        return None
    return bool(value)

class Answerable(ResultDatabase[AnswerableResult]):
    _setup_sql = '''
        CREATE TABLE IF NOT EXISTS Answerable (
            id INTEGER NOT NULL PRIMARY KEY,
            idx INTEGER NOT NULL,
            split INTEGER NOT NULL,
            answer_ability TEXT,
            answer_sentiment TEXT,
            introspect BOOL,
            correct BOOL,
            error TEXT
        ) STRICT
    '''
    _put_sql = '''
        REPLACE INTO Answerable(id, idx, split, answer_ability, answer_sentiment, introspect, correct, error)
        VALUES (:rowid, :idx, :split, :answer_ability, :answer_sentiment, :introspect, :correct, :error)
    '''
    _has_sql = '''
        SELECT EXISTS(SELECT 1 FROM Answerable WHERE id = ?)
    '''
    _get_sql = '''
        SELECT answer_ability, answer_sentiment, introspect, correct, error
        FROM Answerable
        WHERE id = ?
    '''

    def _get_unpack(self, answer_ability, answer_sentiment, introspect, correct, error):
        return {
            'answer_ability': answer_ability,
            'answer_sentiment': answer_sentiment,
            'introspect': to_bool(introspect),
            'correct': to_bool(correct),
            'error': error
        }
