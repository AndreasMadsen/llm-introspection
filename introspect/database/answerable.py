
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
            answer_ability TEXT NOT NULL,
            answer_sentiment TEXT NOT NULL,
            introspect BOOL,
            correct BOOL
        )
    '''
    _put_sql = '''
        REPLACE INTO Answerable(id, idx, split, answer_ability, answer_sentiment, introspect, correct)
        VALUES (:rowid, :idx, :split, :answer_ability, :answer_sentiment, :introspect, :correct)
    '''
    _has_sql = '''
        SELECT EXISTS(SELECT 1 FROM Answerable WHERE id = ?)
    '''
    _get_sql = '''
        SELECT answer_ability, answer_sentiment, introspect, correct
        FROM Answerable
        WHERE id = ?
    '''

    def _get_unpack(self, answer_ability, answer_sentiment, introspect, correct):
        return {
            'answer_ability': answer_ability,
            'answer_sentiment': answer_sentiment,
            'introspect': to_bool(introspect),
            'correct': to_bool(correct)
        }
