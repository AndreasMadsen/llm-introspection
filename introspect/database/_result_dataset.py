
from traceback import format_exception
from pathlib import Path
from typing import Generic, TypeVar, Type
import pickle
import inspect
import typing
import types
from functools import cached_property

from ..types import TaskCategories, DatasetSplits, OfflineError, TaskResult, GenerateError
from ._abstract_dataset import AbstractDatabase

_split_to_id = { DatasetSplits.TRAIN: 0, DatasetSplits.VALID: 1, DatasetSplits.TEST: 2 }

def _idx_split_to_rowid(split: DatasetSplits, idx: int) -> int:
   return idx * 3 + _split_to_id[split]

def _to_bool(value: int|None) -> bool|None:
    if value is None:
        return None
    return bool(value)

def _simplify_type(type_def) -> Type[float]|Type[int]|Type[bool]|Type[str]:
    # if the type is already simple, stop
    if type_def in (float, int, bool, str):
        return type_def

    # check for Literal['A', 'B']
    elif typing.get_origin(type_def) is typing.Literal:
        options = typing.get_args(type_def)
        if not all(isinstance(option, str) for option in options):
            raise ValueError('unsupported type')
        return str
    else:
        raise ValueError('unsupported type')

TaskResultType = TypeVar('TaskResultType', bound=TaskResult)

class ResultDatabase(AbstractDatabase, Generic[TaskResultType]):
    """Create Database to store results

    This class uses a TypedDict (TaskResultType) to define the database schema. All SQL
    queries are templated based on this TypedDict.
    """
    task: TaskCategories

    _result_type: Type[TaskResultType]
    _table_name: str

    @cached_property
    def _table_def(self) -> dict[str, Type[str]|Type[bool]|Type[int]|Type[float]]:
        all_table_types: dict[str, Type[str]|Type[bool]|Type[int]|Type[float]] = dict()

        for property_name, type_def in inspect.get_annotations(self._result_type).items():
            if typing.get_origin(type_def) is not typing.Required:
                raise ValueError(f'The TypedDict\'s {property_name} type must be Required')

            type_def, = typing.get_args(type_def)

            # Unpack Optional[]
            if typing.get_origin(type_def) in [typing.Union, types.UnionType]:
                type_options = typing.get_args(type_def)

                if len(type_options) > 2 or type(None) not in type_options:
                    raise ValueError(f'The TypedDict\'s {property_name} type must be "Type | None"')

                type_def = type_options[1] if type_options[0] is None else type_options[0]

            # Get primitive type
            all_table_types[property_name] = _simplify_type(type_def)

        return all_table_types

    @cached_property
    def _setup_sql(self) -> str:
        sql = (
            f'CREATE TABLE IF NOT EXISTS {self._table_name} (\n'
            'id INTEGER NOT NULL PRIMARY KEY,\n'
            'idx INTEGER NOT NULL,\n'
            'split INTEGER NOT NULL,\n'
        )

        for property_name, type_def in self._table_def.items():
            if type_def is float:
                sql_type = 'REAL'
            elif type_def is int or type_def is bool:
                sql_type = 'INTEGER'
            elif type_def is str:
                sql_type = 'TEXT'
            else:
                raise ValueError('missing translation from python type to sqlite type')
            sql += f'{property_name} {sql_type},\n'

        sql += (
            'error BLOB,\n'
            'traceback TEXT\n'
            ') STRICT'
        )

        return sql

    @cached_property
    def _put_error_sql(self) -> str:
        return (
            f'REPLACE INTO {self._table_name}(id, idx, split, error, traceback)\n'
            'VALUES (:rowid, :idx, :split, :error, :traceback)\n'
        )

    @cached_property
    def _put_obs_sql(self) -> str:
        sql_columns = ', '.join(self._table_def.keys())
        sql_values = ', '.join(map(lambda c: f':{c}', self._table_def.keys()))

        return (
            f'REPLACE INTO {self._table_name}(id, idx, split, {sql_columns})\n'
            f'VALUES (:rowid, :idx, :split, {sql_values})'
        )

    async def put(self, split: DatasetSplits, idx: int, data: TaskResultType|GenerateError) -> None:
        """Add or update an entry to the database

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index
            data (TaskResultType): data to add, input is a dictionary.
                The index of the observation is identified by the idx property.
        """
        rowid = _idx_split_to_rowid(split, idx)

        match data:
            case OfflineError():
                # There is information value in saving an OfflineError
                return

            case GenerateError():
                await self._con.execute(self._put_error_sql, {
                    'error': pickle.dumps(data),
                    'traceback': ''.join(format_exception(data)),
                    'split': _split_to_id[split],
                    'idx': idx,
                    'rowid': rowid
                })

            case Exception():
                raise ValueError(
                    "data['error'] is an error, but not a GenerationError"
                ) from data['error'] # type:ignore

            case _:
                await self._con.execute(self._put_obs_sql, {
                    **data,
                    'split': _split_to_id[split],
                    'idx': idx,
                    'rowid': rowid
                })

        self._transactions_queued += 1
        self._maybe_commit()

    @cached_property
    def _has_sql(self):
        return f'SELECT EXISTS(SELECT 1 FROM {self._table_name} WHERE id = ?)'

    async def has(self, split: DatasetSplits, idx: int) -> bool:
        """Check if observation exists

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index

        Returns:
            bool: True if the observation exists.
        """
        rowid = _idx_split_to_rowid(split, idx)
        cursor = await self._con.execute(self._has_sql, (rowid, ))
        exists, = await cursor.fetchone() # type: ignore
        return exists == 1

    @cached_property
    def _get_sql(self) -> str:
        sql_columns = ', '.join((*self._table_def.keys(), 'error'))
        return (
            f'SELECT {sql_columns}\n'
            f'FROM {self._table_name}\n'
            f'WHERE id = ?'
        )

    async def get(self, split: DatasetSplits, idx: int) -> TaskResultType|GenerateError|None:
        """Get entry by index

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index

        Returns:
            TaskResultType|None: Returns the entry if it exists.
                Otherwise, return None.
        """
        rowid = _idx_split_to_rowid(split, idx)
        cursor = await self._con.execute(self._get_sql, (rowid, ))
        results = await cursor.fetchone()
        if results is None:
            return None

        # error is set
        if results[-1] is not None:
            return pickle.loads(results[-1])

        # unpack
        return {
            property_name: _to_bool(value) if type_def is bool else value
            for value, (property_name, type_def)
            in zip(results, self._table_def.items())
        } # type:ignore
