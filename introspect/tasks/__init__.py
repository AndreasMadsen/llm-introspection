
__all__ = ['SentimentTasks', 'SentimentTasks', 'tasks']

from typing import Type

from ._abstract_tasks import AbstractTasks

from .sentiment import SentimentTasks

tasks: dict[str, Type[AbstractTasks]] = {
    Tasks._category: Tasks
    for Tasks
    in [SentimentTasks]
}
