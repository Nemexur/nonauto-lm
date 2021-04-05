from typing import Dict, Any, Callable


class Filter:
    """Registry of filters for loggin Handlers."""

    _registry: Dict[str, Callable] = {}

    def __init__(self, type: str, condition: str) -> None:
        self._filter = Filter._registry[type]
        self._type = type
        self._condition = condition

    def __repr__(self) -> str:
        cls_name = str(self.__class__.__name__)
        return f"{cls_name}({self._name})"

    def __call__(self, record: Dict[str, Any]) -> bool:
        return self._filter(record, self._condition)


Filter._registry = {
    "defult": lambda record, condition: record["extra"].get("type") == condition,
    "has_attr": lambda record, condition: condition in record["extra"],
}
