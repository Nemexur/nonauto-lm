from typing import List
from cleo import Command
from cleo import Application as BaseApplication
from .commands import (
    TrainCommand,
    EvaluateCommand,
    MakeShardsCommand,
    ItemSampleCommand,
    PriorSampleCommand,
    InterpolateSampleCommand,
    TrainSentencePieceModelCommand,
)


class Application(BaseApplication):
    def __init__(self, prog: str) -> None:
        super().__init__(prog)
        # Add commands
        for command in self.get_default_commands():
            self.add(command)

    def get_default_commands(self) -> List[Command]:
        return [
            TrainCommand(),
            EvaluateCommand(),
            MakeShardsCommand(),
            ItemSampleCommand(),
            PriorSampleCommand(),
            InterpolateSampleCommand(),
            TrainSentencePieceModelCommand(),
        ]
