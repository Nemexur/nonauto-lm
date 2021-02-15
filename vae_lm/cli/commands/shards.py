import shutil
import random
from tqdm import tqdm
from pathlib import Path
from cleo import argument, option, Command


class MakeShardsCommand(Command):
    name = "make-shards"
    description = "Construct shards for Distributed Training from one txt file."
    arguments = [argument("path", description="Path to txt file.")]
    options = [
        option(
            "shards",
            "s",
            description="Number of shrads to split txt file.",
            flag=False,
            value_required=False,
        ),
        option(
            "seed",
            None,
            default=13,
            description="Random seed for to randomly assign sample to a shard.",
            flag=False,
            value_required=False,
        ),
    ]

    def handle(self) -> None:
        rng = random.Random(int(self.option("seed")))
        shards = int(self.option("shards"))
        file_path = Path(self.argument("path"))
        directory = file_path.parent / file_path.stem
        if directory.exists():
            confirmed = self.confirm(
                f"Directory at path `{directory}` already exists. Delete it?",
                default=True,
            )
            if confirmed:
                shutil.rmtree(directory)
                self.line("Directory successfully deleted!", style="info")
                directory.mkdir(exist_ok=False)
            else:
                self.add_style("warning", fg="yellow", options=["bold"])
                self.line(
                    "Working with current serialization directory then. "
                    "Probably you know what you are doing.",
                    style="warning",
                )
        else:
            directory.mkdir(exist_ok=False)
        with file_path.open("r", encoding="utf-8") as file:
            for line in tqdm(map(lambda x: x.strip(), file), desc="Sharding dataset"):
                choice = rng.randint(0, shards - 1)
                self.write_to_shard(line, directory / f"shard-{choice}.txt")

    @staticmethod
    def write_to_shard(line: str, shard_path: Path) -> None:
        with shard_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
