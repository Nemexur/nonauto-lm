import random
from tqdm import tqdm
from pathlib import Path
from .command import BaseCommand
from cleo import argument, option


class MakeShardsCommand(BaseCommand):
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
        self.prepare_directory(directory)
        with file_path.open("r", encoding="utf-8") as file:
            for line in tqdm(map(lambda x: x.strip(), file), desc="Sharding dataset"):
                choice = rng.randint(0, shards - 1)
                self.write_to_shard(line, directory / f"shard-{choice}.txt")

    @staticmethod
    def write_to_shard(line: str, shard_path: Path) -> None:
        with shard_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
