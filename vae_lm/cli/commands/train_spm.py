from typing import Dict
import re
import json
import shutil
from pathlib import Path
from loguru import logger
import sentencepiece as spm
from torch_nlp_utils.common import Params
from cleo import option, argument, Command


class TrainSentencePieceModelCommand(Command):
    name = "train-spm"
    description = "Train Sentence Piece Model for BPE encoding."
    arguments = [argument("config", description="Config to use for SPM training.")]
    options = [
        option(
            "serialization-dir",
            "s",
            description="Directory to save trained SPM Model.",
            flag=False,
            value_required=False,
        ),
        option(
            "extra-vars",
            None,
            description=(
                "Extra variables to inject in JsonNet config in such format: "
                "{key_name1}={new_value1},{key_name2}={new_value2},..."
            ),
            flag=False,
            value_required=False,
        ),
    ]

    def handle(self) -> None:
        extra_vars = self.parse_extra_vars()
        config = Params.from_file(self.argument("config"), ext_vars=extra_vars)
        # Add serialization directory to config and create it
        serialization_dir = Path(self.option("serialization-dir"))
        if serialization_dir.exists():
            confirmed = self.confirm(
                f"Serialization directory at path `{serialization_dir}` already exists. Delete it?",
                default=True,
            )
            if confirmed:
                shutil.rmtree(serialization_dir)
                self.line("Directory successfully deleted!", style="info")
                serialization_dir.mkdir(exist_ok=False)
            else:
                self.add_style("warning", fg="yellow", options=["bold"])
                self.line(
                    "Working with current serialization directory then. "
                    "Probably you know what you are doing.",
                    style="warning",
                )
        else:
            serialization_dir.mkdir(exist_ok=False)
        # Log config to console and save
        config["model_prefix"] = str(serialization_dir / config["model_prefix"])
        logger.info(
            "Config: {}".format(json.dumps(config.as_flat_dict(), indent=2, ensure_ascii=False))
        )
        with (serialization_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config.as_dict(quiet=True), file, indent=2, ensure_ascii=False)
        # Train SPM Model
        spm.SentencePieceTrainer.train(**config)
        self.line(
            f"Finished SentencePiece model training and saved at path: `{serialization_dir}`",
            style="info",
        )

    def parse_extra_vars(self) -> Dict[str, str]:
        extra_vars = self.option("extra-vars")
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9\_\-\.\+\\\/]+)"
        return (
            {param: value for param, value in re.findall(regex, extra_vars, flags=re.I)}
            if extra_vars is not None else None
        )
