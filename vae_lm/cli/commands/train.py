from typing import Dict, List
import os
import re
import json
import wandb
import shutil
from pathlib import Path
from loguru import logger
import vae_lm.training.ddp as ddp
from torch_nlp_utils.common import Params
from cleo import option, argument, Command
from vae_lm.scripts.train_worker import train_worker


class TrainCommand(Command):
    name = "train"
    description = "Train FlowSeq model for unconditional text generation."
    arguments = [argument("config", description="Config to use for model training.")]
    options = [
        option(
            "serialization-dir",
            "s",
            description="Directory to save model",
            flag=False,
            value_required=True,
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
        option(
            "cuda-devices",
            None,
            description=(
                "CUDA Devices to train model on in format: {gpu_idx},{gpu_idx}. "
                "Example: 012 means training model on 3 gpus."
            ),
            flag=False,
            value_required=False,
            default="-1",
        ),
        option(
            "seed",
            None,
            description="Define random state for reproducibility.",
            flag=False,
            value_required=False,
            default="13",
        ),
        option(
            "use-wandb",
            None,
            description="Whether to log metrics to Weights&Biases or not.",
            flag=True,
            value_required=False,
        ),
        option(
            "tags",
            None,
            description=(
                "Tags for train run in Weights&Biases Dashboard. "
                "Considered only if `use-wandb` is set True."
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
        logger.info(
            "Config: {}".format(json.dumps(config.as_flat_dict(), indent=2, ensure_ascii=False))
        )
        with (serialization_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config.as_dict(quiet=True), file, indent=2, ensure_ascii=False)
        # Login to wandb there
        use_wandb = self.option("use-wandb")
        if use_wandb:
            wandb.login(key=os.getenv("WANDB_LOGIN_KEY"))
        # Add extra properties to config
        config["serialization_dir"] = str(serialization_dir)
        config["use_wandb"] = use_wandb
        config["cuda_devices"] = self.parse_cuda_devices()
        config["tags"] = self.parse_tags()
        config["seed"] = int(self.option("seed"))
        # Run train worker in distributed mode or not depending on cuda devices
        if len(config["cuda_devices"]) > 1:
            ddp.spawn(process=train_worker, args=config, world_size=len(config["cuda_devices"]))
        else:
            train_worker(process_rank=0, config=config, world_size=1)

    def parse_extra_vars(self) -> Dict[str, str]:
        extra_vars = self.option("extra-vars")
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9\_\-\.\+\\\/]+)"
        return (
            {param: value for param, value in re.findall(regex, extra_vars, flags=re.I)}
            if extra_vars is not None else None
        )

    def parse_cuda_devices(self) -> List[int]:
        cuda = self.option("cuda-devices")
        if cuda is None or cuda == "-1":
            return [-1]
        return [int(cuda) for idx in re.findall(r"\d+", cuda, flags=re.I)]

    def parse_tags(self) -> List[str]:
        tags = self.option("tags")
        regex = r"[a-z0-9\_\-\.\+\\\/\s]+"
        return re.findall(regex, tags, flags=re.I) if tags is not None else None
