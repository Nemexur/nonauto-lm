from typing import Dict, List
import re
import json
import shutil
from pathlib import Path
from loguru import logger
import nonauto_lm.ddp as ddp
from _jsonnet import evaluate_file
from nonauto_lm.utils import archive_model
from cleo import option, argument, Command
from nonauto_lm.scripts.train_worker import train_worker
# Modules
import nonauto_lm.flowseq_lm  # noqa: F401


class TrainCommand(Command):
    name = "train"
    description = "Train FlowSeq model for unconditional text generation."
    arguments = [argument(name="config", description="Config to use for model training.")]
    options = [
        option(
            "serialization-dir",
            "s",
            description="Directory to save model",
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
        option(
            "cuda-devices",
            None,
            description=(
                "CUDA Devices to train model on in format: {gpu_idx}{gpu_idx}. "
                "Example: 012 means training model on 3 gpus."
            ),
            flag=False,
            value_required=False,
            default="-1",
        ),
    ]

    def handle(self) -> None:
        extra_vars = self.parse_extra_vars()
        config = json.loads(evaluate_file(self.argument("config"), ext_vars=extra_vars))
        # Add serialization directory to config and create it
        serialization_dir = Path(self.option("serialization-dir"))
        if serialization_dir.exists():
            confirmed = self.confirm(
                f"Serialization directory at path: {serialization_dir} already exists. Delete it?",
                default=True,
            )
            if confirmed:
                shutil.rmtree(serialization_dir)
                self.line("Directory successfully deleted!", style="info")
            else:
                self.add_style("warning", fg="yellow", options=["bold"])
                self.line(
                    "Working with current serialization directory then. "
                    "Probably you know what you are doing.",
                    style="warning",
                )
        serialization_dir.mkdir(exist_ok=False)
        config["serialization_dir"] = str(serialization_dir)
        # Log config to console and save
        logger.info(
            "Config: {}".format(json.dumps(config, indent=2, ensure_ascii=False)),
        )
        with (serialization_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
        # Get CUDA Devices
        config["cuda_devices"] = self.parse_cuda_devices()
        if len(config["cuda_devices"]) > 1:
            ddp.spawn(process=train_worker, args=config, world_size=len(config["cuda_devices"]))
        else:
            train_worker(process_rank=0, config=config, world_size=1)
        # Archive model
        archive_model(
            serialization_dir=serialization_dir,
            weights=serialization_dir / "models" / "best",
        )

    def parse_extra_vars(self) -> Dict[str, str]:
        extra = self.option("extra-vars")
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9\_\-\.\+\\\/]+)"
        return {param: value for param, value in re.findall(regex, extra, flags=re.I)}

    def parse_cuda_devices(self) -> List[int]:
        cuda = self.option("cuda-devices")
        if cuda == "-1":
            return [-1]
        return [int(cuda) for idx in list(cuda)]
