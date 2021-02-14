from typing import List
import re
import json
from pathlib import Path
from loguru import logger
from vae_lm.training.utils import load_archive, log_metrics
from cleo import option, argument, Command


class SampleCommand(Command):
    name = "sample"
    description = "Sample unconditional texts from NonAuto model."
    arguments = [argument("archive", description="Path to archive with trained model.")]
    options = [
        option(
            "num-samples",
            "s",
            description="Num latent samples to generate.",
            flag=False,
            value_required=False,
        ),
        option(
            "lengths",
            "l",
            description=(
                "Length for each latent sample. "
                "In such format: {number} or {number},{number},{number}..."
            ),
            flag=False,
            value_required=False,
        ),
        option(
            "plot-time",
            None,
            description=(
                "Whether to plot generation time or not."
            ),
            flag=True,
            value_required=False,
        ),
        option(
            "log-model-info",
            None,
            description=(
                "Whether to log model information like its config and metrics or not."
            ),
            flag=True,
            value_required=False,
        ),
    ]

    def handle(self) -> None:
        # Load archive
        archive = load_archive(Path(self.argument("archive")))
        if self.option("log-model-info"):
            # Log model config
            logger.info(
                "Config: {}".format(json.dumps(archive.config.as_flat_dict(), indent=2, ensure_ascii=False))
            )
            # Log model metrics
            log_metrics("Trained model", archive.metrics)
        num_samples = int(self.option("num-samples"))
        lengths = self.parse_lengths()
        samples, log_prob = archive.model.sample(num_samples, lengths)
        samples = archive.model.make_output_human_readable(samples)
        print(samples["texts"])

    def parse_lengths(self) -> List[int]:
        lengths = self.option("lengths")
        if lengths is None:
            return None
        return [int(x) for x in re.findall(r"\d+", lengths, flags=re.I)]
