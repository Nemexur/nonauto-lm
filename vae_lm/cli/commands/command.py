import shutil
from pathlib import Path
from cleo import Command


class BaseCommand(Command):
    def prepare_directory(self, directory: Path) -> None:
        """Safely create `directory` with confirmation from user if the directory already exists."""
        if directory.exists():
            confirmed = self.confirm(
                f"Serialization directory at path `{directory}` already exists. Delete it?",
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
