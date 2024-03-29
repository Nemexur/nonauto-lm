from .application import Application
from vae_lm.utils import setup_logging


def main():
    setup_logging()
    return Application(prog="vae-lm").run()
