from .flow import Flow


# TODO: Work in progress
@Flow.register("maf")
class MAFlow(Flow):
    def __init__(self, input_size: int) -> None:
        super().__init__()
