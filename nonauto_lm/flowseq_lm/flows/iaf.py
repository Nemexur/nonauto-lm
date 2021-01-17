from .flow import Flow


# TODO: Work in progress
@Flow.register("iaf")
class IAFlow(Flow):
    def __init__(self, input_size: int) -> None:
        super().__init__()
