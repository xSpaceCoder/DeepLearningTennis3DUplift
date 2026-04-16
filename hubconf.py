dependencies = [
    "torch",
    "numpy",
    "einops",
    "torch.utils.tensorboard.summary",
    "hparams",
    "random",
    "os",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_agg",
]

from torchhub.interface_rallies_dynamicAnkle import (
    UpliftingModel as UpliftingModel_r_dynamicAnkle,
)
from torchhub.interface_rallies_dynamic import (
    UpliftingModel as UpliftingModel_r_dynamic,
)
from torchhub.interface_trajectories_dynamic import (
    UpliftingModel as UpliftingModel_t_dynamic,
)
from torchhub.interface_trajectories_dynamicAnkle import (
    UpliftingModel as UpliftingModel_t_dynamicAnkle,
)


def tennis_uplifting(input_type="rallies", mode="dynamicAnkle", **kwargs):
    """
    Loads the Tennis Uplifting Model.
    Available iput_types: 'trajectories', 'rallies'
    Available modes: 'dynamic', 'dynamicAnkle'
    """
    if input_type == "rallies":
        if mode == "dynamicAnkle":
            return UpliftingModel_r_dynamicAnkle()
        elif mode == "dynamic":
            return UpliftingModel_r_dynamic()
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for input_type 'rallies'. Available modes: 'dynamic', 'dynamicAnkle'"
            )
    elif input_type == "trajectories":
        if mode == "dynamic":
            return UpliftingModel_t_dynamic()
        elif mode == "dynamicAnkle":
            return UpliftingModel_t_dynamicAnkle()
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for input_type 'trajectories'. Available modes: 'dynamic', 'dynamicAnkle'"
            )
    else:
        raise ValueError(
            f"Invalid input_type '{input_type}'. Available input_types: 'rallies', 'trajectories'"
        )
