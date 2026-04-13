import torch
from uplifting_rallies.model import get_model
from uplifting_rallies.transformations import get_transforms


def load_model(model_path):
    """
    Load the uplifting model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        model (torch.nn.Module): Loaded uplifting model.
        transform (callable): Transformation function for input images.
    """
    loaded_dict = torch.load(model_path, weights_only=False)
    model_name = loaded_dict["additional_info"]["name"]
    model_size = loaded_dict["additional_info"]["size"]
    tabletoken_mode = loaded_dict["additional_info"]["tabletoken_mode"]
    time_rotation = loaded_dict["additional_info"]["time_rotation"]
    transform_mode = loaded_dict["additional_info"]["transform_mode"]
    randdet_prob, randmiss_prob, tablemiss_prob = (
        loaded_dict["additional_info"]["randdet_prob"],
        loaded_dict["additional_info"]["randmiss_prob"],
        loaded_dict["additional_info"]["tablemiss_prob"],
    )
    uplifting_model = get_model(
        model_name,
        size=model_size,
        mode=tabletoken_mode,
        time_rotation=time_rotation,
        interpolate_missing=False,
    )

    # Load state dict with strict=False to handle missing ankle embedding parameters
    missing_keys, unexpected_keys = uplifting_model.load_state_dict(
        loaded_dict["model_state_dict"], strict=False
    )

    uplifting_model.eval()
    transform = get_transforms(config=None, mode='test')
    print(
        f"Loaded Uplifting model: {model_name} with size {model_size}, tabletoken_mode: {tabletoken_mode}, time_rotation: {time_rotation}, transform_mode: {transform_mode}"
    )
    print(
        f"Noise settings during training - randdet_prob: {randdet_prob}, randmiss_prob: {randmiss_prob}, tablemiss_prob: {tablemiss_prob}"
    )
    return uplifting_model, transform, transform_mode