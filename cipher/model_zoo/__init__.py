from . import cnn_dist  # noqa: F401
from . import cnn_local  # noqa: F401
from . import deepbind  # noqa: F401
from . import deepbind_custom  # noqa: F401
from . import residualbind  # noqa: F401


def get_model(name: str):
    """Return function that can be used to instantiate a model architecture.

    Parameters
    ----------
    name : str
        Name of the model architecture.

    Returns
    -------
    Callable
        Function that can be used to instantiate a model architecture.
    """
    models = {
        "cnn_dist": cnn_dist.model,
        "cnn_local": cnn_local.model,
        "deepbind": deepbind.model,
        "deepbind_custom": deepbind_custom.model,
        "residualbind": residualbind.model,
    }
    try:
        return models[name.lower()]
    except KeyError:
        ms = "', '".join(models.keys())
        raise KeyError(f"Unknown model name '{name}'. Available models are '{ms}'.")
