import pytest
import numpy as np

from cipher.model_zoo import cnn_dist
from cipher.model_zoo import cnn_local
from cipher.model_zoo import deepbind
from cipher.model_zoo import deepbind_custom
from cipher.model_zoo import get_model
from cipher.model_zoo import residualbind


@pytest.mark.parametrize(
    "name,model_fn,error",
    [
        ("cnn_dist", cnn_dist.model, None),
        ("cnn_local", cnn_local.model, None),
        ("deepbind", deepbind.model, None),
        ("deepbind_custom", deepbind_custom.model, None),
        ("residualbind", residualbind.model, None),
        ("fakemodel", None, KeyError),
    ],
)
def test_get_model(name: str, model_fn, error):
    if error is not None:
        with pytest.raises(error):
            get_model(name)
    else:
        assert get_model(name) is model_fn
        assert get_model(name.upper()) is model_fn


@pytest.mark.parametrize(
    "model_fn,x_shape",
    [
        pytest.param(cnn_dist.model, (100, 4), id="cnn_dist"),
        pytest.param(cnn_local.model, (100, 4), id="cnn_local"),
        pytest.param(deepbind.model, (10, 4), id="deepbind"),
        pytest.param(deepbind_custom.model, (10, 4), id="deepbind_custom"),
        pytest.param(residualbind.model, (10, 4), id="residualbind"),
    ],
)
def test_model_fit_required_args_only(model_fn, x_shape):
    rng = np.random.default_rng()
    # batch size 2, 24-nt sequence, alphabet length 4.
    x = rng.uniform(size=[2, *x_shape])
    # shape (2, 2). 2 sequences, 2 classes.
    y = np.array(
        [[0, 1], [1, 0]],
        dtype=np.float32,
    )
    model = model_fn(input_shape=x_shape, output_shape=2)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    model.fit(x, y)


@pytest.mark.xfail
def test_models_bad_args():
    # TODO: test that model funcs raise useful errors when passing invalid args.
    #   For example, error on bad length of list arguments.
    raise NotImplementedError()
