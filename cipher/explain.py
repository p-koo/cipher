"""This module implements model interpretability methods."""

import numpy as np
import tensorflow as tf


class Explainer:
    """wrapper class for attribution maps"""

    def __init__(self, model, class_index=None, func=tf.math.reduce_mean):

        self.model = model
        self.class_index = class_index
        self.func = func

    def saliency_maps(self, X, batch_size=128):

        return function_batch(
            X,
            saliency_map,
            batch_size,
            model=self.model,
            class_index=self.class_index,
            func=self.func,
        )

    def smoothgrad(self, X, num_samples=50, mean=0.0, stddev=0.1):

        return function_batch(
            X,
            smoothgrad,
            batch_size=1,
            model=self.model,
            num_samples=num_samples,
            mean=mean,
            stddev=stddev,
            class_index=self.class_index,
            func=self.func,
        )

    def integrated_grad(self, X, baseline_type="random", num_steps=25):

        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            baseline = self.set_baseline(x, baseline_type, num_samples=1)
            intgrad_scores = integrated_grad(
                x,
                model=self.model,
                baseline=baseline,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
            )
            scores.append(intgrad_scores)
        return np.concatenate(scores, axis=0)

    def expected_integrated_grad(
        self, X, num_baseline=25, baseline_type="random", num_steps=25
    ):

        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            baselines = self.set_baseline(x, baseline_type, num_samples=num_baseline)
            intgrad_scores = expected_integrated_grad(
                x,
                model=self.model,
                baselines=baselines,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
            )
            scores.append(intgrad_scores)
        return np.concatenate(scores, axis=0)

    def mutagenesis(self, X, class_index=None):
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            scores.append(mutagenesis(x, self.model, class_index))
        return np.concatenate(scores, axis=0)

    def set_baseline(self, x, baseline, num_samples):
        if baseline == "random":
            baseline = random_shuffle(x, num_samples)
        else:
            baseline = np.zeros((x.shape))
        return baseline


# @tf.function
def saliency_map(x, model, class_index=None, reducer=tf.math.reduce_mean):
    """Calculate saliency map of input `x` given a TensorFlow Keras model.

    Parameters
    ----------
    x : tf.Tensor
        Inputs to model.
    model : tf.keras model
        Any tf.keras model (eg, functional, sequential, sub-classed).
    class_index : int, optional
        The class over which to calculate the saliency map. If omitted, reduces model
        outputs using `reducer` and then gets saliency map.
    reducer : callable, optional
        Callable that takes in a sequence of values and reduces them to one value.

    Returns
    -------
    tf.Tensor
        Saliency map.
    """
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        outputs = model(x)
        if class_index is not None:
            outputs = outputs[:, class_index]
        else:
            outputs = reducer(outputs)
    return tape.gradient(outputs, x)


# @tf.function
def hessian(x, model, class_index: int = None, reducer=tf.math.reduce_mean):
    """Calculate Hessian.

    Parameters
    ----------
    x : tf.Tensor
        Inputs to model.
    model : tf.keras model
        Any tf.keras model (eg, functional, sequential, sub-classed).
    class_index : int, optional
        The class over which to calculate the Hessian. If omitted, reduces model
        outputs using `reducer` and then gets saliency map.
    reducer : callable, optional
        Callable that takes in a sequence of values and reduces them to one value.

    Returns
    -------
    tf.Tensor
        Hessian.
    """
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            outputs = model(x)
            if class_index is not None:
                outputs = outputs[:, class_index]
            else:
                outputs = reducer(outputs)
        g = tape1.gradient(outputs, x)
    return tape2.jacobian(g, x)


def smoothgrad(
    x,
    model,
    num_samples=50,
    mean=0.0,
    stddev=0.1,
    class_index=None,
    reducer=tf.math.reduce_mean,
):
    """Run smoothgrad on inputs and model.

    Parameters
    ----------
    x : tf.Tensor
        Inputs to model.
    model : tf.keras model
        Any tf.keras model (eg, functional, sequential, sub-classed).
    num_samples : int, optional
        Number of samples over which to get saliency maps.
    mean : float, optional
        Mean of Gaussian noise distribution added to the inputs.
    stddev : float, optional
        Standard deviation of Gaussian noise distribution added to the inputs.
    class_index : int, optional
        The class over which to get saliency maps. If omitted, then model outputs are
        reduced using `reducer`.
    reducer : callable, optional
        Callable that takes in a sequence of values and reduces them to one value.

    Returns
    -------
    tf.Tensor
        Smoothgrad results.
    """
    x = tf.convert_to_tensor(x)
    if x.ndim != 3:
        raise ValueError(f"Expected input to have 3 dimensions but got {x.ndim}")
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = saliency_map(x_noise, model, class_index=class_index, reducer=reducer)
    return tf.reduce_mean(grad, axis=0, keepdims=True)


def integrated_grad(
    x,
    model,
    baseline,
    num_steps=25,
    class_index: int = None,
    reducer=tf.math.reduce_mean,
):
    """Calculate integrated gradients on inputs and model.

    Parameters
    ----------
    x : tf.Tensor
        Inputs to model.
    model : tf.keras model
        Any tf.keras model (eg, functional, sequential, sub-classed).
    baseline : ???
    num_steps : int, optional
        ???
    class_index : int, optional
        The class over which to get saliency maps. If omitted, then model outputs are
        reduced using `reducer`.
    reducer : callable, optional
        Callable that takes in a sequence of values and reduces them to one value.

    Returns
    -------
    tf.Tensor
        Integrated gradients results.
    """

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def interpolate_data(baseline, x, steps):
        steps_x = steps[:, tf.newaxis, tf.newaxis]
        delta = x - baseline
        x = baseline + steps_x * delta
        return x

    x = tf.convert_to_tensor(x)
    steps = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)
    x_interp = interpolate_data(baseline, x, steps)
    grad = saliency_map(x_interp, model, class_index=class_index, reducer=reducer)
    avg_grad = integral_approximation(grad)
    return avg_grad * (x - baseline)


def expected_integrated_grad(
    x,
    model,
    baselines,
    num_steps=25,
    class_index: int = None,
    reducer=tf.math.reduce_mean,
):
    """Average integrated gradients across different backgrounds

    Parameters
    ----------

    Returns
    -------
    """

    grads = []
    for baseline in baselines:
        grads.append(
            integrated_grad(
                x,
                model,
                baseline,
                num_steps=num_steps,
                class_index=class_index,
                reducer=tf.math.reduce_mean,
            )
        )
    return np.mean(np.array(grads), axis=0)


def mutagenesis(x, model, class_index: int = None, batch_size=512):
    """In silico mutagenesis analysis for a given sequence.

    Parameters
    ----------
    x : tf.Tensor
        Inputs to model.
    model : tf.keras model
        Any tf.keras model (eg, functional, sequential, sub-classed).
    class_index : int, optional
        The class from which to keep scores.
    batch_size : int, optional
        Batch size for model prediction. From a few tests, it seems like batch sizes
        between 500 and 600 have optimal time performance.

    Returns
    -------
    Numpy array
        Mutant scores minus wildtype scores.
    """

    def generate_mutagenesis(x):
        _, L, A = x.shape
        x_mut = []
        for ll in range(L):
            for a in range(A):
                x_new = np.copy(x)
                x_new[0, ll, :] = 0
                x_new[0, ll, a] = 1
                x_mut.append(x_new)
        return np.concatenate(x_mut, axis=0)

    def reconstruct_map(predictions):
        _, L, A = x.shape

        mut_score = np.zeros((1, L, A))
        k = 0
        for ll in range(L):
            for a in range(A):
                mut_score[0, ll, a] = predictions[k]
                k += 1
        return mut_score

    def get_score(x, model, class_index):
        score = model.predict(x, batch_size=batch_size)
        if class_index is None:
            score = np.sqrt(np.sum(score ** 2, axis=-1, keepdims=True))
        else:
            score = score[:, class_index]
        return score

    # generate mutagenized sequences
    x_mut = generate_mutagenesis(x)

    # get baseline wildtype score
    wt_score = get_score(x, model, class_index)
    predictions = get_score(x_mut, model, class_index)

    # reshape mutagenesis predictiosn
    mut_score = reconstruct_map(predictions)

    return mut_score - wt_score


def grad_times_input(x, scores):
    new_scores = []
    for i, score in enumerate(scores):
        new_scores.append(np.sum(x[i] * score, axis=1))
    return np.array(new_scores)


def l2_norm(scores):
    """Calculate L2 norm on `scores`."""
    # TODO: if we want to stay in tensorflow-land, we can look into
    # https://www.tensorflow.org/api_docs/python/tf/norm
    return np.sum(np.sqrt(scores ** 2), axis=2)


def function_batch(x, fun, batch_size=128, **kwargs):
    """Run a function over `x` in batches.

    Parameters
    ----------
    x : array, tensor
        Input data over which to batch.
    fun : callable
        Function to call on batches of `x`.
    """

    # TODO: tf.data.Datasets have a maximum size of 2GB, so this might not work for
    # large arrays.
    dataset = tf.data.Dataset.from_tensor_slices(x)
    outputs = []
    for this_x in dataset.batch(batch_size):
        outputs.append(fun(this_x, **kwargs))
    return np.concatenate(outputs, axis=0)


def random_shuffle(x, num_samples=1):
    """Randomly shuffle sequences.

    This assumes x shape is `(N, L, A)`.
    """

    x_shuffle = []
    for i in range(num_samples):
        shuffle = np.random.permutation(x.shape[1])
        x_shuffle.append(x[0, shuffle, :])
    return np.array(x_shuffle)
