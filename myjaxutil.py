from typing import NamedTuple
import optax
import jax


class TrainState(NamedTuple):
    """
    A class that stores the state of the training.

    Attributes:
    -----------
    optimizer: optax.TransformUpdateFn
        The optimizer function used for updating the parameters.
    state: optax.OptState
        The optimizer state.
    """

    update: optax.TransformUpdateFn
    state: optax.OptState


def init_optimiser(lr, params, name="adam", **kwargs):
    """
    Initializes an optimizer.

    Parameters:
    -----------
    lr: float
        The learning rate.
    params: Any
        The parameters to be optimized.
    name: str
        The name of the optimizer. Default is "adam".

    Returns:
    --------
    TrainState
        The state of the training.
    """
    if name == "sgd":
        optimizer = optax.sgd(lr, **kwargs)
    elif name == "adam":
        optimizer = optax.adam(lr, **kwargs)
    elif name == "rmsprop":
        optimizer = optax.rmsprop(lr, **kwargs)
    elif name == "optimistic_gradient_descent":
        optimizer = optax.optimistic_gradient_descent(lr, **kwargs)
    else:
        raise ValueError(f"Invalid optimizer name: {name}")

    opt_init, opt_update = optimizer
    opt_state = opt_init(params)
    return TrainState(opt_update, opt_state)


def minimize(
    f, x0, proj=lambda x: x, num_iters=1000, lr=1e-2, name="adam", verbose=True
):
    """
    Minimizes a function using an optimizer.

    Parameters:
    -----------
    f: Callable
        The function to be minimized.
    x0: Any
        The initial value of the parameters.
    proj: Callable
        The projection function. Default is the identity function.
    num_iters: int
        The number of iterations. Default is 1000.
    lr: float
        The learning rate. Default is 1e-2.
    name: str
        The name of the optimizer. Default is "adam".
    verbose: bool
        Whether to print the loss at each iteration. Default is True.

    Returns:
    --------
    Any
        The optimized parameters.
    """
    # Define the gradient function
    grad_f = jax.grad(f)

    # Define the optimizer
    opt_update, opt_state = init_optimiser(lr, x0, name)

    # Define the update function
    @jax.jit
    def update(x, opt_state):
        grads = grad_f(x)
        updates, new_state = opt_update(grads, opt_state, x)
        new_x = optax.apply_updates(x, updates)
        new_x = proj(new_x)
        return new_x, new_state

    # Minimize the function
    x = x0
    for i in range(num_iters):
        if verbose:
            if i % 100 == 0:
                print(f"Iteration {i}, loss {f(x):.5f}")
        x, opt_state = update(x, opt_state)

    return x


def maximize(
    f, x0, proj=lambda x: x, num_iters=1000, lr=1e-2, name="adam", verbose=True
):
    """
    Maximizes a function using an optimizer.

    Parameters:
    -----------
    f: Callable
        The function to be maximized.
    x0: Any
        The initial value of the parameters.
    proj: Callable
        The projection function. Default is the identity function.
    num_iters: int
        The number of iterations. Default is 1000.
    lr: float
        The learning rate. Default is 1e-2.
    name: str
        The name of the optimizer. Default is "adam".
    verbose: bool
        Whether to print the loss at each iteration. Default is True.

    Returns:
    --------
    Any
        The optimized parameters.
    """
    return minimize(
        lambda x: -f(x),
        x0,
        proj=lambda x: x,
        num_iters=1000,
        lr=1e-2,
        name="adam",
        verbose=True,
    )
