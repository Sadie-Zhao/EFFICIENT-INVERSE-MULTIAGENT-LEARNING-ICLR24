"""
This module contains classes and functions for implementing a stochastic Fisher market. 

Classes:
- Trajectory: A trajectory storage.
- Policy: A class that stores network and parameters, and applies the policy to a state.
- StochasticFisher: A class representing a stochastic Fisher market.

Functions:
- init_optimiser: Initializes an optimizer for training.
"""
# TODO STATE BECOMES ONLY BUDGET AND EVERYHTING ELSE BECOMES MARKET PARAMETER
from functools import partial
from typing import Callable
import haiku as hk
import jax
import optax
import myjaxutil as myjax
import jax.numpy as jnp
from flax import struct
from typing import Dict


@struct.dataclass
class Trajectory:
    """A trajectory storage.
    Attributes:
        states: States of the trajectory
        actions: Actions of the trajectory
        rewards: Rewards of the trajectory
    """

    s: Dict[str, jnp.ndarray]
    a: jnp.ndarray
    b: jnp.ndarray
    r: jnp.ndarray = None

    def get_transition(self, t):
        """Gets the SA(B)R tuple at time t.
        Args:
            t: The time step.
        Returns:
            The transition at time t.
        """
        return jax.tree_util.tree_map(lambda x: x[t], self)


@struct.dataclass
class StochasticFisher:
    """
    A class representing a stochastic Fisher market.

    Attributes:
    -----------
    utils : Callable
        A function that takes in allocations and types and returns utilities.
    br_network : hk.Transformed
        A Haiku network representing the best-response function.
    br_params : hk.Params
        The parameters of the best-response network.
    """

    utils: Callable = struct.field(pytree_node=False)
    market_actor_network: hk.Transformed = struct.field(pytree_node=False)
    br_network: hk.Transformed = struct.field(pytree_node=False)
    world_states: jnp.ndarray
    seed: int = 42

    @jax.jit
    def reward(self, market_params, state, prices, allocations, savings):
        """
        Computes the reward for a given state, prices, allocations, and savings.

        Parameters:
        -----------
        state : dict
            A dictionary containing the state of the market.
        prices : jnp.ndarray
            An array containing the prices of the goods.
        allocations : jnp.ndarray
            An array containing the allocations of the goods.
        savings : jnp.ndarray
            An array containing the savings of the buyers.

        Returns:
        --------
        The reward for the given state, prices, allocations, and savings.
        """
        supplies = state["supplies"]
        types = state["types"]
        budgets = state["budgets"]
        discount = market_params["discount"]

        u = self.utils(allocations, types)
        spending = jnp.clip(budgets - savings, a_min=1e-8)

        num_goods = prices.shape[0]
        negated_profit = prices.T @ (supplies - jnp.sum(allocations, axis=0))
        nsw = spending.T @ (jnp.log(u / spending))
        return negated_profit + nsw + jnp.sum(budgets - savings)

    @jax.jit
    def reward_and_step(self, market_params, state, prices, allocations, savings):
        """
        Computes the reward and next state for a given state, prices, allocations, and savings.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        state : dict
            A dictionary containing the state of the market.
        prices : jnp.ndarray
            An array containing the prices of the goods.
        allocations : jnp.ndarray
            An array containing the allocations of the goods.
        savings : jnp.ndarray
            An array containing the savings of the buyers.

        Returns:
        --------
        The reward and next state for the given state, prices, allocations, and savings.
        """
        supplies = state["supplies"]
        types = state["types"]
        budgets = state["budgets"]

        key = jax.random.PRNGKey(self.seed)

        r = self.reward(market_params, state, prices, allocations, savings)
        next_state = self.step(market_params, state, prices, allocations, savings)

        return r, next_state

    @jax.jit
    def step(self, market_params, state, prices, allocations, savings):
        """
        Computes the next state for a given state, prices, allocations, and savings.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        state : dict
            A dictionary containing the state of the market.
        prices : jnp.ndarray
            An array containing the prices of the goods.
        allocations : jnp.ndarray
            An array containing the allocations of the goods.
        savings : jnp.ndarray
            An array containing the savings of the buyers.

        Returns:
        --------
        The next state for the given state, prices, allocations, and savings.
        """
        supplies = state["supplies"]
        types = state["types"]

        key = jax.random.PRNGKey(self.seed)

        next_budgets = savings * market_params["ir"] + market_params["replenishment"]
        next_state = {"supplies": supplies, "types": types, "budgets": next_budgets}

        return next_state

    @partial(jax.jit, static_argnames=["price_policy", "buyer_policy", "num_episodes"])
    def state_value(
        self, market_params, state, price_policy, buyer_policy, num_episodes
    ):
        """
        Computes the state value for a given state, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        state : dict
            A dictionary containing the state of the market.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The state value for the given state, price policy, buyer policy, and number of episodes.
        """
        discount = market_params["discount"]
        cumul_reward = jnp.array([0.0])

        def episode_step(episode_num, episode_state):
            cumul_reward, state = episode_state
            prices = price_policy(state)
            allocations, savings = buyer_policy(state)
            reward, new_state = self.reward_and_step(
                market_params, state, prices, allocations, savings
            )
            cumul_reward += (discount**episode_num) * reward

            return cumul_reward, new_state

        cumul_reward, state = jax.lax.fori_loop(
            0, num_episodes, episode_step, (cumul_reward, state)
        )

        return jnp.squeeze(cumul_reward)

    @partial(
        jax.jit,
        static_argnames=[
            "shock_policy",
            "price_policy",
            "buyer_policy",
            "num_episodes",
        ],
    )
    def world_state_value(
        self, world_state, shock_policy, price_policy, buyer_policy, num_episodes
    ):
        """
        Computes the state value for a given state, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        state : dict
            A dictionary containing the state of the market.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The state value for the given state, price policy, buyer policy, and number of episodes.
        """
        market_params = shock_policy(world_state)
        discount = market_params["discount"]
        cumul_reward = jnp.array([0.0])
        state = market_params["init_state"]

        def episode_step(episode_num, episode_state):
            cumul_reward, world_state, state = episode_state
            prices = price_policy(world_state, state)
            allocations, savings = buyer_policy(world_state, state)
            market_params = shock_policy(world_state)
            reward, new_state = self.reward_and_step(
                market_params, state, prices, allocations, savings
            )
            new_world_state = self.world_states[episode_num + 1, :]
            cumul_reward += (discount**episode_num) * reward

            return cumul_reward, new_world_state, new_state

        cumul_reward, _, _ = jax.lax.fori_loop(
            0, num_episodes, episode_step, (cumul_reward, world_state, state)
        )

        return jnp.squeeze(cumul_reward)

    @partial(jax.jit, static_argnames=["price_policy", "buyer_policy", "num_episodes"])
    def trajectory(self, market_params, price_policy, buyer_policy, num_episodes):
        """
        Computes the trajectory for a given market, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The trajectory for the given market, price policy, buyer policy, and number of episodes.
        """
        discount = market_params["discount"]
        s = market_params["init_state"]

        def episode_step(s, traj):
            prices = price_policy(s)
            allocations, savings = buyer_policy(s)
            next_s = self.step(market_params, s, prices, allocations, savings)

            # Return next state and transition
            transition = Trajectory(s, prices, (allocations, savings))
            return next_s, transition

        # Get trajectory
        _, trajectory = jax.lax.scan(episode_step, s, None, length=num_episodes)

        return trajectory

    @partial(
        jax.jit,
        static_argnames=[
            "shock_policy",
            "price_policy",
            "buyer_policy",
            "num_episodes",
        ],
    )
    def world_state_trajectory(
        self, shock_policy, price_policy, buyer_policy, num_episodes
    ):
        """
        Computes the trajectory for a given market, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The trajectory for the given market, price policy, buyer policy, and number of episodes.
        """
        world_state = self.world_states[0, :]
        market_params = shock_policy(world_state)
        discount = market_params["discount"]
        world_state = self.world_states[0, :]
        state = market_params["init_state"]

        def episode_step(global_state, traj):
            # jax.debug.print("global_state {global_state}", global_state=global_state)
            # print("global_state", global_state)
            episode_num, world_state, state = global_state

            market_params = shock_policy(world_state)

            prices = price_policy(world_state, state)
            allocations, savings = buyer_policy(world_state, state)

            new_state = self.step(market_params, state, prices, allocations, savings)
            new_world_state = self.world_states[episode_num + 1, :]

            episode_num += 1
            # Return next state and transition
            transition = Trajectory(state, prices, (allocations, savings))
            return (episode_num, new_world_state, new_state), transition

        # Get trajectory
        _, trajectory = jax.lax.scan(
            episode_step, (0, world_state, state), None, length=num_episodes
        )

        return trajectory

    @partial(jax.jit, static_argnames=["price_policy", "buyer_policy", "num_episodes"])
    def trajectory_and_value(
        self, market_params, price_policy, buyer_policy, num_episodes
    ):
        """
        Computes the trajectory and cumulative reward for a given market, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The trajectory and cumulative reward for the given market, price policy, buyer policy, and number of episodes.
        """
        discount = market_params["discount"]
        cumul_reward = 0.0
        trajectory = []
        next_s = market_params["init_state"]

        def episode_step(episode_num, episode_state):
            curr_traj, cumul_reward, s = episode_state
            prices = price_policy(s)
            allocations, savings = buyer_policy(s, jax.lax.stop_gradient(prices))

            # Get reward and next state
            r, next_s = self.reward_and_step(
                market_params, s, prices, allocations, savings
            )

            # Update cumulative reward
            cumul_reward += (discount**episode_num) * r

            # Return trajectory, cumulative reward and next state
            return (
                curr_traj.append((s, prices, (allocations, savings), r)),
                cumul_reward,
                next_s,
            )

        # Get trajectory and cumulative reward
        trajectory, cumul_reward, _ = jax.lax.fori_loop(
            0, num_episodes, episode_step, (trajectory, s)
        )

        return trajectory, cumul_reward

    # @partial(jax. jit, static_argnames=["price_policy", "buyer_policy", "num_episodes"])
    def cumul_util(self, market_params, price_policy, buyer_policy, num_episodes):
        """
        Computes the cumulative utility for a given market, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The cumulative utility for the given market, price policy, buyer policy, and number of episodes.
        """

        return self.state_value(
            market_params,
            market_params["init_state"],
            price_policy,
            buyer_policy,
            num_episodes,
        )

    def world_cumul_util(self, shock_policy, price_policy, buyer_policy, num_episodes):
        """
        Computes the cumulative utility for a given market, price policy, buyer policy, and number of episodes.

        Parameters:
        -----------
        market_params : dict
            A dictionary containing the market parameters.
        price_policy : Callable
            A function that takes in a state and returns prices.
        buyer_policy : Callable
            A function that takes in a state and returns allocations and savings.
        num_episodes : int
            The number of episodes to simulate.

        Returns:
        --------
        The cumulative utility for the given market, price policy, buyer policy, and number of episodes.
        """
        init_world_state = self.world_states[0, :]

        return self.world_state_value(
            init_world_state, shock_policy, price_policy, buyer_policy, num_episodes
        )

    @partial(jax.jit, static_argnames=["policy", "br_policy", "num_episodes"])
    def cumulative_regret(self, market_params, policy, br_policy, num_episodes):
        price_policy, buyer_policy = policy
        br_price_policy, br_buyer_policy = br_policy

        buyer_br_util = self.cumul_util(
            market_params, price_policy, br_buyer_policy, num_episodes
        )
        price_br_util = self.cumul_util(
            market_params, br_price_policy, buyer_policy, num_episodes
        )

        return buyer_br_util - price_br_util

    @partial(
        jax.jit, static_argnames=["shock_policy", "policy", "br_policy", "num_episodes"]
    )
    def world_cumulative_regret(self, shock_policy, policy, br_policy, num_episodes):
        price_policy, buyer_policy = policy
        br_price_policy, br_buyer_policy = br_policy

        buyer_br_util = self.world_cumul_util(
            shock_policy, price_policy, br_buyer_policy, num_episodes
        )
        price_br_util = self.world_cumul_util(
            shock_policy, br_price_policy, buyer_policy, num_episodes
        )

        return buyer_br_util - price_br_util

    @partial(jax.jit, static_argnames=["num_br_epochs", "num_episodes"])
    def exploit(
        self,
        market_params,
        policy_params,
        learn_rate,
        num_episodes,
        num_br_epochs,
        init_br_params,
    ):
        prng = hk.PRNGSequence(self.seed)

        # Initialize best-response network
        market_actor_network = self.market_actor_network
        br_network = self.br_network

        # Initialize Optimizer for buyers
        br_update, br_opt_state = myjax.init_optimiser(learn_rate, init_br_params)

        @jax.jit
        def neural_cumul_regret(market_params, policy_params, br_params):
            br_price_policy = lambda state: br_network.apply(br_params, state)[0]
            br_buyer_policy = lambda state: br_network.apply(br_params, state)[1]
            price_policy = lambda state: market_actor_network.apply(
                policy_params, state
            )[0]
            buyer_policy = lambda state: market_actor_network.apply(
                policy_params, state
            )[1]

            return self.cumulative_regret(
                market_params,
                (price_policy, buyer_policy),
                (br_price_policy, br_buyer_policy),
                num_episodes,
            )

        grad_neural_cumul_regret = jax.value_and_grad(neural_cumul_regret, argnums=2)

        @jax.jit
        def br_step(num_br_epoch, br_step_state):
            br_params, br_opt_state = br_step_state
            cumul_regret_val, grads = grad_neural_cumul_regret(
                jax.lax.stop_gradient(market_params),
                jax.lax.stop_gradient(policy_params),
                br_params,
            )

            negated_grad = jax.tree_util.tree_map(lambda x: -x, grads)
            updates, new_opt_state = br_update(negated_grad, br_opt_state, br_params)
            new_br_params = optax.apply_updates(br_params, updates)
            return new_br_params, new_opt_state

        new_br_params, br_opt_state = jax.lax.fori_loop(
            0,
            num_br_epochs,
            br_step,
            (init_br_params, br_opt_state),
        )

        return (
            neural_cumul_regret(
                market_params, policy_params, jax.lax.stop_gradient(new_br_params)
            ).clip(min=0.0),
            new_br_params,
        )

    @partial(jax.jit, static_argnames=["num_br_epochs", "num_episodes"])
    def world_exploit(
        self,
        policy_params,
        learn_rate,
        num_episodes,
        num_br_epochs,
        init_br_params,
    ):
        prng = hk.PRNGSequence(self.seed)

        # Initialize best-response network
        market_actor_network = self.market_actor_network
        br_network = self.br_network

        # Initialize Optimizer for buyers
        br_update, br_opt_state = myjax.init_optimiser(learn_rate, init_br_params)

        dummy_state = {
            "supplies": jnp.array([1.0, 1.0]),
            "types": jnp.array([[2.0, 2.0], [3.0, 3.0], [2.0, 3.0]]),
            "budgets": jnp.array([10.0, 10.0, 10.0]),
        }

        # dummy_state = {
        #     "init_state": {
        #         "supplies": jnp.array([1.0, 1.0]),
        #         "types": jnp.array([[2.0, 2.0], [3.0, 3.0], [2.0, 3.0]]),
        #         "budgets": jnp.array([10.0, 10.0, 10.0]),
        #     },
        #     "ir": 1.0,
        #     "replenishment": jnp.array([0.0, 0.0, 0.0]),
        #     "discount": 0.9,
        # }

        @jax.jit
        def neural_cumul_regret(policy_params, br_params):
            br_price_policy = lambda world_state, state: br_network.apply(
                br_params, world_state, state
            )[0]
            br_buyer_policy = lambda world_state, state: br_network.apply(
                br_params, world_state, state
            )[1]
            shock_policy = lambda world_state: market_actor_network.apply(
                policy_params, world_state, dummy_state
            )[0]
            price_policy = lambda world_state, state: market_actor_network.apply(
                policy_params, world_state, state
            )[1]
            buyer_policy = lambda world_state, state: market_actor_network.apply(
                policy_params, world_state, state
            )[2]

            return self.world_cumulative_regret(
                shock_policy,
                (price_policy, buyer_policy),
                (br_price_policy, br_buyer_policy),
                num_episodes,
            )

        grad_neural_cumul_regret = jax.value_and_grad(neural_cumul_regret, argnums=1)

        @jax.jit
        def br_step(num_br_epoch, br_step_state):
            br_params, br_opt_state = br_step_state
            cumul_regret_val, grads = grad_neural_cumul_regret(
                jax.lax.stop_gradient(policy_params),
                br_params,
            )

            negated_grad = jax.tree_util.tree_map(lambda x: -x, grads)
            updates, new_opt_state = br_update(negated_grad, br_opt_state, br_params)
            new_br_params = optax.apply_updates(br_params, updates)
            return new_br_params, new_opt_state

        new_br_params, br_opt_state = jax.lax.fori_loop(
            0,
            num_br_epochs,
            br_step,
            (init_br_params, br_opt_state),
        )

        return (
            neural_cumul_regret(
                policy_params, jax.lax.stop_gradient(new_br_params)
            ).clip(min=0.0),
            new_br_params,
        )
