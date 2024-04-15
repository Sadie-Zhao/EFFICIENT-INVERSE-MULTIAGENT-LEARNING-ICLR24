import haiku as hk


class LinearLayer(hk.Module):
    """A linear layer class.

    Attributes:
      num_nodes: number of nodes
      activation: activation function
    """

    def __init__(self, num_nodes, activation=lambda x: x):
        super().__init__()
        self.num_nodes = num_nodes
        self.activation = activation

    def __call__(self, x):
        """Applies the linear layer to the input.

        Args:
          x: input tensor

        Returns:
          The output tensor after applying the linear layer.
        """
        x = hk.Linear(self.num_nodes)(x)
        x = self.activation(x)
        return x


class ResidualBlock(hk.Module):
    """A residual block class.

    Attributes:
      num_nodes: number of nodes in the linear layers
      activation: activation function
    """

    def __init__(self, num_nodes, activation=lambda x: x):
        super().__init__()
        self.num_nodes = num_nodes
        self.activation = activation

    def __call__(self, x):
        """Applies the residual block to the input.

        Args:
          x: input tensor

        Returns:
          The output tensor after applying the residual block.
        """
        h = LinearLayer(self.num_nodes, self.activation)(x)
        h = LinearLayer(self.num_nodes)(h)
        return self.activation(h + x, axis=-1)


class BatchNormLinearLayer(hk.Module):
    """A batch norm linear layer class.

    Attributes:
      layer1: a type of layer, e.g., linear
      batch_norm: batch norm layer
      activation: activation function
    """

    def __init__(self, num_nodes, activation):
        super().__init__()
        self.layer1 = hk.Linear(num_nodes)
        self.batch_norm = hk.BatchNorm(False, False, 0.9)
        self.activation = activation

    def __call__(self, x, is_training):
        """Applies the batch norm linear layer to the input.

        Args:
          x: input tensor
          is_training: boolean indicating whether the model is training or not

        Returns:
          The output tensor after applying the batch norm linear layer.
        """
        x = self.layer1(x)
        x = self.batch_norm(x, is_training)
        x = self.activation(x)
        return x


class MyMLP(hk.Module):
    """A multi-layer perceptron class.

    Attributes:
      layers: list of layers
      are_batch_norm: list of booleans indicating whether each layer has batch norm or not
    """

    def __init__(self, num_nodes: list, activations: list, are_batch_norm: list):
        super().__init__()
        self.layers = []
        self.are_batch_norm = are_batch_norm

        for num_node, activation, is_batch_norm in zip(
            num_nodes, activations, are_batch_norm
        ):
            if is_batch_norm:
                self.layers.append(BatchNormLinearLayer(num_node, activation))
            else:
                self.layers.append(LinearLayer(num_node, activation))

    def __call__(self, x, is_training):
        """Applies the multi-layer perceptron to the input.

        Args:
          x: input tensor
          is_training: boolean indicating whether the model is training or not

        Returns:
          The output tensor after applying the multi-layer perceptron.
        """
        for layer, is_batch_norm in zip(self.layers, self.are_batch_norm):
            if is_batch_norm:
                x = layer(x, is_training)
            else:
                x = layer(x)
        return x
