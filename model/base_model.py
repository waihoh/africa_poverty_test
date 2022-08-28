class BaseModel:
    """
    The base class of models
    """

    def __init__(self, inputs, num_outputs, is_training, fc_reg, conv_reg):
        """
        :param inputs: tf.Tensor, shape [batch_size, H, W, C], type float32
        :param num_outputs: int, number of output classes. Set to None if we are only extracting features
        :param is_training: bool, or tf.placeholder of type tf.bool
        :param fc_reg: float, regularization for weights in the fully-connected layer
        :param conv_reg: float, regularization for weights in the conv layers
        """

        self.inputs = inputs
        self.num_outputs = num_outputs
        self.is_training = is_training
        self.fc_reg = fc_reg
        self. conv_reg = conv_reg

        # in subclasses, these should be initialized during __init__()
        self.outputs = None  # tf.Tensor, shape [batch_size, num_outputs]
        self.features_layer = None  # tf.Tensor, shape [batch_size, num_features]

    def init_from_numpy(self, path, sess, *args, **kwargs):
        """
        :param path: str, path to saved weights
        :param sess: tf.Session
        """
        raise NotImplementedError

    def get_first_layer_weights(self):
        """
        Gets the weights in the first layer of the CNN
        :return: tf.Tensor
        """
        raise NotImplementedError

    def get_final_layer_weights(self):
        """
        Get the weights in the final fully-connected layer after the conv layers
        :return: list of tf.Tensor
        """
        raise NotImplementedError

    def get_first_layer_summaries(self, ls_bands=None, nl_band=None):
        """
        Creates the following summaries:
            - histogram of weights in 1st conv layer
            - (if model includes batch-norm layer), histogram of 1st batch-norm layer's moving mean
        :param ls_bands: one of [None, 'rgb', 'ms'], if 'ms' then add separate histograms for RGB vs other channel weights the first layer of the CNN
        :param nl_band: one of [None, 'split', 'merge']
        :return: summaries: tf.summary, merged summaries
        """
        raise NotImplementedError
