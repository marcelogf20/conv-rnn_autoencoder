""" File with the useful enums for the project """


from keras_custom import *

from enum import Enum, IntEnum
from functools import partial
import tensorflow as tf


def _get_element_from_string(string, enum_class, value=False):
    """ Function that receives a string and tries to get the corresponding
        enum in the enum class. It can return the value of the num, instead.
        It throws an exception if the string doesn't have a corresponding enum
        name.
    """
    elements = [e for e in enum_class]
    e_names = list(map(lambda x: x.name.lower(), elements))

    element = None
    try:
        if string is not None:
            pos = e_names.index(string.lower())
            element = elements[pos]

            if value:
                element = element.value

    except ValueError:
        error_str = string + ' is not a valid type of ' + enum_class
        ValueError(error_str)

    return element


class Folders(IntEnum):
    """ Enum representing the names of the folders for output """
    TEST = 0
    VALIDATION = 1
    TENSORBOARD = 2
    CHECKPOINTS = 3


class ExecMode(IntEnum):
    """ Enum representing the possible modes for running the model """
    TRAIN = 0
    VALID = 1
    TEST = 2


class Metrics(IntEnum):
    """ Enum representing the metrics used for comparison """
    PSNR = 0
    SSIM = 1
    MSSSIM = 2


class Codecs(IntEnum):
    """ Enum representing the codecs used in the code """
    NET = 0
    JPEG = 1
    JPEG2K = 2


class OutputType(IntEnum):
    """ Enum representing the output considered in the code """
    NONE = 0
    RESIDUES = 1
    RECONSTRUCTION = 2


class ImgData(Enum):
    """ Enum representing accepted ranges for pixels. """
    FLOAT = [0., 1.]
    UBYTE = [0, 255]

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum.
        """
        try:
            element = _get_element_from_string(string, ImgData, False)
        except ValueError:
            raise ValueError('Data format not known!')
        return element


class Optimizers(Enum):
    """ Enum represeting the acceptable optimizers """
    ADAM = tf.train.AdamOptimizer

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum.
        """
        try:
            element = _get_element_from_string(string, Optimizers, True)
        except ValueError:
            raise ValueError('Optimizer not known!')
        return element


class Activations(Enum):
    """ Enum with the available activations """
    # Enums cannot receive directly functions. They'll not become enum members
    RELU = partial(tf.keras.activations.relu)
    TANH = partial(tf.keras.activations.tanh)
    SIGMOID = partial(tf.keras.activations.sigmoid)
    ELU = partial(tf.keras.activations.elu)

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum.
        """
        try:
            element = _get_element_from_string(string, Activations, True)
        except ValueError:
            raise ValueError('Activation not known!')
        return element


class Losses(Enum):
    """ Enum with the available losses """
    MSE = partial(tf.losses.mean_squared_error)

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum. It can return the value of the enum, instead.
        """
        try:
            element = _get_element_from_string(string, Losses, True)
        except ValueError:
            raise ValueError('Loss not known!')
        return element


class Schedules(Enum):
    """ Enum with the available losses """
    EXP = partial(tf.train.exponential_decay)

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum. It can return the value of the enum, instead.
        """
        try:
            element = _get_element_from_string(string, Schedules, True)
        except ValueError:
            raise ValueError('Loss not known!')
        return element


class KLayers(Enum):
    """ Enum with keras layers (the strings used in the config file) """
    CONV2D = partial(tf.keras.layers.Conv2D)
    CONV2D_LSTM = partial(tf.keras.layers.ConvLSTM2D)
    CONV2D_TRANSPOSE = partial(tf.keras.layers.Conv2DTranspose)
    CONV3D = partial(tf.keras.layers.Conv3D)
    CONV3D_TRANSPOSE = partial(tf.keras.layers.Conv3DTranspose)
    DENSE = partial(tf.keras.layers.Dense)
    SUBTRACT = partial(tf.keras.layers.Subtract)
    RESHAPE = partial(tf.keras.layers.Reshape)

    # Custom layers
    EXPAND_DIMS = partial(ExpandDims)
    QUANTIZE = partial(Quantize)
    BINARIZE = partial(Binarize)
    GET_ONES = partial(GetOnes)
    DEPTH_TO_SPACE = partial(DepthToSpace)
    ADD_VALUE = partial(AddValue)

    @staticmethod
    def from_string(string):
        """ Function that receives a string and tries to get the corresponding
            enum.
        """
        try:
            element = _get_element_from_string(string, KLayers, True)
        except ValueError:
            raise ValueError('Layer not known!')
        return element
