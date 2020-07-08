# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0



# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
