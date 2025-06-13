import os
import numpy as np
import random
import tensorflow as tf


def disable_randomness(repeatability):
    os.environ['PYTHONHASHSEED'] = repeatability["PYTHONHASHSEED"]
    np.random.seed(repeatability["seed"])
    random.seed(repeatability["seed"])
    tf.random.set_seed(repeatability["seed"])
