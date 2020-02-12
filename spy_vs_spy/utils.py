import errno
import os
import numpy as np


def create_dir(directory):
    try:
        os.makedirs(os.path.dirname(directory))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


def find_bin_equal_width(value, lower_bound, upper_bound, num_of_bins):
    """
    From here: https://stackoverflow.com/a/6163403/7308982

    :param value:
    :param lower_bound:
    :param upper_bound:
    :param num_of_bins:
    :return:
    """

    bins = np.linspace(lower_bound, upper_bound, num_of_bins)
    digitized = np.digitize(value, bins)

    return digitized
