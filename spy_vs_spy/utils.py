import errno
import os


def create_dir(directory):
    try:
        os.makedirs(os.path.dirname(directory))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
