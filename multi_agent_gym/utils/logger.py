import logging


def create_standard_logger(python_file_name, stdout_level, log_file=None, log_file_level=logging.DEBUG):
    logger = logging.getLogger(python_file_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(stdout_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
