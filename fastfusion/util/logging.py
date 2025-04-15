import logging
import logging.handlers
from multiprocessing import Manager


def log_worker(log_name):
    """
    Decorates a function that takes as an input an argument log_queue,
    which has type Queue, and converts that argument to a logger that
    posts to that queue.

    If log_queue is None, then the function is left unmodified.
    """
    def decorator(f: callable):
        def wrapper(*args, **kwargs):
            if "log_queue" not in kwargs:
                raise KeyError("Missing argument log_queue")
            log_queue = kwargs["log_queue"]
            if log_queue is None:
                return f(*args, **kwargs)

            worker_logger = logging.getLogger(log_name)
            queue_handler = logging.handlers.QueueHandler(log_queue)
            worker_logger.addHandler(queue_handler)
            worker_logger.setLevel("DEBUG")
            kwargs["log_queue"] = worker_logger

            return f(*args, **kwargs)
        return wrapper
    return decorator


def make_queue_and_listener():
    log = logging.getLogger()
    manager = Manager()
    log_queue = manager.Queue()
    log_queue_listener = logging.handlers.QueueListener(log_queue,
                                                        *log.handlers)
    return log_queue, log_queue_listener