import os
import random
import logging
import torch
import numpy as np
import os.path as osp


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("Spatio-Temporal-Hypergraph-Model")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def set_logger(args):
    """
    Configure logging to write INFO+ to main log and DEBUG to a separate file.

    - Main log: train.log/test.log (INFO and above)
    - Debug log: train.debug.log/test.debug.log (DEBUG only)

    :param args: A argparse.Namespace object containing the command line arguments.
    """
    if args.do_train:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'train.log')
        debug_log_file = osp.join(args.log_path or args.init_checkpoint, 'train.debug.log')
    else:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'test.log')
        debug_log_file = osp.join(args.log_path or args.init_checkpoint, 'test.debug.log')

    # Remove all handlers associated with the root logger object
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)

    # Formatters
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', '%Y-%m-%d %H:%M:%S')

    # Root at DEBUG so DEBUG records are emitted; handlers decide what to persist
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # INFO+ file handler (main log)
    file_handler = logging.FileHandler(log_file, mode='w+', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # DEBUG-only file handler (separate debug log)
    class OnlyDebugFilter(logging.Filter):
        """
        Filter to only allow DEBUG records to pass through.
        """
        def filter(self, record: logging.LogRecord) -> int:
            """
            Filter a LogRecord to see if it should be emitted.

            :param record: The LogRecord to be filtered.
            :return: 1 if the record should be emitted, 0 otherwise.
            """
            return 1 if record.levelno == logging.DEBUG else 0

    debug_handler = logging.FileHandler(debug_log_file, mode='w+', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(OnlyDebugFilter())
    debug_handler.setFormatter(formatter)
    root.addHandler(debug_handler)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
