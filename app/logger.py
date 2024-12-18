import logging
import sys
import os


def setup_logger(name, log_level=logging.INFO) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a single handler for stdout
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create a file handler for log file
    f_handler: logging.FileHandler = logging.FileHandler("app.log")
    f_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter: logging.Formatter = logging.Formatter(
        "%(levelname)s: [%(asctime)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(f_handler)

    # Optionally, prevent log duplication in FastAPI/Uvicorn
    logger.propagate = False

    return logger


# Instantiate logger
logger = setup_logger(os.path.basename(os.path.dirname(__file__)), logging.INFO)
