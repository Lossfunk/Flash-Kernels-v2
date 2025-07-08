import logging
import sys
from dotenv import load_dotenv

from utils.config import config


def configure_logging():
    """Configure root logger only once. Level is controlled via LOG_LEVEL env var (default INFO).
    The format includes timestamp, level, logger name and message."""
    if logging.getLogger().handlers:
        # Already configured by another import. Do nothing.
        return

    # Load .env so secrets available
    load_dotenv()

    # Determine log level from YAML-configured value
    log_level_name = config.log_level
    # Fallback to INFO if value is invalid
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any previous basicConfig in Jupyter/REPL
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child logger after ensuring the root logger is configured."""
    configure_logging()
    return logging.getLogger(name)