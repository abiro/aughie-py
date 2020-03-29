def _setup_logger():
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"'')  # noqa 501
    ch.setFormatter(formatter)
    logger.addHandler(ch)


__version__ = '0.0.4'
_setup_logger()
