import logging, os, sys

def get_logger():
        
    logger = logging.getLogger('GraphicalModel')
    logger.propagate = False
    
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(module)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)

    return logger
