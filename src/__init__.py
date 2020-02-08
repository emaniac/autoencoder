import logging

FORMAT = '[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s'
DATEFORMAT = '%Y-%b-%d %H:%M:%S'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt=DATEFORMAT)