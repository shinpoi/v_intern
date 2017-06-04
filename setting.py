# encode: utf-8
# Python3.5
import logging
import time

DATA_PATH = './data/'

# set one of DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = logging.INFO
LOG_NAME = 'v_intern_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'

##################################################
# logging setting

logging.basicConfig(level=LOG_LEVEL,
                    format='[%(levelname)s]   \t %(asctime)s \t%(message)s\t',
                    datefmt='%Y/%m/%d (%A) - %H:%M:%S',
                    filename=LOG_NAME,
                    filemode='a'
                    )

console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)
formatter = logging.Formatter('[%(levelname)s]  \t%(message)s\t')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("saving log at: %s" % LOG_NAME)