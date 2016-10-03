from .weather import *
from .utils import *
from .hadoop import *

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
