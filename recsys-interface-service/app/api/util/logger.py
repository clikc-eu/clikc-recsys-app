import logging
from ..constants import FilePath

# Setup Recsys Interface Service Logger
logging.basicConfig(
    filename=FilePath.LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('recsys-interface-service')