import logging
from constants import FilePath

# Setup Model Service Logger
logging.basicConfig(
    filename=FilePath.LOG_PATH,
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger('model-service')