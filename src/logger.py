import logging
import os
from datetime import datetime

# Create logs directory in project root
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create unique log file
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [Line:%(lineno)d] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create logger object
logger = logging.getLogger("InsuranceProjectLogger")
