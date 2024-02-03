import logging
import warnings
import os 

LOG_LEVEL = int( os.environ.get("LOG_LEVEL", logging.INFO ) )
CSV_DATA_SOURCE = os.environ.get("CSV_DATA_SOURCE", "data/weatherAUS.csv")

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig( level=LOG_LEVEL)