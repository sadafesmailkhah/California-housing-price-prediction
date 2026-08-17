import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pandas as pd
import requests
import tarfile



logger = logging.getLogger(__name__)

DATASETS_PATH = Path('datasets')
HOUSING_URL = 'https://github.com/ageron/data/raw/refs/heads/main/housing.tgz'


#Avoid side effects when this module is imported
def setup_logging(
        log_path: Path = Path('log.log')) -> None:

    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear() #Remove existing handlers

    handler = RotatingFileHandler(log_path,
                                  maxBytes=1_000_000,
                                  backupCount=3  ,
                                  encoding="utf-8")
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def dataset_downloader(
        url: str,
        tgz_path: Path,
        timeout: int = 30 #Maximum time to wait
) -> None:

    
    if tgz_path.is_file():
        return

    tgz_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = tgz_path.with_suffix(tgz_path.suffix + '.part') #Download to a temporary file first.
    
    try:
        logger.info('Trying to download dataset from %s', url)
        with requests.get(url, timeout=timeout, stream=True) as response: 
            response.raise_for_status() #Raise an error for bad status codes
            with temp_path.open('wb') as f:
                for chunk in response.iter_content(chunk_size=1 << 16): #Read 64KB at a time
                    f.write(chunk)
        temp_path.replace(tgz_path)
    except requests.exceptions.RequestException:
        logger.exception('Download failed')
        temp_path.unlink(missing_ok=True) #Clean up the temporary file
        raise #Re-raise the exception


def extract_archive(
        tgz_path: Path,
          datasets_path: Path) -> None:

    
    try:
        logger.info('Trying to extract file to %s', datasets_path)
        with tarfile.open(tgz_path) as f:
            f.extractall(path=datasets_path, filter='data')
    except (tarfile.TarError, EOFError, OSError):
        logger.exception('Extraction failed')
        raise


def dataset_loader(
    datasets_path: Path = DATASETS_PATH,
    url: str = HOUSING_URL,
) -> pd.DataFrame:

    
    csv_path = datasets_path / 'housing' / 'housing.csv'
    tgz_path = datasets_path / 'housing.tgz'

    if not csv_path.is_file():
        dataset_downloader(url, tgz_path)
        extract_archive(tgz_path, datasets_path)

    try:
        logger.info('Trying to read csv file from %s', csv_path)
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.exception('File not found after extraction')
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.exception('CSV file is corrupt or unreadable')
        raise

    if df.empty:
        logger.warning('The loaded dataset from %s is empty', csv_path)

    logger.info('Loaded %d rows dataset from %s', len(df), csv_path)
    return df


def main() -> pd.DataFrame:
    setup_logging()
    housing = dataset_loader()
    logger.info('Dataset ready with shape %s', housing.shape)
    return housing


if __name__ == '__main__':
    main()