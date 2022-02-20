from pathlib import Path

LOGGER_FOLDER = Path("../log/").resolve()
MODEL_FOLDER = Path("../model/").resolve()
LOGGER_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)