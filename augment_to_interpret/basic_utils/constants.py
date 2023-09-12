"""
Contain constants of the library.
"""

from pathlib import Path

CUDA_ID = 0


def mkdir_and_return(*path, exist_ok=True):
    """Creates the directory at path if needed and returns it.
    :param path: Tuple[str or os.PathLike]; the parts of the path to the directory.
    :param exist_ok: bool; ignore FileExistsError on creation
    :return: Path; the path to the directory.
    """
    path = Path(*path).resolve()
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


#   /
PATH_ROOT = mkdir_and_return(__file__, "..", "..", "..")
#       files/
PATH_FILES = mkdir_and_return(PATH_ROOT, "files")
#           data/
PATH_DATA = mkdir_and_return(PATH_FILES, "data")
#           results/
PATH_RESULTS = mkdir_and_return(PATH_FILES, "results")
#           snakeconfig/
PATH_SNAKE_CONFIG = mkdir_and_return(PATH_FILES, "snakeconfig")
#           figures/
PATH_FIGURES = mkdir_and_return(PATH_FILES, "figures")
#       scripts/
PATH_SCRIPTS = mkdir_and_return(PATH_ROOT, "scripts")
#       external_src/
PATH_EXTERNAL_SRC = mkdir_and_return(PATH_ROOT, "external_src")
#           MEGA/
PATH_MEGA = mkdir_and_return(PATH_EXTERNAL_SRC, "MEGA")
