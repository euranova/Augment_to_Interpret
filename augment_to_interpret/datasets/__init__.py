"""
Everything about datasets. Some code is from other repositories.
"""

from .ba_2motifs import SynGraphDataset
from .mnist import MNIST75sp
from .mutag import Mutag, MutagNoise
from .spmotif import SPMotif
from .utils import get_task
from .zinc import ZINC

from .get_data_loaders import load_node_dataset, get_data_loaders
