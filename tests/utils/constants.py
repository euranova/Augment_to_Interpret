""" Constants used in the tests """

from augment_to_interpret.basic_utils import C, get_device

LOW_RESOURCES = True  # set to True if e.g. your computer cannot load the mutag dataset

PATH_TEST_FILES = C.mkdir_and_return(C.PATH_ROOT, "tests", "test_files")
PATH_TMP_TEST_FILES = C.mkdir_and_return(PATH_TEST_FILES, "tmp")

DEVICE, C.CUDA_ID = get_device(C.CUDA_ID)
