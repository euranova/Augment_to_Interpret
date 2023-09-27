# Augment to Interpret

This repository contains the official implementation to reproduce the results of the paper
*Augment to Interpret: Unsupervised and Inherently Interpretable Graph Embeddings*,
by G. Scafarto, M. Ciortan, S. Tihon and Q. Ferré.


## How to install
This repository uses Docker to run. Install [Docker](https://www.docker.com/) if necessary.

If your computer has enough RAM to handle the Mutag dataset, change the `LOW_RESOURCES` constant
in [tests/utils/constants.py](tests/utils/constants.py) from `True` to `False`.

Then, in a terminal located at the root of the repository, enter the following commands:
```bash
make docker_build  # create the docker image
make docker_run  # for gpu, run the docker image
# make docker_run_cpu  # for cpu, run the docker image
make download_data  # download Mutag and MNIST datasets
pytest tests  # download and/or create all the datasets with the same seed as the authors
# It also tests the installation.
```

## How to run
To reproduce all our results at once, you can enter the command `make run_all`
in the running docker container. This may take weeks.

To run an individual experiment, you can use the following command:
```bash
python scripts/main.py --help  # display some help about arguments to the function
python scripts/main.py \
    --dataset ba_2motifs \
    --loss simclr_double_aug_info_negative \
    --seed 0 \
    --model_name GIN  \
    --epochs 150 \
    # etc.  # train and save one model

make run_post_hoc  # run the analysis of the results
```

## Structure of the repository
The repository is structured as follows:
- [`augment_to_interpret`](augment_to_interpret/__init__.py) contains most of the code, coded as a library.
It can be installed using `pip` thanks to the file [`setup.py`](setup.py).
This will however not install the dependencies.
We recommend using Docker to run our code, as described earlier.
- [`scripts`](scripts/README.md) contains all the entry-points to the code.
- [`tests`](tests/__init__.py) contains Pytest tests (and corresponding files) to ensure the validity of the code.
The tests do not cover all functions of the code, but should be enough to detect
most installation problems.
- [`external_src`](external_src/README.md) contains external repositories that have been integrated in our pipeline.
These repositories have been slightly modified to ease integration.
- [`files`](files/README.md) contains most non-code files, such as data files, config files, result files, and so on.
