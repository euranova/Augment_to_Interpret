
.PHONY:
	clean_repo download_mutag download_mnist download_data
	docker_clean docker_build docker_run_cpu docker_run docker_terminal
	jupyter
	snakegraphs snakerun run_debug run_debug_fast run_graph_classif run_node_classif
	run_varying_r run_watchman run_adgcl run_mega run_sparsity_analysis run_post_hoc run_all

IDU=$(shell id -u)
IDG=$(shell id -g)

PARALLEL_SNAKERUNS=1


### -------------------------- Preparation --------------------------------- ###

clean_repo:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find files/* \( ! -path 'files/snakeconfig/*' -a ! -name snakeconfig \) -delete
	rm -r tests/test_files/tmp || true
	rm -r .snakemake augment_to_interpret.egg-info || true

download_mutag:
	mkdir -p files/data/mutag
	wget https://github.com/flyingdoog/PGExplainer/blob/master/dataset/Mutagenicity.pkl.zip?raw=true -O files/data/mutag/Mutagenicity.pkl.zip
	cd files/data/mutag; unzip Mutagenicity.pkl.zip && rm Mutagenicity.pkl.zip
	wget https://github.com/flyingdoog/PGExplainer/blob/master/dataset/Mutagenicity.zip?raw=true -O files/data/mutag/Mutagenicity.zip
	cd files/data/mutag; unzip Mutagenicity.zip && rm Mutagenicity.zip

download_mnist:
	mkdir -p files/data/mnist/raw
	python scripts/generate_superpixels.py -s train -t 4 -d ./files/data -o ./files/data/mnist/raw ;
	python scripts/generate_superpixels.py -s test -t 4 -d ./files/data -o ./files/data/mnist/raw ;

download_data: download_mutag download_mnist


### -------------------------- Docker commands ----------------------------- ###

docker_clean:
	docker kill augment_running || true
	jupyter notebook stop 8888 || true

docker_build:
	docker build -t augment --build-arg USER_ID=$(IDU) --build-arg GROUP_ID=$(IDG) .

docker_run_cpu:
	docker run --rm -it --shm-size 4G --cpu-shares 4 -v $(shell pwd):/workspace -p 8888:8888 --name augment_running augment

docker_run:
	docker run --rm -it --shm-size 4G --gpus all -v $(shell pwd):/workspace -p 8888:8888 --name augment_running augment #--runtime=nvidia

# Either docker_run or docker_run_gpu must have been run before (in a different terminal)
docker_terminal:
	docker exec -it augment_running bash


### ------------------------------- Helper --------------------------------- ###

jupyter:
	jupyter-notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''
# The notebook can now be accessed at 127.0.0.1:8888


### -------------------------- Running the workflow ------------------------ ###

snakegraphs:
	snakemake --forceall --dag | grep -v UserWarning | dot -Tpdf > files/dag.pdf
	snakemake --forceall --rulegraph | grep -v UserWarning | dot -Tpdf > files/rulegraph.pdf

snakerun:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0  # --configfile files/snakeconfig/default.yml

run_debug:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/debug.yml

run_debug_fast:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/debug_fast.yml

run_graph_classif:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/graph_classif_experiments.yml

run_node_classif:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/node_classif_experiments.yml

run_varying_r:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/varying_r_experiments.yml

run_watchman:
	snakemake --resources golden_ticket=$(PARALLEL_SNAKERUNS) --cores all --restart-times 0 --configfile files/snakeconfig/watchman_experiments.yml

run_adgcl:
	python scripts/baselines/adgcl.py

run_mega:
	python scripts/baselines/mega.py

run_sparsity_analysis:
	python scripts/sparsity_analysis.py

run_post_hoc:
	python scripts/posthoc_embedding_analysis.py
	python scripts/training_curve_analysis.py
	python scripts/result_dataframe_process.py

run_all: run_varying_r run_watchman run_node_classif snakerun
