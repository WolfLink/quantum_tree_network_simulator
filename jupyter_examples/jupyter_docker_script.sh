# This script will launch a jupyter notebook in a docker container that will correctly run the demos
docker run --rm -it -v $(pwd)/..:/home/jovyan/work -p 8888:8888 jupyter/scipy-notebook
