IMAGE_NAME=my-trtllm
IMAGE_TAG=1.1.0rc5
CONTAINER_NAME=trtllm_dev
# TODO: Notebooks are outdated and unused.
NOTEBOOKS_DIR=~/Desktop/NTNU/prosjekt/notebooks

build:
        docker build --no-cache -t my-trtllm:1.1.0rc5 .
        docker image ls -a

clean:
        docker image prune -f
        docker image ls -a

run-shell:
        docker run --rm -it \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                --gpus=all \
                -p 25566:8888 \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v trtllm_data:/workspace \
                -v $(NOTEBOOKS_DIR):/notebooks \
                $(IMAGE_NAME):$(IMAGE_TAG) \
                bash



run-jupyter:
        docker run --rm -it \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                --gpus=all \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v trtllm_data:/workspace \
                -v $(NOTEBOOKS_DIR):/notebooks \
                -p 25566:8888 \
                $(IMAGE_NAME):$(IMAGE_TAG) \
                jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                        --NotebookApp.token='' --NotebookApp.password=''

run-shell-detached:
        docker run -dit \
                --name trt_shell \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                --gpus=all \
                -p 25566:8888 \
                -p 8000:8000 \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v trtllm_data:/workspace \
                -v $(NOTEBOOKS_DIR):/notebooks \
                $(IMAGE_NAME):$(IMAGE_TAG) \
                bash

run-shell-detached-hosted:
        docker run -dit \
                --name trt_shell \
                --net=host \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                --gpus=all \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v trtllm_data:/workspace \
                -v $(NOTEBOOKS_DIR):/notebooks \
                $(IMAGE_NAME):$(IMAGE_TAG) \
                bash
