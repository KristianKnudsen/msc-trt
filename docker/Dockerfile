# Start from NVIDIA's official TRT-LLM release image
FROM nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5

# Install Python tools with exact versions
RUN pip install --no-cache-dir \
    transformers==4.55.0 \
    datasets==3.1.0 \
    accelerate==1.10.1 \
    lm-eval==0.4.9.1 \
    jupyterlab==4.4.3 \
    notebook==7.4.3 \
    ipywidgets==8.1.7 \
    matplotlib==3.10.3 \
    pandas==2.2.3 \
    psutil==7.0.0 \
    ipykernel \
    "nvidia-modelopt[hf]"

# Register a stable Jupyter kernel inside the container
RUN python -m ipykernel install --name trtllm --display-name "Python (trtllm)" --user

# Set default work directory
WORKDIR /workspace
