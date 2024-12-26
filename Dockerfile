# ------------------------------------------------------------------------------
# 1) Consider switching to a runtime-based image if you do NOT need the full
#    CUDA compiler (nvcc) for building custom CUDA extensions. If you do,
#    you have to stay on `-devel`. But if not, try:
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#
# Otherwise, stick to devel:
    FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

    # Set timezone in one RUN statement, using noninteractive
    ENV DEBIAN_FRONTEND=noninteractive
    ENV TZ=Europe/Amsterdam
    RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
    
    # ------------------------------------------------------------------------------
    # 2) Combine apt-get install steps in a single RUN, remove apt list files
    #    afterwards to reduce layer size. Also use `--no-install-recommends` widely.
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            software-properties-common \
            git \
            git-lfs \
            curl \
            build-essential \
            libffi-dev \
            libxml2-dev \
            libjpeg-turbo8-dev \
            zlib1g-dev \
            libjpeg8 \
            libopenmpi-dev \
            libgl1 \
            openslide-tools \
            libopenslide0 \
            && add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.10-venv \
            libpython3.10-dev \
            && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # ------------------------------------------------------------------------------
    # 3) Create and activate python venv
    RUN python3.10 -m venv /venv
    ENV PATH=/venv/bin:$PATH
    
    # ------------------------------------------------------------------------------
    # 4) Install ASAP in one RUN layer, then remove .deb file afterward.
    # Install ASAP
    RUN : \
        && apt-get update \
        && apt-get -y install curl \
        && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2204.deb" \
        && dpkg --install ASAP-2.1-Ubuntu2204.deb || true \
        && apt-get -f install --fix-missing --fix-broken --assume-yes \
        && ldconfig -v \
        && apt-get clean \
        && echo "/opt/ASAP/bin" > /venv/lib/python3.10/site-packages/asap.pth \
        && rm ASAP-2.1-Ubuntu2204.deb \
        && apt-get -y install libgl1 \
        && :
    
    # ------------------------------------------------------------------------------
    # 5) Install Python dependencies. Combine as many pip installs as possible,
    #    always use --no-cache-dir to avoid leftover pip caches.
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir openslide-python
    
    # ------------------------------------------------------------------------------
    # 6) Set necessary NVIDIA environment variables
    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
    LABEL com.nvidia.volumes.needed="nvidia_driver"
    
    # Ensures that Python output to stdout/stderr is not buffered
    ENV PYTHONUNBUFFERED=1
    
    # ------------------------------------------------------------------------------
    # 7) Create a non-root user and set working directory
    RUN groupadd -r user && useradd -m --no-log-init -r -g user user
    RUN chown -R user:user /venv/
    USER user
    WORKDIR /opt/app
    
    # ------------------------------------------------------------------------------
    # 8) Copy requirements and resources first so Docker can cache these layers if
    #    they don’t change frequently.
    COPY --chown=user:user requirements.txt /opt/app/
    COPY --chown=user:user resources /opt/app/resources
    
    # ------------------------------------------------------------------------------
    # 9) Install python dependencies (requirements.txt).
    #    (You can combine some specialized pip install lines into this step.)
    RUN pip install --no-cache-dir -r /opt/app/requirements.txt
    
    # Example: Install nightly PyTorch
    RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    
    # ------------------------------------------------------------------------------
    # 10) Example of installing multiple packages in one RUN command
    RUN pip install --no-cache-dir \
        git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main \
        git+https://github.com/ultralytics/ultralytics.git@main \
        ensemble-boxes timm nltk wheel mpi4py
    
    # ------------------------------------------------------------------------------
    # 11) If you need detectron2 from a fork or custom build:
    RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/MaureenZOU/detectron2-xyz.git
    
    # ------------------------------------------------------------------------------
    # 12) Additional HPC or specialized repos. For each, consider removing the clone
    #     afterward if you do not need the repo locally. 
    RUN git clone --depth 1 https://github.com/microsoft/BiomedParse.git /opt/app/BiomedParse
    
    # Copy custom requirements for Biomp
    COPY --chown=user:user biomp/requirements.txt /tmp/requirements.txt
    RUN pip install --no-cache-dir -r /tmp/requirements.txt
    
    # Copy custom requirements
    COPY --chown=user:user biomp/requirements_custom.txt /tmp/requirements_custom.txt
    RUN pip install --no-cache-dir -r /tmp/requirements_custom.txt
    
    # ------------------------------------------------------------------------------
    # 13) Offline huggingface environment variables (if you are truly operating offline)
    ENV HUGGINGFACE_OFFLINE=1
    ENV TRANSFORMERS_OFFLINE=1
    ENV HF_HUB_OFFLINE=1
    ENV HF_DATASETS_OFFLINE=1
    ENV HF_HOME="/opt/ml/model"
    ENV MKL_THREADING_LAYER=GNU
    ENV NCCL_DEBUG=INFO
    ENV PYTHONPATH="/opt/app:$PYTHONPATH"
    
    # ------------------------------------------------------------------------------
    # 14) Clone huggingface models if needed locally
    #RUN git clone https://huggingface.co/openai/clip-vit-base-patch32 
    #RUN git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext 
    
    # ------------------------------------------------------------------------------
    # 15) Additional optional libraries
    RUN pip install --no-cache-dir pydicom nibabel SimpleITK
    
    # ------------------------------------------------------------------------------
    # 16) Download or copy model weights
    #WORKDIR /opt/ml/model
    #RUN curl -sSL -o model_state_dict.pt \
    #    https://www.doc.ic.ac.uk/~bkainz/models/model_state_dict.pt
    
    # ------------------------------------------------------------------------------
    # 17) Copy your application files last. This ensures that if they change,
    #     Docker only re-builds from this step forward rather than invalidating
    #     the entire build cache.
    WORKDIR /opt/app
    COPY --chown=user:user settings.json /home/user/.config/Ultralytics/
    COPY --chown=user:user inference.py inference_large.py init.py structures.py wsdetectron2.py /opt/app/
    COPY --chown=user:user biomp/ /opt/app/biomp/
    
    # ------------------------------------------------------------------------------
    # 18) Final entrypoint
    ENTRYPOINT ["/venv/bin/python3.10", "/opt/app/inference_large.py"]
    