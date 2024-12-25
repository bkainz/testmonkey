FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

#12.6.3-cudnn-devel-ubuntu22.04 12.4.1-devel-ubuntu22.04  

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.10
RUN : \
    && apt-get update \
    && apt-get install -y git git-lfs\
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.10-venv \
    && apt-get install libpython3.10-dev -y \
    && apt-get clean \
    && :


# Add env to PATH
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH


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

#COPY ASAP /opt/ASAP

# Install OpenSlide dependencies
RUN : \
    && apt-get update \
    && apt-get install -y openslide-tools libopenslide0 \
    && apt-get install -y build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev libjpeg8 libopenmpi-dev\
    && apt-get clean \
    && :

# Install OpenSlide Python bindings
RUN /venv/bin/python3.10 -m pip install --no-cache-dir openslide-python

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# update permissions
RUN chown -R user:user /venv/

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Update pip
RUN /venv/bin/python3.10 -m pip install pip --upgrade


# You can add any Python dependencies to requirements.txt
RUN /venv/bin/python3.10 -m pip install \
    --no-cache-dir \
    -r /opt/app/requirements.txt

#install pytorch
#RUN /venv/bin/python3.10 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#RUN /venv/bin/python3.10 -m pip install torch torchvision torchaudio 

RUN /venv/bin/python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify torch installation to ensure it's available
RUN /venv/bin/python3.10 -c "import torch; print(torch.__version__)"

## Install detectron2
#RUN /venv/bin/python3.8 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#RUN /venv/bin/python3.10 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#RUN git clone https://github.com/facebookresearch/detectron2.git
#RUN /venv/bin/python3.10 -m pip install -e detectron2
# Install Whole Slide Data
RUN /venv/bin/python3.10 -m pip install 'git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main'

RUN /venv/bin/python3.10 -m pip install git+https://github.com/ultralytics/ultralytics.git@main
#RUN /venv/bin/python3.10 -m pip install ultralytics

RUN /venv/bin/python3.10 -m pip install ensemble-boxes

RUN /venv/bin/python3.10 -m pip install timm
RUN /venv/bin/python3.10 -m pip install nltk

#RUN /venv/bin/python3.10 -m pip install torch==2.4.0+cu124 torchvision==0.15.0+cu124 torchaudio==2.4.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html

#RUN /venv/bin/python3.10 -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'

#RUN pip install git+https://github.com/cocodataset/panopticapi.git
#RUN pip install git+https://github.com/openai/CLIP.git

RUN /venv/bin/python3.10 -m pip install git+https://github.com/cocodataset/panopticapi.git
RUN /venv/bin/python3.10 -m pip install git+https://github.com/openai/CLIP.git
RUN /venv/bin/python3.10 -m pip install torch torchvision torchaudio 

RUN /venv/bin/python3.10 -m pip install wheel

RUN /venv/bin/python3.10 -m pip install --no-build-isolation 'git+https://github.com/MaureenZOU/detectron2-xyz.git'

RUN /venv/bin/python3.10 -m pip install mpi4py

#RUN /venv/bin/python3.10 -m pip install deepspeed


#RUN /venv/bin/python3.10 -m pip install 'git+https://github.com/microsoft/BiomedParse.git'
RUN git clone https://github.com/microsoft/BiomedParse.git
#RUN /venv/bin/python3.10 -m pip install -r BiomedParse/assets/requirements/requirements.txt

#RUN nvidia-smi

COPY biomp/requirements.txt /tmp/requirements.txt
RUN /venv/bin/python3.10 -m pip install -r /tmp/requirements.txt

COPY biomp/requirements_custom.txt /tmp/requirements_custom.txt
RUN pip install -r /tmp/requirements_custom.txt
#RUN git clone https://github.com/MaureenZOU/detectron2-xyz.git
#RUN /venv/bin/python3.10 -m pip install  -e detectron2-xyz --no-use-pep517
#RUN /venv/bin/python3.10 -c "import torch; print(torch.__version__)" > /dev/stdout

#RUN /venv/bin/python3.10 -m pip install -e detectron2

#RUN git lfs install

ENV PYTHONPATH="/opt/app:$PYTHONPATH"
ENV HUGGINGFACE_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HOME="/opt/app/cache/huggingface"
ENV MKL_THREADING_LAYER=GNU
ENV NCCL_DEBUG=INFO
#RUN cp /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi chmod ogu+x /usr/bin/nvidia-smi
#RUN nvidia-smi

RUN git clone https://huggingface.co/openai/clip-vit-base-patch32

RUN git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext


RUN /venv/bin/python3.10 -m pip install pydicom nibabel
RUN /venv/bin/python3.10 -m pip install SimpleITK

#RUN /venv/bin/python3.10 /opt/app/init.py
ENV HTTP_PROXY="http://proxy.rrze.uni-erlangen.de:80"
ENV https_proxy="http://proxy.rrze.uni-erlangen.de:80"

RUN mkdir -p /home/user/.config/Ultralytics/
COPY --chown=user:user settings.json /home/user/.config/Ultralytics/ 
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user init.py /opt/app/
COPY --chown=user:user structures.py /opt/app/
COPY --chown=user:user wsdetectron2.py /opt/app/
COPY --chown=user:user biomp/ /opt/app/biomp/
#COPY --chown=user:user cache /opt/app/

USER user
ENTRYPOINT ["/venv/bin/python3.10", "/opt/app/inference_large.py"]
