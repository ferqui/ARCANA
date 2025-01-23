# FROM ghcr.io/nvidia/jax:jax-2024-09-10
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# FROM ghcr.io/nvidia/jax

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y git python3 python3-pip python-is-python3 python3-tk nvtop ffmpeg locales locales-all && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
RUN pip install -U pip

# Generic python libraries
RUN pip install matplotlib seaborn pandas tqdm scikit-learn ipykernel ipywidgets tensorboard

# PyTorch
RUN pip install torch torchvision torchaudio tonic

# Dev libraries
RUN pip install pre-commit black black[jupyter] isort flake8-black flake8-isort flake8-bugbear sphinx>=4.0 sphinx-book-theme myst_nb git+https://github.com/fehlings/boilerplot.git
# flake8-print flake8-bandit flake8-pylint flake8-builtins flake8-spellcheck flake8-pytest-style flake8-fastapi
