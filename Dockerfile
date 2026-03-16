FROM condaforge/miniforge3:latest

WORKDIR /app

# Copie la recette conda et crée l'env
COPY environment.yml /tmp/environment.yml
RUN conda config --set channel_priority flexible \
 && conda config --set always_yes true \
 && conda config --set remote_connect_timeout_secs 60 \
 && conda config --set remote_read_timeout_secs 300 \
 && conda config --set remote_max_retries 10 \
 && conda install -n base -c conda-forge -y mamba \
 && mamba env create -y -f /tmp/environment.yml \
 && conda clean -afy

# Active l'env par défaut (sans "conda activate" interactif)
ENV PATH=/opt/conda/envs/netgpt/bin:$PATH \
    CONDA_DEFAULT_ENV=netgpt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copie le code (utile si tu veux exécuter sans volume),
# mais dans ton usage tu vas surtout monter le repo en volume.
COPY . /app

CMD ["python", "-c", "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"]
