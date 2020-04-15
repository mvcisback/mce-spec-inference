# Start from a core stack version
FROM jupyter/datascience-notebook:9f9e5ca8fe5a
# Install from requirements.txt file
COPY requirements.txt /tmp/
COPY experiment.ipynb /home/jovyan/work
RUN pip install --requirement /tmp/requirements.txt
