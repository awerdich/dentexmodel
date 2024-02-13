<p float="left">
    <img style="vertical-align: top" src="./images/ccb_logo_text.jpeg" width="50%" />
    <img style="vertical-align: top" src="./images/train_195_boxes.png" width="40%" />
    
</p>

# The CCB Computer Vision Repository #

This repository contains code examples as a starting point for new computer vision projects. 
All frameworks, libraries and data sets are open source and publicly available.

## Docker container to create a reproducible environment ##

The included docker file can be used to create a reproducible environment in a docker 
container with all required dependencies installed.

```bash
# Create a docker image from the included docker file
docker compose build
# Create a container and run a jupyter lab server
docker compose -f docker-compose-cpu.yml up

# The default docker-compose.yml file is configured to use 
# the NVIDIA Container Toolkit runtime. 
# Create a container with the container toolkit installed
docker compose up 
```

## Docker with GPU support ##

The NVIDIA Container Toolkit is a set of tools designed to enable GPU-accelerated applications to run within Docker containers. 
This toolkit facilitates the integration of NVIDIA GPUs with container runtimes, 
allowing developers and data scientists to harness the power of GPU computing in containerized environments.
See the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) page for installation instructions.

### Dependencies ###

This package requires dependencies. 
If you are using the Docker environment, you should be good to go.
Rather than using a requirements.txt file, we
will use [pipenv](https://pipenv.readthedocs.io/en/latest/) to provide a pure, repeatable, application environment.
Mac/windows users should [install pipenv](https://pipenv.readthedocs.io/en/latest/#install-pipenv-today) into
their main python environment as instructed.  Unfortunately, using pipenv or
other virtual environments inside a conda environment is not recommended.

```bash
# Create a pipenv environment with all dependencies
pipenv install -e . --dev
# Run jupyter lab
pipenv run jupyter lab
```

### Download links

The data set: https://zenodo.org/records/7812323#.ZDQE1uxBwUG (11GB training, 150MB validation)

