# DAT540-EvolutionaryANN
Project in DAT540 for exploring evolutionary ANNs in OpenAI's cartpole environment.

## Group Members
* Bjørn Christian Weinbach - Data Science
* Marius Sørensen - Data Science
* Ove Jørgensen - Data Science
* Håvard Godal - Data Science
* Johanna Kinstad - Data Science
* Vegard Rongve - Data Science

## Anaconda Environment
To organise the libraries utilised in our project. A anaconda environment is stored in the folder "Environment". 

For a deeper explanation of managing environments. see the documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Setting up environment from file

Open anaconda command line and type 

> conda env create -f DAT540EANN.yml

Anaconda will then set up a environment with the packages specifiedin the .yml file

### Updating environment

If you need to install another package, update the environment.yml file manually and use the command

> conda env update --file DAT540EANN.yml  --prune

The --prune option causes conda to remove any dependencies that are no longer required from the environment.
