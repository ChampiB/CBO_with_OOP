# Causal Bayesian Optimisation

Code for the causal Bayesian optimization (CBO) algorithm (http://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf).
This repository implements CBO using object-oriented programming.

# Installation, environment variables, and usage

It is recommended to use [Pycharm](https://www.jetbrains.com/pycharm/) and a [Virtual Environment](https://docs.python.org/3/library/venv.html).

To install the CBO package, you first need to create a virtual environment if it does not exist, and then activate this virtual environment.

Then, assuming you are in the `CBO_with_OOP` directory, the installation of the CBO package can be performed as follows:
```
pip install -e .
```

## Running the script using a terminal

After installing the CBO package, two environment variables need to be set.

The first is the `XP_PATH`, which corresponds to the experiment folder where the experiment logs will be stored.

Assuming that the repository was clone in the user's home, this variable can be set as follows:
```
export XP_PATH=/home/user_name/CBO_with_OOP/data/logs/
```

The second is the `DATA_PATH`, which corresponds to the directory where the graphs data is stored.

If the `CBO_with_OOP` repository was cloned in the user's home, this variable can be set as follows:
```
export DATA_PATH=/home/user_name/CBO_with_OOP/data/
```

Finally, the code running the CBO algorthm is in `./bin/run_cbo_algo.py`, and this script can be run as follows:
```
python ./bin/run_cbo_algo
```

## Running the script using Pycharm

# TODO fill this section

# Changing the script behaviour

It is possible to change the causal graph used by the CBO script just by changing the following configuration file:
```
./bin/config/run_cbo.yaml
```

The folder `./data/` contains the observational and interventional data used to produce the results in the paper.

# Contacts

Feel free to contact the first author of the paper ([Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/)) 
or one of the person maintaining this repository (e.g., [Theophile Champion](https://scholar.google.co.uk/citations?user=gjSgc9kAAAAJ&hl=en)) for questions.
