# Causal Bayesian Optimisation

Code for the causal Bayesian optimization (CBO) algorithm (http://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf).
This repository implements CBO using object-oriented programming.

# Installation, environment variables, and usage

It is recommended to a [Virtual Environment](https://docs.python.org/3/library/venv.html).

To install the CBO package, you first need to create a virtual environment if it does not exist, and then activate this virtual environment.

Then, assuming you are in the `CBO_with_OOP` directory, the installation of the CBO package can be performed as follows:
```
pip install -e .
```

## Running the script using a terminal

After installing the CBO package, two environment variables need to be set.

The first is the `XP_PATH`, which corresponds to the experiment folder where the experiment logs will be stored.

Assuming that the repository was cloned in the user's home, this variable can be set as follows:
```
export XP_PATH=/home/user_name/CBO_with_OOP/data/logs/
```

The second is the `DATA_PATH`, which corresponds to the directory where the graphs data is stored.
The folder `data/` of this repository contains the observational and interventional data used to produce the results in the paper.
Thus, if the `CBO_with_OOP` repository was cloned in the user's home, this variable can be set as follows:
```
export DATA_PATH=/home/user_name/CBO_with_OOP/data/
```

Finally, the code running the CBO algorthm is in `bin/run_cbo_algo.py`, and this script can be run as follows:
```
run_cbo_algo
```
If this does not work, make sure that the virtual environment created for this project is activated in your current 
shell.

## Changing the graph used

All the graphs configurations are stored in `bin/config/graph`.
One can pass the names of the graphs listed in this folder to the `run_cbo_algo` at runtime. 
The default graph is complete graph but one can change it, for example to coral graph, as so:
```
run_cbo_algo graph=coral
```
One can add custom graphs by creating new configuration files in the graph folder and use them in the
same way (see below for more on how to create custom graphs). For example, a new graph configuration created 
in `bin/config/graph/new_graph.yaml` can be used with
```
run_cbo_algo graph=new_graph
```

## Changing the cost type

#TODO[theo]: explain that better from the point of view of someone who have read the paper

Four cost types are used in CBO:
- 1: the cost of each node is 1 and there is no variable cost
- 2: the cost of some nodes is higher and there is no variable cost
- 3: the cost of each node is 1 and there is a variable cost
- 4: the cost of some nodes is higher and there is no variable cost

The cost type is 1 by default, but one can update the cost type used by doing:
```
run_cbo_algo cost_type=2
```

## Changing the number of initial samples

By default the number of initial samples is set to 100, but it can be updated as follows:
```
run_cbo_algo n_initial_samples=200
```

The different arguments can also be combined, for example:
```
run_cbo_algo n_initial_samples=200 cost_type=2 graph=coral
```

# Running CBO on custom graphs
#TODO[lisa] Do this part once the config structure is more definitive

## Creating a new graph configuration file

## Creating new nodes

## Using custom algorithms



# Contacts

Feel free to contact the first author of the paper ([Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/)) 
or one of the person maintaining this repository (e.g., [Theophile Champion](https://scholar.google.co.uk/citations?user=gjSgc9kAAAAJ&hl=en)) for questions.
