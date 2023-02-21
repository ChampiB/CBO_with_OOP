# Causal Bayesian Optimisation
Code for the causal Bayesian optimization (CBO) algorithm (http://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf).
This repository implements CBO using object-oriented programming.

# Usage
The code to run the CBO algorthm is in `runCBO.py`. It is possible to change the causal graph that by changing the value
of the variables 'experiment'. The results are saved in the folder `data/` when running the experiments. The folder 
`data/` contains the observational and interventional data used to produce the results in the paper. 

# Contacts
Feel free to contact the first author of the paper ([Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/)) for questions
