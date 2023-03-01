import numpy as np
from collections import OrderedDict
import sys
sys.path.append("../..")


# Define a cost variable for each intervention
def cost_X_fix_equal(intervention_value, **kwargs):
    return 1.


def cost_Z_fix_equal(intervention_value, **kwargs):
    return 1.


# Define a cost variable for each intervention
def cost_X_fix_different(intervention_value, **kwargs):
    return 1.


def cost_Z_fix_different(intervention_value, **kwargs):
    return 10.


# Define a cost variable for each intervention
def cost_X_fix_different_variable(intervention_value, **kwargs):
    return np.sum(np.abs(intervention_value)) + 1.


def cost_Z_fix_different_variable(intervention_value, **kwargs):
    return np.sum(np.abs(intervention_value)) + 10.


# Define a cost variable for each intervention
def cost_X_fix_equal_variable(intervention_value, **kwargs):
    return np.sum(np.abs(intervention_value)) + 1.


def cost_Z_fix_equal_variable(intervention_value, **kwargs):
    return np.sum(np.abs(intervention_value)) + 1.


def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict([
            ('X', cost_X_fix_equal),
            ('Z', cost_Z_fix_equal),
        ])
    if type_cost == 2:
        costs = OrderedDict([
            ('X', cost_X_fix_different),
            ('Z', cost_Z_fix_different),
        ])

    if type_cost == 3:
        costs = OrderedDict([
            ('X', cost_X_fix_different_variable),
            ('Z', cost_Z_fix_different_variable),
        ])

    if type_cost == 4:
        costs = OrderedDict([
            ('X', cost_X_fix_equal_variable),
            ('Z', cost_Z_fix_equal_variable),
        ])

    return costs

