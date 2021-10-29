import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class WhileLoopError(Exception):
    pass

class WindyGridworld:
    def __init__(self):
        self.states_ = None
        self.rewards_ = {}
        self.gamma_ = 0
        self.starting_grid_ = None
        self.terminal_grid_ = None
        self.death_grid_ = None
        self.soft_wind_cols_ = []
        self.hard_wind_cols_ = []
        self.dim1_ = None
        self.dim2_ = None

    def set_states(self,dim1,dim2):
        """
        Creates the Gridworld states
        """
        if not isinstance(dim1,int) or not isinstance(dim1,int):
            raise Exception("Dimensions of the Grid must be an Integer")

        elif dim1 < 0 or dim2 < 0:
            raise Exception("Dimensions must be positive")

        else:
            self.states_ = [(a,b) for a in range(dim1) for b in range(dim2)]
            self.dim1_ = dim1
            self.dim2_ = dim2


    def set_rewards(self,val):
        """
        Creates the set of rewards and stores into a dictionary based on event
        """
        if type(val) is not tuple and len(val) != 4:
            raise Exception("Rewards must be a tuple of dimension 3")

        for v in val:
            if type(v) is not int and type(v) is not float:
                raise Exception("Rewards value must be Integer or Float")
        else:
            self.rewards_['move'] = val[0]
            self.rewards_['death'] = val[1]
            self.rewards_['terminal'] = val[2]


    def set_discount(self,val):
        """
        Sets the discount factor of each step in the environment
        """
        if type(val) is not int and type(val) is not float:
            raise Exception("Discount factor must be an Integer or Float")

        elif val < 0 or val > 1:
            raise Exception("Invalid Discounting factor must be between 0 and 1")

        else:
            self.gamma_ = val


    def set_starting_grid(self,state):
        """
        Sets the starting position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Starting State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Starting State must be within the Gridworld")

        elif state == self.get_death_grid():
            raise Exception("Starting State cannot be on death grid")

        elif state == self.get_terminal_grid():
            raise Exception("Starting State cannot be Terminal State")

        else:
            self.starting_grid_ = state


    def set_terminal_grid(self,state):
        """
        Sets the terminal position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Terminal State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Terminal State must be within the Gridworld")

        elif state == self.get_death_grid():
            raise Exception("Terminal State cannot be within death grid")

        elif state == self.get_starting_grid():
            raise Exception("Terminal State cannot be Starting State")

        else:
            self.terminal_grid_ = state


    def set_death_grid(self,state):
        """
        Sets the death grid position
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Death Grid must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Death Grid must be within the Gridworld")

        elif state in [self.get_starting_grid(), self.get_terminal_grid()]:
            raise Exception("Death Grid cannot be Starting or Terminal State")

        elif state == self.get_death_grid():
            raise Exception("State is already in Death Grid")

        else:
            self.death_grid_ = state


    def set_soft_windy_columns(self, val:int):
        self.soft_wind_cols_.append(val)


    def set_hard_windy_columns(self, val:int):
        self.hard_wind_cols_.append(val)


    def get_rewards(self):
        """
        Returns list of reward values
        """
        return self.rewards_


    def get_states(self):
        """
        Returns list of all the states
        """
        return self.states_


    def get_discount(self):
        """
        Return discount value
        """
        return self.gamma_


    def get_starting_grid(self):
        """
        Return Starting Point of Grid
        """
        return self.starting_grid_


    def get_terminal_grid(self):
        """
        Return Terminal Point of Grid
        """
        return self.terminal_grid_


    def get_death_grid(self):
        """
        Return Death Grid
        """
        return self.death_grid_


    def get_dim(self):
        """
        Return Dimensions of the Gridworld
        """
        return (self.dim1_, self.dim2_)

    def get_soft_windy_cols(self):
        return self.soft_wind_cols_

    def get_hard_windy_cols(self):
        return self.hard_wind_cols_


class WindyGridworld:
    def __init__(self):
        self.states_ = None
        self.rewards_ = {}
        self.gamma_ = 0
        self.starting_grid_ = None
        self.terminal_grid_ = None
        self.death_grid_ = None
        self.soft_wind_cols_ = []
        self.hard_wind_cols_ = []
        self.dim1_ = None
        self.dim2_ = None

    def set_states(self,dim1,dim2):
        """
        Creates the Gridworld states
        """
        if not isinstance(dim1,int) or not isinstance(dim1,int):
            raise Exception("Dimensions of the Grid must be an Integer")

        elif dim1 < 0 or dim2 < 0:
            raise Exception("Dimensions must be positive")

        else:
            self.states_ = [(a,b) for a in range(dim1) for b in range(dim2)]
            self.dim1_ = dim1
            self.dim2_ = dim2


    def set_rewards(self,val):
        """
        Creates the set of rewards and stores into a dictionary based on event
        """
        if type(val) is not tuple and len(val) != 4:
            raise Exception("Rewards must be a tuple of dimension 3")

        for v in val:
            if type(v) is not int and type(v) is not float:
                raise Exception("Rewards value must be Integer or Float")
        else:
            self.rewards_['move'] = val[0]
            self.rewards_['death'] = val[1]
            self.rewards_['terminal'] = val[2]


    def set_discount(self,val):
        """
        Sets the discount factor of each step in the environment
        """
        if type(val) is not int and type(val) is not float:
            raise Exception("Discount factor must be an Integer or Float")

        elif val < 0 or val > 1:
            raise Exception("Invalid Discounting factor must be between 0 and 1")

        else:
            self.gamma_ = val


    def set_starting_grid(self,state):
        """
        Sets the starting position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Starting State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Starting State must be within the Gridworld")

        elif state == self.get_death_grid():
            raise Exception("Starting State cannot be on death grid")

        elif state == self.get_terminal_grid():
            raise Exception("Starting State cannot be Terminal State")

        else:
            self.starting_grid_ = state


    def set_terminal_grid(self,state):
        """
        Sets the terminal position in the grid
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Terminal State must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Terminal State must be within the Gridworld")

        elif state == self.get_death_grid():
            raise Exception("Terminal State cannot be within death grid")

        elif state == self.get_starting_grid():
            raise Exception("Terminal State cannot be Starting State")

        else:
            self.terminal_grid_ = state


    def set_death_grid(self,state):
        """
        Sets the death grid position
        """
        if type(state) is not tuple and len(state) != 2:
            raise Exception("Death Grid must be a Tuple of dimension 2")

        elif state[0] >= self.get_dim()[0] or state[1] >= self.get_dim()[1]:
            raise Exception("Death Grid must be within the Gridworld")

        elif state in [self.get_starting_grid(), self.get_terminal_grid()]:
            raise Exception("Death Grid cannot be Starting or Terminal State")

        elif state == self.get_death_grid():
            raise Exception("State is already in Death Grid")

        else:
            self.death_grid_ = state


    def set_soft_windy_columns(self, val:int):
        self.soft_wind_cols_.append(val)


    def set_hard_windy_columns(self, val:int):
        self.hard_wind_cols_.append(val)


    def get_rewards(self):
        """
        Returns list of reward values
        """
        return self.rewards_


    def get_states(self):
        """
        Returns list of all the states
        """
        return self.states_


    def get_discount(self):
        """
        Return discount value
        """
        return self.gamma_


    def get_starting_grid(self):
        """
        Return Starting Point of Grid
        """
        return self.starting_grid_


    def get_terminal_grid(self):
        """
        Return Terminal Point of Grid
        """
        return self.terminal_grid_


    def get_death_grid(self):
        """
        Return Death Grid
        """
        return self.death_grid_


    def get_dim(self):
        """
        Return Dimensions of the Gridworld
        """
        return (self.dim1_, self.dim2_)

    def get_soft_windy_cols(self):
        return self.soft_wind_cols_

    def get_hard_windy_cols(self):
        return self.hard_wind_cols_
