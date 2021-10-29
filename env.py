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


class Agent(WindyGridworld):
    def __init__(self):
        super().__init__()
        self.actions_ = None
        self.v_ = None
        self.policy_ = None
        self.q_values = None
        self.q_optimized = None

    def set_full_actions(self):
        self.actions_ = ['N','NE','E','SE','S','SW','W','NW']


    def set_restricted_actions(self):
        self.actions_ = ['N','E','S','W']


    def init_v_and_policy(self):
        """
        Initialize the matrix of policy and value function from Gridworld dimensions
        """
        self.set_v(np.zeros(int(self.get_dim()[0] * self.get_dim()[1])))
        self.set_policy(np.array(['X' for _ in range(int(self.get_dim()[0] * self.get_dim()[1]))]))


    def init_q_values(self):
        self.q_values = {(s,a):0 for s in self.get_states() for a in self.get_actions()}


    def set_v(self, vector):
        """
        Modify Value for each States
        """
        self.v_ = vector


    def set_policy(self, vector):
        """
        Modify Policy for each States
        """
        self.policy_ = vector


    def get_q_values(self):
        return self.q_values


    def get_actions(self):
        """
        Returns list of Actions
        """
        return self.actions_


    def get_v(self):
        """
        Returns Value for each States
        """
        return self.v_


    def get_policy(self):
        """
        Returns Policy for each States
        """
        return self.policy_



    def wind_move(self, s):
        """
        This function corrects the movement of the agent based on the wind
        """
        temp_s = s
        if temp_s[1] in self.get_soft_windy_cols():
            temp_s = (temp_s[0]-1, temp_s[1])

        elif temp_s[1] in self.get_hard_windy_cols():
            temp_s = (temp_s[0]-2, temp_s[1])

        return temp_s


    def wall_move(self,s):
        """
        This function corrects the movement of the agent based on if the agent hits a wall
        """
        temp_s = s

        if temp_s[0] >= self.get_dim()[0]:
            temp_s = (self.get_dim()[0] -1, temp_s[1])

        if temp_s[0] < 0:
            temp_s = (0, temp_s[1])

        if temp_s[1] >= self.get_dim()[1]:
            temp_s = (temp_s[0],self.get_dim()[1] - 1)

        if temp_s[1] < 0:
            temp_s = (temp_s[0],0)

        return temp_s

    def move_conditions(self,s,a):
        """
        Returns the next state based on action and booleans if hit side wall or move
        """
        if a == 'N':
            temp_s = (s[0]-1,s[1])

        elif a == 'S':
            temp_s = (s[0]+1,s[1])

        elif a == 'W':
            temp_s = (s[0],s[1]-1)

        elif a == 'E':
            temp_s = (s[0],s[1]+1)

        elif a == 'NE':
            temp_s = (s[0]-1,s[1]+1)

        elif a == 'SE':
            temp_s = (s[0]+1,s[1]+1)

        elif a == 'SW':
            temp_s = (s[0]+1,s[1]-1)

        elif a == 'NW':
            temp_s = (s[0]-1,s[1]-1)

        else:
            raise Exception("Invalid Action")

        # Check whether or not the agent moved on the wall
        temp_s = self.wall_move(temp_s)
        # Moves the agent based on wind conditions
        temp_s = self.wind_move(temp_s)
        # Corrects for a wall in case the wind pushed to the wall
        temp_s = self.wall_move(temp_s)

        return temp_s


    def reward_policy(self, s):
        """
        Return the corresponsind reward based on the state agent is in
        """
        if s == self.get_terminal_grid():
            return self.get_rewards()['terminal']

        elif s == self.get_death_grid():
            return self.get_rewards()['death']

        elif s == None:
            return 0

        else:
            return self.get_rewards()['move']


    def sarsa(self, s,a):
        """
        SARSA Policy
        """
        if s == self.get_terminal_grid() and s == self.get_death_grid():
            return None

        else:
            s_prime = self.move_conditions(s,a)
            r = self.reward_policy(s_prime)
            a_prime = np.random.choice(self.get_actions())

            return(s,a,r,s_prime,a_prime)


    def random_state(self):
        """
        Function that starts the agent at a random state in the grid
        """
        length = len(self.get_states())
        rdm_idx = np.random.randint(length)
        # Makes sure we don't start in a non achievable state
        while start_state != self.get_terminal_grid() and start_state != self.get_death_grid():
            start_state = self.get_states()[rdm_idx]
        return self.get_states()[rdm_idx]


    def compute_sarsa_algo(self, lr = 0.01, iterations = 50000):
        """
        Computes the state-action pair for each grid according to the SARSA algorithm
        """
        t1 = time.time()
        j = 1

        for i in range(iterations):
            s = self.random_state()
            a = np.random.choice(self.get_actions())

            if (i % int(iterations/5) == 0 and i > 1) or (i == iterations - 1):
                print('{}% SARSA DONE in {:.2f}s'.format(20*j, t2 - t1))
                j += 1

            # Explore the grid until we finish in a terminal state
            while s != self.get_terminal_grid() and s != self.get_death_grid():
                (s, a, r, s_prime, a_prime) = self.sarsa(s,a)
                self.get_q_values()[s,a] = self.get_q_values()[s,a] + lr * (r + self.get_q_values()[s_prime, a_prime] - self.get_q_values()[s,a])
                (s,a) = (s_prime, a_prime)

                if s == self.get_terminal_grid() or s == self.get_death_grid():
                    t2 = time.time()

    def compute_optimal_actions(self):
        """
        Loops through each state and computes the optimal value of each state by the value of the state-action pair that is highest
        """
        self.q_optimized = [(s,self.get_actions()[np.argmax([self.get_q_values()[s,a] for a in self.get_actions()])]) for s in self.get_states()]


    def q_learning(self, epsilon=0.999, lr=0.01, iterations=50000):
        """
        Q-Learning algorithm with epsilon decay
        """
        t1 = time.time()
        j = 1
        rdm_count = 0
        greedy_count = 0

        for i in range(iterations):
            s = self.random_state()
            a = np.random.choice(self.get_actions())
            epsilon *= epsilon

            if (i % int(iterations/5) == 0 and i > 1) or (i == iterations - 1):
                print('{}% Q-LEARNING DONE in {:.2f}s'.format(20*j, t2 - t1))
                j += 1
                print("Greedy:{:.2f}% and Random:{:.2f}%".format(100*(greedy_count/(greedy_count+rdm_count)),100*(rdm_count/(greedy_count+rdm_count))))

            # Explore the grid until we finish in a terminal state
            while s != self.get_terminal_grid() and s != self.get_death_grid():

                # Pick a greedy action for the first 5% of iterations then selects the action according epsilon greedy policy
                # Forcing randomness at the beginning helps explore before explotation
                u = np.random.random()
                if u >= epsilon and i > int(0.05*iterations):
                    a = self.get_actions()[np.argmax([self.get_q_values()[s,a] for a in self.get_actions()])]
                    greedy_count +=1
                else:
                    a = np.random.choice(self.get_actions())
                    rdm_count +=1

                s_prime = self.move_conditions(s,a)

                self.get_q_values()[s,a] = self.get_q_values()[s,a] + lr * (self.reward_policy(s_prime) + np.max([self.get_q_values()[s_prime,a] for a in self.get_actions()]) - self.get_q_values()[s,a])

                s = s_prime

                if s == self.get_terminal_grid() or s == self.get_death_grid():
                    t2 = time.time()



    def find_action_from_state(self, s):
        """
        Return the optimal action for a given state
        """
        for state, action in self.q_optimized:
            if s == state:
                return action

    def plot_optimized_grid(self):
        """
        Plot Optimal Direction Matrix for all the states
        """
        fig = plt.figure(figsize=(14,14))
        ax = fig.add_subplot(111)
        data = np.empty((self.get_dim()[0]+2, self.get_dim()[1])) * np.nan
        cax = ax.matshow(data, cmap = 'binary')

        for state, action in self.q_optimized:
            if state != self.get_terminal_grid() and state != self.get_death_grid():
                ax.text(state[1]+0.5, state[0]+0.5, action, color='blue', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
            if state == self.get_starting_grid():
                    ax.text(self.get_starting_grid()[1]+0.5, self.get_starting_grid()[0]+0.75, action, color='green', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))

        ax.text(self.get_starting_grid()[1]+0.5, self.get_starting_grid()[0]+0.25, 'Start', color='green', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))

        ax.text(self.get_death_grid()[1]+0.5, self.get_death_grid()[0]+0.5, 'X', color='red', ha='center', va='center', fontsize = 40)

        ax.text(self.get_terminal_grid()[1]+0.5, self.get_terminal_grid()[0]+0.5, 'T', color = 'red', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

        for soft_wind in self.get_soft_windy_cols():
            ax.text(soft_wind+0.5, self.get_dim()[0]+0.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        for hard_wind in self.get_hard_windy_cols():
            ax.text(hard_wind+0.5, self.get_dim()[0]+0.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            ax.text(hard_wind+0.5, self.get_dim()[0]+1.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        extent = (0, data.shape[1], data.shape[0], 0)
        ax.set_xticks(np.arange(0,10,1))
        ax.grid(which = 'major', color='k',lw=2)
        ax.fill_between([9,10], 4, 5, facecolor='black')
        ax.fill_between([0,10], 7, 9, facecolor='black')
        ax.imshow(data, extent=extent)
        plt.show()


    def plot_optimal_set_directions(self):
        """
        Plot Optimal Direction Matrix from starting grid position
        """
        fig = plt.figure(figsize=(14,14))
        ax = fig.add_subplot(111)
        data = np.empty((self.get_dim()[0]+2, self.get_dim()[1])) * np.nan
        cax = ax.matshow(data, cmap = 'binary')
        curr_grid = self.get_starting_grid()

        count = 0
        while curr_grid != self.get_terminal_grid() and curr_grid != self.get_death_grid():
            count +=1
            action = self.find_action_from_state(curr_grid)
            if curr_grid == self.get_starting_grid():
                ax.text(curr_grid[1]+0.5, curr_grid[0]+0.75, action, color='blue', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
                curr_grid = self.move_conditions(curr_grid, action)
                continue

            ax.text(curr_grid[1]+0.5, curr_grid[0]+0.5, action, color='blue', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
            curr_grid = self.move_conditions(curr_grid, action)

            if count == 1e4:
                self.plot_optimized_grid()
                return None


        ax.text(self.get_starting_grid()[1]+0.5, self.get_starting_grid()[0]+0.25, 'Start', color='green', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))

        ax.text(self.get_death_grid()[1]+0.5, self.get_death_grid()[0]+0.5, 'X', color='red', ha='center', va='center', fontsize = 40)

        ax.text(self.get_terminal_grid()[1]+0.5, self.get_terminal_grid()[0]+0.5, 'T', color = 'red', ha='center', va='center', fontsize = 20,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

        for soft_wind in self.get_soft_windy_cols():
            ax.text(soft_wind+0.5, self.get_dim()[0]+0.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        for hard_wind in self.get_hard_windy_cols():
            ax.text(hard_wind+0.5, self.get_dim()[0]+0.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            ax.text(hard_wind+0.5, self.get_dim()[0]+1.5, u'\u2191', color='gray', ha='center', va='center', fontsize = 40,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        extent = (0, data.shape[1], data.shape[0], 0)
        ax.set_xticks(np.arange(0,10,1))
        ax.grid(which = 'major', color='k',lw=2)
        ax.fill_between([9,10], 4, 5, facecolor='black')
        ax.fill_between([0,10], 7, 9, facecolor='black')
        ax.imshow(data, extent=extent)
        plt.show()

    def complete_sarsa_episode(self):
        """
        Completes a whole episode of the SARSA algorithm and plot the optimal path based on the algorithm
        """
        self.init_q_values()
        self.compute_sarsa_algo()
        self.compute_optimal_actions()
        self.plot_optimal_set_directions()

    def complete_q_episode(self):
        """
        Completes a whole episode of the Q-Learning algorithm and plot the optimal path based on the algorithm
        """
        self.init_q_values()
        self.q_learning()
        self.compute_optimal_actions()
        self.plot_optimal_set_directions()
