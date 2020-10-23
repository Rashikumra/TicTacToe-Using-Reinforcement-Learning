from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
       # define possible index collections in each of (horizontal, vertical, diagonal) directions

        # 3 rows of a 3 X 3 playing board
        Row_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # 3 columns of a 3 X 3 playing board
        Column_index = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

        # 2 diagonals
        diagonal_index = [[0, 4, 8], [2, 4, 6]]

        #Calculating sum across indices  to check the sum equals 15 to win the game
        sum_horizontally = [np.sum(np.array(curr_state)[i]) for i in Row_index]
        sum_vertically= [np.sum(np.array(curr_state)[i]) for i in Column_index]
        sum_diagonally = [np.sum(np.array(curr_state)[i]) for i in diagonal_index]

        Row_winner = list(filter(lambda x: x == 15, sum_horizontally))
        Column_winner = list(filter(lambda x: x == 15, sum_vertically))
        diagonal_winner = list(filter(lambda x: x == 15, sum_diagonally))

        #  If sum across any direction is equal to 15 winner id declared
        if len(Row_winner) != 0 or len(Column_winner) != 0 or len(diagonal_winner) != 0:
            return True
        else:
            return False



    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""
        
        '''added'''
        allowed_positions = self.allowed_positions(curr_state)
        allowed_values = self.allowed_values(curr_state)

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        
        #  Takes current state and action and returns the board position just after agent's move.
        current_new_state = [i for i in curr_state]

        # update current action
        current_new_state[curr_action[0]] = curr_action[1]

        return current_new_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""

        current_new_state = self.state_transition(curr_state, curr_action)


        terminal_state_reached, message = self.is_terminal(current_new_state)

        if terminal_state_reached:
            # set reward after game reaches to a terminal state due to agent move
            if message == "Win":
                reward = 10
                game_msg = "Agent Win"
            else:
                reward = 0
                game_msg = "Tie"

            return (current_new_state, reward, terminal_state_reached, game_msg)
        else:
            # terminal state not reached

            # randomly generating  environment action
            _, env_actions = self.action_space(current_new_state)
            env_action = random.choice([ac for i, ac in enumerate(env_actions)])

           
            state_dueto_env_action = self.state_transition(current_new_state, env_action)

          
            terminal_state_reached, message = self.is_terminal(
                state_dueto_env_action
            )

            # deciding whether its a tie or game can continue further
            if terminal_state_reached:
                if message == "Win":
                    reward = -10
                    game_msg = "Environment Win"
                else:
                    reward = 0
                    game_msg = "Tie"
            else:
                reward = -1
                game_msg = "continue"

            return (
                state_dueto_env_action,
                reward,
                terminal_state_reached,
                game_msg,
            )


    def reset(self):
        return self.state
