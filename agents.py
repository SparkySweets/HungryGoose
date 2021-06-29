import random
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col


def epsilon_really_greedy_agent(observation, configuration, epsilon):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        player_index = observation.index
        player_goose = observation.geese[player_index]
        player_head = player_goose[0]
        player_row, player_column = row_col(player_head, configuration.columns)
        food = observation.food[0]
        food_row, food_column = row_col(food, configuration.columns)
        
        if food_row > player_row:
            return 2
        if food_row < player_row:
            return 0
        if food_column > player_column:
            return 1
        return 3