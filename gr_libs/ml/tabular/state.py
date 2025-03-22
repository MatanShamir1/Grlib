from abc import ABC


class TabularState(ABC):
    def __init__(self,
                 agent_x_position: int,
                 agent_y_position: int,
                 agent_direction: int
                 ):
        self._agent_x_position = agent_x_position
        self._agent_y_position = agent_y_position
        self._agent_direction = agent_direction

    @staticmethod
    def gen_tabular_state(environment, observation):
        x, y = environment.unwrapped.agent_pos
        direction = observation['direction']
        return TabularState(agent_x_position=x, agent_y_position=y, agent_direction=direction)

    def __str__(self):
        return f"({self._agent_x_position},{self._agent_y_position}):{self._agent_direction}"
