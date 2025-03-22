import logging
import sys
from gr_libs.environment.environment import MINIGRID, PANDA, PARKING, POINT_MAZE, EnvProperty, MinigridProperty, PandaProperty, ParkingProperty, PointMazeProperty


def domain_to_env_property(domain_name: str):
	if domain_name == MINIGRID:
		return MinigridProperty
	elif domain_name == PARKING:
		return ParkingProperty
	elif domain_name == PANDA:
		return PandaProperty
	elif domain_name == POINT_MAZE:
		return PointMazeProperty
	else:
		logging.error(f"Domain {domain_name} is not supported.")
		sys.exit(1)