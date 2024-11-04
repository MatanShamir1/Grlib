import numpy as np


def minigrid_str_to_goal(str):
	"""
	This function extracts the goal size (width and height) from a MiniGrid environment name.

	Args:
		env_name: The name of the MiniGrid environment (string).

	Returns:
		A tuple of integers representing the goal size (width, height).
	"""
	# Split the environment name by separators
	parts = str.split("-")
	# Find the part containing the goal size (usually after "DynamicGoal")
	goal_part = [part for part in parts if "x" in part]
	# Extract width and height from the goal part
	width, height = goal_part[0].split("x")
	return (int(width), int(height))

def maze_str_to_goal(str):
	"""
	This function extracts the goal size (width and height) from a MiniGrid environment name.

	Args:
		env_name: The name of the MiniGrid environment (string).

	Returns:
		A tuple of integers representing the goal size (width, height).
	"""
	# Split the environment name by separators
	parts = str.split("-")
	# Find the part containing the goal size (usually after "DynamicGoal")
	sizes_parts = [part for part in parts if "x" in part]
	goal_part = sizes_parts[1]
	# Extract width and height from the goal part
	width, height = goal_part.split("x")
	return (int(width), int(height))

def parking_str_to_goal(str):
	"""
	This function extracts the goal size (width and height) from a MiniGrid environment name.

	Args:
		env_name: The name of the MiniGrid environment (string).

	Returns:
		A tuple of integers representing the goal size (width, height).
	"""
	# Split the environment name by separators
	return str.split("-")[-2]

def panda_str_to_goal(str):
	"""Parses a string of the format 'PandaMyReachDenseXM0y3XM0y3X0y1-v3' into a list of 3 floats.
	Args:
	string: The input string to parse.
	Returns:
	A list of 3 floats representing the parsed values.
	"""
	try:
		numeric_part = str.split('PandaMyReachDenseX')[1]
		components = [component.replace('-v3', '').replace('y', '.').replace('M', '-') for component in numeric_part.split('X')]
		floats = []
		for component in components:
			floats.append(float(component))
		return np.array([floats], dtype=np.float32)
	except Exception as e:
		return "general"

def goal_str_to_tuple(str):
	assert str[0] == "(" and str[-1] == ")"
	str = str[1:-1]
	width, height = str.split(',')
	return (int(width), int(height))

def goal_to_task_str_pointmaze_dense_11(tuply, env_str):
	tuply = tuply[1:-1] # remove the braces
	#print(tuply)
	nums = tuply.split(',')
	#print(nums)
	return f'PointMaze-{env_str}EnvDense-11x11-Goal-{nums[0]}x{nums[1]}'

def goal_to_task_str_pointmaze_dense_11_obstacles(tuply):
	goal_to_task_str_pointmaze_dense_11(tuply, "Obstacles")
	
def goal_to_task_str_pointmaze_dense_11_four_rooms(tuply):
	goal_to_task_str_pointmaze_dense_11(tuply, "FourRooms")