import sys
import os
import pickle
import inspect
import gymnasium
from PIL import Image
import numpy as np

from gymnasium.envs.registration import register
from minigrid.core.world_object import Wall
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)
from ml.utils.format import minigrid_str_to_goal

def get_plans_result_path(env_name):
	return os.path.join("dataset", (env_name), "plans")

def get_policy_sequences_result_path(env_name):
	return os.path.join("dataset", (env_name), "policy_sequences")

def create_sequence_image(sequence, img_path, problem_name):
	if not os.path.exists(os.path.dirname(img_path)): os.makedirs(os.path.dirname(img_path))
	env_id = "MiniGrid-CustomColorS13N4-DynamicGoal-" + problem_name.split("-DynamicGoal-")[1]
	result = register(
		id=env_id,
		entry_point="gr_libs.minigrid_scripts.envs:CustomColorEnv",
		kwargs={"size": 13, "num_crossings": 4, "goal_pos": minigrid_str_to_goal(problem_name), "obstacle_type": Wall, "start_pos": (1, 1), "plan": sequence},
	)
	print(result)
	env = gymnasium.make(id=env_id)
	env = RGBImgPartialObsWrapper(env) # Get pixel observations
	env = ImgObsWrapper(env) # Get rid of the 'mission' field
	obs, _ = env.reset() # This now produces an RGB tensor only

	img = env.get_frame()

	####### save image to file
	image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
	image_pil.save(r"{}.png".format(img_path))


# TODO: instead of loading the model and having it produce the sequence again, just save the sequence from the framework run, and have this script accept the whole path (including is_fragmented etc.)
def analyze_and_produce_images(env_name):
	models_dir = get_models_dir(env_name=env_name)
	for dirname in os.listdir(models_dir):
		if dirname.startswith('MiniGrid'):
			model_dir = get_model_dir(env_name=env_name, model_name=dirname, class_name="MCTS")
			model_file_path = os.path.join(model_dir, "mcts_model.pth")
			try:
				with open(model_file_path, 'rb') as file:  # Load the pre-existing model
					monteCarloTreeSearch = pickle.load(file)
					full_plan = monteCarloTreeSearch.generate_full_policy_sequence()
					plan = [pos for ((state, pos), action) in full_plan]
					plans_result_path = get_plans_result_path(env_name)
					if not os.path.exists(plans_result_path): os.makedirs(plans_result_path)
					img_path = os.path.join(get_plans_result_path(env_name), dirname)
					print(f"plan to {dirname} is:\n\t{plan}\ngenerating image at {img_path}.")
					create_sequence_image(plan, img_path, dirname)
					
			except FileNotFoundError as e:
				print(f"Warning: {e.filename} doesn't exist. It's probably a base goal, not generating policy sequence for it.")

if __name__ == "__main__":
	# preventing circular imports. only needed for running this as main anyway.
	from ml.utils.storage import get_models_dir, get_model_dir
	# checks:
	assert len(sys.argv) == 2, f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 2.\n Example: \n\t /usr/bin/python scripts/get_plans_images.py MiniGrid-Walls-13x13-v0"
	assert os.path.exists(get_models_dir(sys.argv[1])), "plans weren't made for this environment, run graml_main.py with this environment first."
	analyze_and_produce_images(sys.argv[1])