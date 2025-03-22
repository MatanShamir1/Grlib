import sys
import os
import pickle
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)

def get_plans_result_path(env_name):
	return os.path.join("dataset", (env_name), "plans")

def get_policy_sequences_result_path(env_name):
	return os.path.join("dataset", (env_name), "policy_sequences")


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
	from gr_libs.ml.utils.storage import get_models_dir, get_model_dir
	# checks:
	assert len(sys.argv) == 2, f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 2.\n Example: \n\t /usr/bin/python scripts/get_plans_images.py MiniGrid-Walls-13x13-v0"
	assert os.path.exists(get_models_dir(sys.argv[1])), "plans weren't made for this environment, run graml_main.py with this environment first."
	analyze_and_produce_images(sys.argv[1])