from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np
import gr_libs.ml as ml
from minigrid.core.world_object import Wall
#from q_table_plot import save_q_table_plot_image
from gymnasium.envs.registration import register

env_name = "MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0"
# create an agent and train it (if it is already trained, it will get q-table from cache)
agent = ml.TabularQLearner(env_name='MiniGrid-Walls-13x13-v0',problem_name = "MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0")
# agent.learn()

# save_q_table_plot_image(agent.q_table, 15, 15, (10,7))

# add to the steps list the step the trained agent would take on the env in every state according to the q_table
env = agent.env
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs, _ = env.reset() # This now produces an RGB tensor only

img = env.get_frame()

####### save image to file
from PIL import Image
import numpy as np

image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
image_pil.save(r"{}.png".format(env_name))

# ####### show image
# from gym_minigrid.window import Window
# window = Window(r"z")
# window.show_img(img=img)
# window.close()
