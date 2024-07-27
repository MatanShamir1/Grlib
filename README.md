### GRAML ###
## Execution:
# First, generate the results by running the framework with the experiments:
1. GRAML - python graml_main.py graml continuing_partial_obs inference_same_length learn_same_length collect_statistics
2. GRAQL - python graml_main.py graql fragmented_partial_obs collect_statistics
# Second, run the script that generates the results in a report plot:
1. GRAML - python scripts/generate_statistics_plots.py graml MiniGrid-Walls-13x13-v0/fragmented_partial_obs/inference_same_length/learn_same_length
2. GRAQL - python scripts/generate_statistics_plots.py graql MiniGrid-Walls-13x13-v0/fragmented_partial_obs

