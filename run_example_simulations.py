from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os

from plot_sim_results import *
from tree_sim import *

# This file generates the example outputs in example_graphs and example_sim_results

# This function duplicates a dictionary and inserts the specified seed.  It's use by run_dynamic_response_sim()
def dict_with_seed(original_dict, seed):
    newdict = dict()
    newdict.update(original_dict)
    newdict["seed"] = seed
    return newdict

# This function runs the Dynamic Response Simulation and plots the results.
# This is a simulation where we modify the request rate p during the simulation to observe the response
def run_dynamic_response_sim():
    # Configure the simulation parameters.
    launchdict = {
            "p" : 1e-3, # the initial value for request rate p
            "k" : 4, # the branching factor of the tree network
            "n" : 4, # the number of layers in the tree network
            "t" : "3e4", # the duration of the simulation
            "b" : 1, # the request batch size
            "pmod" : [(1e-2,1e4),(1e-3,2e4)], # add changes to the request rate p during the simulation.
            # pmod is specified as a list of changes in the format (time, new value for p)
            }

    # prepare the working directory
    dir_str = "example_sim_results/dynamic_response_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # Run the simulations and save the raw data.
        pool = Pool()
        repeats = 512 # This is the number of times to repeat the simulation with different random seeds for averaging.
        seeds = np.random.SeedSequence().spawn(repeats)
        for i, result in enumerate(tqdm(pool.imap_unordered(launch_sim_from_dict, [dict_with_seed(launchdict, seeds[i].generate_state(1)[0]) for i in range(repeats)]), total=repeats)):
            # The raw data from each simulation is stored in example_sim_results/dynamic_response_sim/(init_data)_run_i.data
            # where each run with a different seed has a unique value for i
            # and (init_data) is the tuple of (p, k, n, t, b), which are specified in the launchdict
            # The output is a pickled total_summary_dict dictionary.  See the description in the readme or in tree_sim.py for more details.
            with open(os.path.join(dir_str, f"{result['init_data']}_run_{i}.data"), "wb") as f:
                pickle.dump(result, f)
    else:
        print(f"Skipping running dynamic_response_sim because path {dir_str} already exists.")

    # Draw the graphs.
    # For the first plot, we will plot the current time in the simulation along the x axis, and the success rate on the y axis.
    plotdict = {
            "x" : "time",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/dynamic_sim_rate.pdf",
            "binsize" : 128, # This is the width of bins (in units of the x axis, so time cycles) for binned averaging.  See plot_sim_results.py for more information.
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    # For the second plot, we will plot the current time in the simulation along the x axis, and the request termination time on the y axis
    # The other parameters for the plot (indir and binsize) we will keep the same, so we will simply mutate the plotdict.
    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/dynamic_sim_time.pdf"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    plotdict["y"] = "buffer"
    plotdict["outdir"] = "example_graphs/dynamic_sim_buffer.pdf"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)
    

# This function runs the Multiple Size Simulation and plots the results.
# This involves running the simulation many times, sweeping over different values of p, and plotting the effects, and then repeating this process for different network sizes, controlled by using different values of n. 
def run_multi_size_sim():
    # Prepare the working directory.
    dir_str = "example_sim_results/multi_size_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # Run the simulations and save the raw data.
        pool = Pool()
        l = 128 # This is the number of different values of p to try.  The values of p that will be sampled are distributed logarithmically between 0.001 and 0.1.
        for num_layers in tqdm([2,3,4,5]): # This list contains the values for n, the number of layers in tree, that will be used.  This is how we control the size of the network.  We run the simulation for several values of n in order to explore how network size affects its properties.
            params = [{"p" : 0.1**(1+2*i/l), "k" : 4, "n" : num_layers, "t" : "1e4", "b" : 1, "resample" : True} for i in range(l)]
            sample_counts = []
            for result in tqdm(pool.imap_unordered(launch_sim_from_dict, params), total=l):
                sample_counts.append(result["samples"])
                os.makedirs(os.path.join(dir_str, f"n_{result['init_data'][2]}"), exist_ok=True) # The data is organized into subfolders based on the value of n.
                with open(os.path.join(dir_str, f"n_{result['init_data'][2]}/{result['init_data']}.data"), "wb") as f: # Each individual simulation has a data file, with a name based on its parameters.
                    pickle.dump(result, f)
        
    else:
        print(f"Skipping running multi_size_sim because path {dir_str} already exists.")

    # Draw the graphs.
    # For the first graph, we plot p as the x axis and the success rate as the y axis.
    plotdict = {
            "x" : "p",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/multi_sim_rate.pdf",
            "legend" : True,
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    # For the second graph, we plot request time as the y axis.  We mutate the plot dictionary because most of the parameters are the same as from the first graph.
    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/multi_sim_time.pdf"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)


# This function runs the Word Size Simulation and plots the results.
# This involves running the simulation many times, sweeping over different values of p, and plotting the effects, and then repeating this process for different word sizes, controlled by using different values of b.
# By "word size" we mean that when two clients make a request, they will instead make a batch of b requests.  This is similar to packet size in classical networking.
def run_word_size_sim():
    # Prepare the working directory.
    dir_str = "example_sim_results/word_size_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # Run the simulations and save the raw data.
        pool = Pool()
        l = 128 # This is the number of different values of p to try.  The values of p that will be sampled are distributed logarithmically between 0.001 and 0.1. 
        for word_size in tqdm([1,2,4,8,16]): # This list contains the values for b, the size of batches of requests that we will make.  This is how we simulate a network in which clients want larger packet or word sizes than just individual e-bits.
            params = [{"p" : 0.1**(1+2*i/l), "k" : 4, "n" : 4, "t" : "1e4", "b" : word_size, "resample": True} for i in range(l)]
            for result in tqdm(pool.imap_unordered(launch_sim_from_dict, params), total=l):
                os.makedirs(os.path.join(dir_str, f"b_{result['init_data'][4]}"), exist_ok=True) # The data is organized into subfolders based on the value of b.
                with open(os.path.join(dir_str, f"b_{result['init_data'][4]}/{result['init_data']}.data"), "wb") as f: # Each individual simulation has a data file, with a name based on its parameters.
                    pickle.dump(result, f)
        
    else:
        print(f"Skipping running word_size_sim because path {dir_str} already exists.")

    # Draw the graphs.
    # For the first graph, we plot p as the x axis and the success rate as the y axis.
    plotdict = {
            "x" : "p",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/word_sim_rate.pdf",
            "legend" : True,
            "detail" : "b"
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    # For the second graph, we plot request time as the y axis.  We mutate the plot dictionary because most of the parameters are the same as from the first graph.
    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/word_sim_time.pdf"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)


# This file can be imported and the individual functions can be called directly.
# Alternatively, this file can be run using "python3 run_example_simulations.py" and all three simulations will be run.  This is how the data in example_sim_results and graphs in example_graphs were made.
# Additionally, this file can be used as an example of how to use the other code in this repo to run other simulations.
if __name__ == "__main__":
    run_dynamic_response_sim()
    run_multi_size_sim()
    run_word_size_sim()
