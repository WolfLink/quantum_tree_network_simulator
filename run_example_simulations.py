from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os

from plot_sim_results import *
from tree_sim import *

# this file is essentially a series of scripts that generate the example outputs

def dict_with_seed(original_dict, seed):
    newdict = dict()
    newdict.update(original_dict)
    newdict["seed"] = seed
    return newdict

def run_dynamic_response_sim():
    # configure the sim params
    launchdict = {
            "p" : 1e-3,
            "k" : 4,
            "n" : 4,
            "t" : "3e4",
            "b" : 1,
            "pmod" : [(1e-2,1e4),(1e-3,2e4)],
            }

    # prepare the working directory
    dir_str = "example_sim_results/dynamic_response_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # run the sims and save the raw data
        pool = Pool()
        repeats = 512
        seeds = np.random.SeedSequence().spawn(repeats)
        for i, result in enumerate(tqdm(pool.imap_unordered(launch_sim_from_dict, [dict_with_seed(launchdict, seeds[i].generate_state(1)[0]) for i in range(repeats)]), total=repeats)):
            with open(os.path.join(dir_str, f"{result['init_data']}_run_{i}.data"), "wb") as f:
                pickle.dump(result, f)
    else:
        print(f"Skipping running dynamic_response_sim because path {dir_str} already exists.")

    # draw the graphs
    plotdict = {
            "x" : "time",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/dynamic_sim_rate.png",
            "binsize" : 256,
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/dynamic_sim_time.png"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

def run_multi_size_sim():
    # prepare the working directory
    dir_str = "example_sim_results/multi_size_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # run the sims and save the raw data
        pool = Pool()
        l = 128
        for num_layers in tqdm([2,3,4,5]):
            params = [{"p" : 0.1**(1+2*i/l), "k" : 4, "n" : num_layers, "t" : "1e4", "b" : 1} for i in range(l)]
            for result in tqdm(pool.imap_unordered(launch_sim_from_dict, params), total=l):
                os.makedirs(os.path.join(dir_str, f"layers_{result['init_data'][2]}"), exist_ok=True)
                with open(os.path.join(dir_str, f"layers_{result['init_data'][2]}/{result['init_data']}.data"), "wb") as f:
                    pickle.dump(result, f)
        
    else:
        print(f"Skipping running multi_size_sim because path {dir_str} already exists.")

    # draw the graphs
    plotdict = {
            "x" : "p",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/multi_sim_rate.png",
            "legend" : True,
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/multi_sim_time.png"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)


def run_word_size_sim():
    # prepare the working directory
    dir_str = "example_sim_results/word_size_sim"
    os.makedirs("example_graphs", exist_ok=True)

    if not os.path.exists(dir_str):
        os.makedirs(dir_str, exist_ok=True)

        # run the sims and save the raw data
        pool = Pool()
        l = 128
        for word_size in tqdm([1,2,4,8,16]):
            params = [{"p" : 0.1**(1+2*i/l), "k" : 4, "n" : 4, "t" : "1e4", "b" : word_size} for i in range(l)]
            for result in tqdm(pool.imap_unordered(launch_sim_from_dict, params), total=l):
                os.makedirs(os.path.join(dir_str, f"layers_{result['init_data'][2]}"), exist_ok=True)
                with open(os.path.join(dir_str, f"layers_{result['init_data'][2]}/{result['init_data']}.data"), "wb") as f:
                    pickle.dump(result, f)
        
    else:
        print(f"Skipping running multi_size_sim because path {dir_str} already exists.")

    # draw the graphs
    plotdict = {
            "x" : "p",
            "y" : "rate",
            "indir" : dir_str,
            "outdir" : "example_graphs/word_sim_rate.png",
            "legend" : True,
            "detail" : "b"
            }

    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

    plotdict["y"] = "time"
    plotdict["outdir"] = "example_graphs/word_sim_time.png"
    if os.path.exists(plotdict["outdir"]):
        print(f"Skipping overwriting existing graph {plotdict['outdir']}")
    else:
        make_plot_from_dict(plotdict)

if __name__ == "__main__":
    run_dynamic_response_sim()
    run_multi_size_sim()
