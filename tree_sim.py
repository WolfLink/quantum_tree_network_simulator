import numpy as np
from tqdm import tqdm
from time import perf_counter as timer
from multiprocessing import Pool, cpu_count, current_process
from matplotlib import pyplot as plt
from scipy.stats import t
import pickle
import os

# This file contains the core simulation code.

# It simulates the behavior of a quantum network organized in a tree structure.

# The shape of the network is defined when creating a new TreeSim, as having branching factor k and a number of layers n.

# The simulation will be run for t cycles.
# Every cycle, each pair of client nodes (the leaves of the tree) has a probability p/2b of making b requests.


# Core simulation constants are defined here.
EXPIRATION_TIME = 1e3
MEMORIES_PER_END_NODE = 10


class SmartTimer:
    def __init__(self):
        self.value = 0

    def tick(self, clock):
        tock = timer()
        self.value += tock - clock
        return tock

def index_for_current_process():
    name = current_process().name
    if name == "MainProcess":
        return 0
    else:
        return current_process()._identity[0]

class TreeSim:
    def __init__(self, p, k, n, t, b=1, include_expired=True):
        # store the sim properties
        self.p = p
        self.k = k
        self.n = n
        self.t = t
        self.b = b
        self.include_expired = include_expired

        # initialize the request queue
        self.request_queue = []
        self.request_ages = []

        # initialize the stats
        self.requests_enqueued = 0
        self.requests_satisfied = 0
        self.requests_expired = 0
        self.requests_cycles = []
        self.requests_success = []
        self.entanglements_expired = 0
        self.entanglements_used = 0
        self.entanglements_made = 0

        # initialize_timers
        self.timer_memory_update = SmartTimer()
        self.timer_request_make = SmartTimer()
        self.timer_request_solve = SmartTimer()

        # initialize the memories
        self.node_names = [(layer, index) for layer in range(n) for index in range(k ** layer)]
        self.entages = dict()
        self.node_pairs = [(node, (node[0]-1, node[1] // k)) for node in self.node_names if node[0] > 0]
        for node_pair in self.node_pairs:
            self.entages[node_pair] = []

    # This describes the output format.
    # init_data : The input values that describe the network size and properties of the simulation.
    # stats : Raw counts of various events regarding client requests.  Respectively: the total number of requests made over the course of the simulation, the number of requests that were completed successively, the number of requests that expired before they could be completed, the number of entanglements that expired before they could be used by requests, the number of entanglements that were consumed by requests, and the total number of entanglements made during the simulation.
    # request_cycles : The age of every request that was terminated (either completed or expired) at the time that it terminated.
    # request_success : Contains a 1 for every completed request and a 0 for every expired request.  Both this array and request_cycles are arranged in order of request termination.
    # timings : How much time was spent in certain sections of the code.  These are recorded to aid in optimizing the simulation.
    def summary_dict(self):
        return {
                "init_data" : (self.p, self.k, self.n, self.t, self.b),
                "stats" : (self.requests_enqueued, self.requests_satisfied, self.requests_expired, self.entanglements_expired, self.entanglements_used, self.entanglements_made),
                "request_cycles" : self.requests_cycles,
                "request_success" : self.requests_success,
                "timings" : (self.timer_memory_update.value, self.timer_request_make.value, self.timer_request_solve.value)
                }



    def run_sim(self):
        if current_process().name == "MainProcess":
            with tqdm(range(int(self.t)), 
                    total=self.t,
                    ) as pbar:
                for _ in pbar:
                    self.time_cycle()
                    pbar.set_description(
                            desc=f"E: {self.entanglements_used}/{self.entanglements_expired}/{self.entanglements_made} R: {self.requests_satisfied}/{self.requests_enqueued}",
                            refresh=False)
        else:
            for _ in range(int(self.t)):
                self.time_cycle()

    def time_cycle(self):
        tick = timer()

        # make new requests
        k = self.k
        N = k ** (self.n - 1)
        n0 = N*(N-1)/2
        p0 = (N*self.p/(2*n0)) / self.b

        r_add_a, r_add_b = (np.random.rand(N,N) < p0).nonzero()
        for i in range(r_add_a.shape[0]):
            self.request_queue.extend([(r_add_a[i], r_add_b[i])] * self.b)
        self.request_ages.extend([0] * r_add_a.shape[0] * self.b)
        self.requests_enqueued += r_add_a.shape[0] * self.b
        # on average there should be N*p/(2*b) pairs per cycle

        tick = self.timer_request_make.tick(tick)

        # attempt to make new entanglements

        for pair in self.node_pairs:
            before_len = len(self.entages[pair])
            self.entages[pair] = [age + 1 for age in self.entages[pair] if age < EXPIRATION_TIME]
            self.entanglements_expired += before_len - len(self.entages[pair])

            memories_per_pair = MEMORIES_PER_END_NODE * (4 ** (self.n - pair[0][0] - 1))

            if len(self.entages[pair]) < memories_per_pair:
                entanglements_made = int(np.sum(np.random.rand(memories_per_pair - len(self.entages[pair])) < 0.001))
                self.entages[pair].extend([0] * entanglements_made)
                self.entanglements_made += entanglements_made

        tick = self.timer_memory_update.tick(tick)

        # attempt to satisfy requests
        marked_for_deletion = []
        for i, request in enumerate(self.request_queue):
            # traverse the tree to check for completion
            curr_a = (self.n - 1, request[0])
            curr_b = (self.n - 1, request[1])
            while curr_a != curr_b:
                # identify the nodes that are one level up from the current node
                up_a = (curr_a[0]-1, curr_a[1] // k)
                up_b = (curr_b[0]-1, curr_b[1] // k)
                
                # check for entaglements between node curr_a and up_a
                if up_a[0] >= 0:
                    if len(self.entages[(curr_a, up_a)]) > 0:
                        # this passes the check
                        curr_a = up_a
                    else:
                        # this fails the check and qualifies for us to immediately stop
                        break

                # check for entaglements between node curr_b and up_b
                if up_b[0] >= 0:
                    if len(self.entages[(curr_b, up_b)]) > 0:
                        # this passes the check
                        curr_b = up_b
                    else:
                        # this fails the check and qualifies for us to immediately stop
                        break

                if curr_a != up_a and curr_b != up_b:
                    print(f"ERROR: infinite loop detected")
                    print(f"A: c: {curr_a} u: {up_a}\tB: c: {curr_b} u: {up_b}")
                    break

            # if needed entanglements are missing, we cannot complete these requests this cycle
            if curr_a != curr_b:
                # age up the request
                self.request_ages[i] += 1

                # if the request is now expired, mark it for deletion
                # we will clean up these marked requests after we are done checking all of the requests
                # also record that the request expired, as well as its age
                if self.request_ages[i] >= EXPIRATION_TIME:
                    marked_for_deletion.append(i)
                    self.requests_expired += 1
                    if self.include_expired:
                        self.requests_cycles.append(self.request_ages[i])
                        self.requests_success.append(0)
                continue

            # this request has passed! so we must go consume the entanglements

            # mark this request to be cleaned up later
            # also record that the request completed successfully, as well as its age
            marked_for_deletion.append(i)
            self.requests_cycles.append(self.request_ages[i])
            self.requests_success.append(1)

            # traverse the tree again, this time consuming the entaglements along the way
            curr_a = (self.n - 1, request[0])
            curr_b = (self.n - 1, request[1])
            while curr_a != curr_b:
                # identify the nodes that are one level up from the current node
                up_a = (curr_a[0]-1, curr_a[1] // k)
                up_b = (curr_b[0]-1, curr_b[1] // k)

                # free the memories
                if up_a[0] >= 0:
                    self.entages[(curr_a, up_a)].remove(max(self.entages[(curr_a, up_a)]))
                    self.entanglements_used += 1
                    curr_a = up_a

                if up_b[0] >= 0:
                    self.entages[(curr_b, up_b)].remove(max(self.entages[(curr_b, up_b)]))
                    self.entanglements_used += 1
                    curr_b = up_b

                if curr_a != up_a and curr_b != up_b:
                    print(f"ERROR: infinite loop detected")
                    print(f"A: c: {curr_a} u: {up_a}\tB: c: {curr_b} u: {up_b}")
                    break

            self.requests_satisfied += 1

        # clean up expired or completed requests
        for offset, index in enumerate(marked_for_deletion):
            try:
                del self.request_queue[index - offset]
                del self.request_ages[index - offset]
            except:
                print(f"DEBUG_INFO_1: {len(self.request_queue)}\t{len(self.request_ages)}")
                print(f"DEBUG_INFO_2: {index}-{offset}={index-offset}")
                print(f"DEBUG_INFO_3: {marked_for_deletion}")
                raise
        tick = self.timer_request_solve.tick(tick)


# This class is an extension to the main simulation program that enables the request rate p to be changed mid-simulation
# It also adds "request_times" to the summary_dict, which records the time when the request terminated, for each terminated request.  This is time simulation time at request termination, as opposed to request age at termination.
class ChangingPTreeSim(TreeSim):
    def __init__(self, p, k, n, t, b=1):
        super().__init__(p, k, n, t, b)
        self.p_changes = dict()
        self.time = 0
        self.requests_finish_times = []


    def set_p_change(self, p, time):
        self.p_changes[time] = p

    # This time_cycle modified p at the necessary times.
    # It calls into the time_cycle function from the superclass to handle the actual simulation.
    def time_cycle(self):
        self.time += 1
        if self.time in self.p_changes:
            self.p = self.p_changes[self.time]
        super().time_cycle()
        self.requests_finish_times.extend([self.time] * (len(self.requests_cycles) - len(self.requests_finish_times)))
        
    def summary_dict(self):
        return {
                "init_data" : (self.p, self.k, self.n, self.t),
                "stats" : (self.requests_enqueued, self.requests_satisfied, self.requests_expired, self.entanglements_expired, self.entanglements_used, self.entanglements_made),
                "request_cycles" : self.requests_cycles,
                "request_times" : self.requests_finish_times,
                "request_success" : self.requests_success,
                "timings" : (self.timer_memory_update.value, self.timer_request_make.value, self.timer_request_solve.value)
                }

# This simply returns a standard default launch dict with some reasonable example parameters.
def default_launch_dict():
    launchdict = {
            "p" : 1e-3,
            "k" : 4,
            "n" : 4,
            "t" : 1e4,
            "b" : 1,
            "o" : ".",
            "pmod" : [],
            }
    return launchdict


# This function decides convergence when trying to take a dynamic number of samples.
# The input "samples" is the list of the current data for the variable of interest.
# The input "max_value" is the maximum possible value for the variable of interest.
# This function is only ever used for "success rate" which ranges from 0 to 1, and "request age" which ranges from 0 to EXPIRATION_TIME.
def accept_convergence(samples, max_value):
    n = len(samples)

    # Establish a minimum so we don't return early from getting very lucky.
    if n < 10:
        return False

    # Establish a maximum to keep the simulation from spending too much time on the hard-to-simulate cases.
    if n >= 1000:
        return True

    # Accept convergence when the size of the 90% confidence interval is smaller than 1% of the max_value (which is the same as the range of the variable, because the minimum value for both variables of interest is 0).
    S = np.var(samples, ddof=1)
    z = t.ppf(0.9, n-1)
    if z * S / np.sqrt(n) < 0.01 * max_value:
        return True
    else:
        return False

# This is a helper function that runs the simulation based on a dictionary that specifies the input parameters.  This makes it easier to run the simulation many times with different parameters in run_example_simulations.py
# This function also manages taking multiple samples for one data point, including dynamic sampling.
def launch_sim_from_dict(launchdict):
    newlaunchdict = default_launch_dict()
    newlaunchdict.update(launchdict)
    launchdict = newlaunchdict
    p = float(launchdict["p"])
    k = int(launchdict["k"])
    n = int(launchdict["n"])
    t = int(float(launchdict["t"]))
    b = int(launchdict["b"])
    p_changes = launchdict["pmod"]

    # This is used to ensure simulations are run with different random seeds, in particular for the dynamic simulation, in which all simulations would end up exactly the same without different seeds.
    if "seed" in launchdict:
        np.random.seed(launchdict["seed"])

    samples = 1
    if "resample" in launchdict:
        if launchdict["resample"] is False or launchdict["resample"] is None:
            # just do one sample
            pass
        elif launchdict["resample"] is True:
            # do dynamic resampling
            samples = None
        else:
            # do the specified number of samples
            samples = launchdict["resample"]


    sim = None
    # If p_changes were specified, run the ChangingPTreeSim, which is handled a bit differently because we are interested in the behavior within a single sim instead of the statistics of sims with different parameters.
    if len(p_changes) > 0:
        if samples != 1:
            print("WARNING: ChangingPTreeSim currently does not support resampling as it is handled differently from other sims")
        sim = ChangingPTreeSim(p, k, n, t, b)
        for p_change in p_changes:
            sim.set_p_change(p_change[0], p_change[1])
        sim.run_sim()
        return sim.summary_dict()
    else:
        # If only one sample was requested, simply run it and return the result.
        if samples == 1:
            sim = TreeSim(p, k, n, t, b)
            sim.run_sim()
            return sim.summary_dict()

        # If dynamic sampling was requested, we will run samples until convergence is achieved.
        elif samples is None:
            samples = 0
            stats_total = [0,0,0,0,0,0]
            timings_total = [0,0,0]

            # Instead of returning the summary_dict from one simulation, we will return a total_summary_dict summarizing multiple simulations.
            # The data in request_cycles and request_success will include all of the data from all of the requests.
            # The number of samples taken is recorded in "samples".
            # The original raw data for each sample is preserved in "individual_samples".  This is a list of the summary_dict from each sample that was taken.
            total_summary_dict = {
                    "init_data" : (p, k, n, t, b),
                    "request_cycles" : [],
                    "request_success" : [],
                    "individual_samples" : []
                    }
            convergence_time = []
            convergence_success = []
            # We require the data for both request age and success rate to pass the convergence test before we accept convergence.
            while not (accept_convergence(convergence_time, EXPIRATION_TIME) and accept_convergence(convergence_success, 1)):
                # run a sim
                sim = TreeSim(p, k, n, t, b)
                sim.run_sim()
                result = sim.summary_dict()

                # record the result
                total_summary_dict["individual_samples"].append(result)
                total_summary_dict["request_cycles"] += result["request_cycles"]
                total_summary_dict["request_success"] += result["request_success"]

                stats = result["stats"]
                for i in range(len(stats)):
                    stats_total[i] += stats[i]

                timings = result["timings"]
                for i in range(len(timings)):
                    timings_total[i] += timings[i]

                convergence_time.append(np.mean(result["request_cycles"]))
                convergence_success.append(np.mean(result["request_success"]))
                samples += 1

            total_summary_dict["stats"] = tuple([stat / samples for stat in stats_total])
            total_summary_dict["timings"] = tuple([timing / samples for timing in timings_total])
            total_summary_dict["samples"] = samples
            return total_summary_dict
        
        # When a specific number of samples > 1, we run that many samples and record all of the data.
        # The number of samples taken is stored in "samples"
        # request_cycles and request_success will include data from all of the runs instead of any individual run
        # The original raw data, in the form of the original summary_dict from each simulation run, is preserved in a list in individual_samples
        else:
            stats_total = [0,0,0,0,0,0]
            timings_total = [0,0,0]
            total_summary_dict = {
                    "init_data" : (p, k, n, t, b),
                    "request_cycles" : [],
                    "request_success" : [],
                    "samples" : samples,
                    "individual_samples" : []
                    }
            for _ in range(samples):
                # run a sim
                sim = TreeSim(p, k, n, t, b)
                sim.run_sim()
                result = sim.summary_dict()

                # record the result
                total_summary_dict["individual_samples"].append(result)
                total_summary_dict["request_cycles"] += result["request_cycles"]
                total_summary_dict["request_success"] += result["request_success"]

                stats = result["stats"]
                for i in range(len(stats)):
                    stats_total[i] += stats[i]

                timings = result["timings"]
                for i in range(len(timings)):
                    timings_total[i] += timings[i]

            total_summary_dict["stats"] = tuple([stat / samples for stat in stats_total])
            total_summary_dict["timings"] = tuple([timing / samples for timing in timings_total])
            return total_summary_dict


# This file can be run as "python3 tree_sim.py" instead of being imported as a module.
# When used as a command line tool, running "python3 tree_sim.py help" will print a description of the available parameters.
if __name__ == "__main__":
    from parse_args import ParsedArgs
    args = ParsedArgs()

    # prepare the default args
    launchdict = default_launch_dict()
    if (len(args.by_index) > 0 and args.by_index[0] == "help") or "-help" in args.by_flag or "--help" in args.by_flag:
        print("Usage:\npython3 tree_sim.py <p> <k> <n> <t> <b>\np = base probability for request making\nk = branches per node in the tree structure\nn = number of layers in the tree structure\nt = number of cycles to run the simulation\nb = number of bits per request\nOmitted options will take on their default value. Options can also be specified by flag e.g. -t 30000\n\nOther flags:\n-o <output_dir>\n-pmod <new p>@<time> adds a marker to change the value of p to new p at the given time.")
        exit(0)
    else:
        # parse the by_index args
        by_index_keys = "pkntb"
        by_index_type = "fiiii"
        for i in range(len(args.by_index)):
            val = args.by_index[i]
            if by_index_type[i] == "f":
                val = float(val)
            elif by_index_type[i] == "i":
                val = int(val)
            launchdict[by_index_keys[i]] = val


        # parse the flags
        for flag in args.by_flag:
            if flag[1:] in "kntb":
                launchdict[flag[1:]] = int(args.by_flag[flag])
            elif flag == "-p":
                launchdict["p"] - float(args.by_flag["-p"])
            elif flag == "-o" or flag == "--output":
                launchdict["o"] = args.by_flag[flag]
            elif flag == "-pmod":
                if type(args.by_flag["-pmod"]) == str:
                    args.by_flag["-pmod"] = [args.by_flag["-pmod"]]
                for entry in args.by_flag["-pmod"]:
                    split = entry.split("@")
                    launchdict["pmod"].append((float(split[0]), int(split[1])))
            else:
                print(f"Unkown flag {flag} given value {args.by_flag[flag]}.")

    summary_dict = launch_sim_from_dict(launchdict)
    output_path = os.path.join(launchdict["o"], f"{summary_dict['init_data']}.data")
    with open(output_path, "wb") as f:
        pickle.dump(summary_dict, f)
