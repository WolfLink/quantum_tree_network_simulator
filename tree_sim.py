import numpy as np
from tqdm import tqdm
from time import perf_counter as timer
from multiprocessing import Pool, cpu_count, current_process
from matplotlib import pyplot as plt
from scipy.stats import t
import pickle
import os


# some constants
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
    def __init__(self, p, k, num_layers, num_cycles, multiplicity=1, include_expired=True):
        # store the sim properties
        self.p = p
        self.k = k
        self.num_layers = num_layers
        self.num_cycles = num_cycles
        self.multiplicity = multiplicity
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
        self.node_names = [(layer, index) for layer in range(num_layers) for index in range(k ** layer)]
        self.entages = dict()
        self.node_pairs = [(node, (node[0]-1, node[1] // k)) for node in self.node_names if node[0] > 0]
        for node_pair in self.node_pairs:
            self.entages[node_pair] = []

    def summary_dict(self):
        return {
                "init_data" : (self.p, self.k, self.num_layers, self.num_cycles, self.multiplicity),
                "stats" : (self.requests_enqueued, self.requests_satisfied, self.requests_expired, self.entanglements_expired, self.entanglements_used, self.entanglements_made),
                "request_cycles" : self.requests_cycles,
                "request_success" : self.requests_success,
                "timings" : (self.timer_memory_update.value, self.timer_request_make.value, self.timer_request_solve.value)
                }



    def run_sim(self):
        if current_process().name == "MainProcess":
            with tqdm(range(int(self.num_cycles)), 
                    total=self.num_cycles,
                    ) as pbar:
                for _ in pbar:
                    self.time_cycle()
                    pbar.set_description(
                            desc=f"E: {self.entanglements_used}/{self.entanglements_expired}/{self.entanglements_made} R: {self.requests_satisfied}/{self.requests_enqueued}",
                            #desc=f"E: {self.entanglements_used}/{self.entanglements_expired}/{self.entanglements_made} R: {self.requests_satisfied}/{self.requests_enqueued} T:{self.timer_memory_update.value}/{self.timer_request_make.value}/{self.timer_request_solve.value}",
                            refresh=False)
        else:
            for _ in range(int(self.num_cycles)):
                self.time_cycle()

    def time_cycle(self):
        tick = timer()
        # make new requests
        k = self.k
        N = k ** (self.num_layers - 1)
        n0 = N*(N-1)/2
        p0 = (N*self.p/(2*n0)) / self.multiplicity

        r_add_a, r_add_b = (np.random.rand(N,N) < p0).nonzero()
        for i in range(r_add_a.shape[0]):
            self.request_queue.extend([(r_add_a[i], r_add_b[i])] * self.multiplicity)
        self.request_ages.extend([0] * r_add_a.shape[0] * self.multiplicity)
        self.requests_enqueued += r_add_a.shape[0] * self.multiplicity

       # print(f"{len(self.request_queue)}\t{len(self.request_ages)}\t{self.requests_enqueued}")

        # on average there should be N*p/2 pairs per cycle

        tick = self.timer_request_make.tick(tick)

        # attempt to make new entanglements

        for pair in self.node_pairs:
            before_len = len(self.entages[pair])
            self.entages[pair] = [age + 1 for age in self.entages[pair] if age < EXPIRATION_TIME]
            self.entanglements_expired += before_len - len(self.entages[pair])

            memories_per_pair = MEMORIES_PER_END_NODE * (4 ** (self.num_layers - pair[0][0] - 1))

            if len(self.entages[pair]) < memories_per_pair:
                entanglements_made = int(np.sum(np.random.rand(memories_per_pair - len(self.entages[pair])) < 0.001))
                self.entages[pair].extend([0] * entanglements_made)
                self.entanglements_made += entanglements_made

        tick = self.timer_memory_update.tick(tick)

        # attempt to satisfy requests
        marked_for_deletion = []
        # potential speedup: only check requests for which an entanglement that may be useful has been made this cycle
        for i, request in enumerate(self.request_queue):
            # traverse the tree to check for completion
            curr_a = (self.num_layers - 1, request[0])
            curr_b = (self.num_layers - 1, request[1])
            while curr_a != curr_b:
                # identify the up-nodes
                up_a = (curr_a[0]-1, curr_a[1] // k)
                up_b = (curr_b[0]-1, curr_b[1] // k)
                
                # check for connections
                #print(self.entages.keys())
                if up_a[0] >= 0:
                    if len(self.entages[(curr_a, up_a)]) > 0:
                        # this passes the check
                        curr_a = up_a
                    else:
                        # this fails the check and qualifies for us to immediately stop
                        break

                # check for connections
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

            # check for success
            if curr_a != curr_b:
                # age up the request
                self.request_ages[i] += 1
                if self.request_ages[i] >= EXPIRATION_TIME:
                    marked_for_deletion.append(i)
                    self.requests_expired += 1
                    if self.include_expired:
                        self.requests_cycles.append(self.request_ages[i])
                        self.requests_success.append(0)
                continue

            # this request has passed! so we must go use the entanglements
            #i = self.request_queue.index(request)
            marked_for_deletion.append(i)
            self.requests_cycles.append(self.request_ages[i])
            self.requests_success.append(1)

            curr_a = (self.num_layers - 1, request[0])
            curr_b = (self.num_layers - 1, request[1])
            while curr_a != curr_b:
                # identify the up-nodes
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




    # at every clock cycle, there is a probability p0=N*p/(2*n0) that each of the n0 communication pairs is requested (all possible pairs in the whole thing)
    # sweep over p



class ChangingPTreeSim(TreeSim):
    def __init__(self, p, k, num_layers, num_cycles, multiplicity=1):
        super().__init__(p, k, num_layers, num_cycles, multiplicity)
        self.p_changes = dict()
        self.time = 0
        self.requests_finish_times = []


    def set_p_change(self, p, time):
        self.p_changes[time] = p

    def time_cycle(self):
        self.time += 1
        if self.time in self.p_changes:
            self.p = self.p_changes[self.time]
        super().time_cycle()
        self.requests_finish_times.extend([self.time] * (len(self.requests_cycles) - len(self.requests_finish_times)))
        
    def summary_dict(self):
        return {
                "init_data" : (self.p, self.k, self.num_layers, self.num_cycles),
                "stats" : (self.requests_enqueued, self.requests_satisfied, self.requests_expired, self.entanglements_expired, self.entanglements_used, self.entanglements_made),
                "request_cycles" : self.requests_cycles,
                "request_times" : self.requests_finish_times,
                "request_success" : self.requests_success,
                "timings" : (self.timer_memory_update.value, self.timer_request_make.value, self.timer_request_solve.value)
                }

def run_tree_sim_with_params(payload):
    desired_num_layers, p = payload
    multiplicity = 5
    sim = TreeSim(p=p, k=4, num_layers=desired_num_layers, num_cycles=1e4, multiplicity=multiplicity)
    sim.run_sim()

    summary_dict = sim.summary_dict()

    output_path = f"summary_dicts_multi{multiplicity}/layers_{desired_num_layers}/{summary_dict['init_data']}.data"
    os.makedirs(f"summary_dicts_multi{multiplicity}/layers_{desired_num_layers}", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(summary_dict, f)
    return summary_dict

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


def accept_convergence(samples, max_value):
    # takes in the number of samples and total, and a new sample
    # returns whether or not to accept convergence under these conditions
    n = len(samples)
    if n < 10:
        return False

    # could establish a maximum?
    if n >= 1000000:
        print("WARNING: sample maximum hit")
        return True

    S = np.var(samples, ddof=1)
    z = t.ppf(0.9, n-1)

    if z * S / np.sqrt(n) < 0.01 * max_value:
        return True
    else:
        return False

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
    if len(p_changes) > 0:
        if samples != 1:
            print("WARNING: ChangingPTreeSim currently does not support resampling as it is handled differently from other sims")
        sim = ChangingPTreeSim(p, k, n, t, b)
        for p_change in p_changes:
            sim.set_p_change(p_change[0], p_change[1])
        sim.run_sim()
        return sim.summary_dict()
    else:
        if samples == 1:
            sim = TreeSim(p, k, n, t, b)
            sim.run_sim()
            return sim.summary_dict()
        elif samples is None:
            samples = 0
            stats_total = [0,0,0,0,0,0]
            timings_total = [0,0,0]
            total_summary_dict = {
                    "init_data" : (p, k, n, t, b),
                    "request_cycles" : [],
                    "request_success" : [],
                    "samples" : samples,
                    "individual_samples" : []
                    }
            convergence_time = []
            convergence_success = []
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
            return total_summary_dict
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
