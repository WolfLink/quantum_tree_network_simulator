This python library performs simulations of tree-shaped quantum networks under random requests for e-bits.

# Usage:

`python3 tree_sim.py help` will print a description of how to use tree\_sim.py as a command-line tool.

`python3 run_example_simulations.py` will perform the simulations that reproduce the data in example\_sim\_results and the graphs in example\_graphs.  It will skip re-doing simulations for which the output data or graphs are already present in the directories it expects.  If you wish to re-run the simulation, or run the simulation after tweaking run\_example\_simulations.py, you must delete the corresponding output files first.


run\_example\_simulations.py can also be used as an example of how to work with tree\_sim.py to run many simulations to compute statistics.



# Raw Data Specification:

The raw data in example\_sim\_results is organized like this:

- The folders dynamic\_response\_sim, multi\_size\_sim, and word\_size\_sim correspond to the similarly named functions in run\_example\_simulations.py.

- Within multi\_size\_sim and word\_size\_sim, results are further split by values of n and b respectively.

- Each individual data file corresponds to one call to tree\_sim.py, but may contain data from many simulations in the case where multi samples or dynamic sampling is requested.

## Individual data files are named (p, k, n, t, b).data where these are the input variables defining the parameters of the simulation:

- `p`: request rate such that $\frac{k^np}{b}$ pairs make $b$ requests every cycle
- `k`: branching factor of the tree
- `n`: number of layers in the tree
- `t`: duration of the simulation in cycles
- `b`: request batch size such that $\frac{k^np}{b}$ pairs make $b$ requests every cycle


The data files in dynamic\_response\_sim are labelled (p, k, n, t, b)_run_i.data where i is an index identifying which run of the simulation it is.


## These data files are pickled objects containing dictionaries, lists, and numpy objects.  The outermost object is a dictionary which always includes the following keys:

- `init_data` : A tuple of (p, k, n, t, b), the same simulation specification parameters used to generate the file name.
- `stats` : A tuple of (requests\_enqueued, requests\_satisfied, requests\_expired, entanglements\_expired, entanglements\_used, entanglements\_made), which are the total counts of these events over the course of the simulations captured in this file.
- `request_cycles` : A list containing the age at expiration or completion for every request that terminated during the simulation(s).
- `request_success` : A list containing a 1 for each request that completed and a 0 for each request that terminated.

### In the case where a data file represents multiple samples, the following keys will also be present:
- `samples` : The number of samples taken.
- `individual_samples` : Dictionaries following the above format that contain the data from each individual sample.

