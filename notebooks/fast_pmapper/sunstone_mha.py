from src.generic_optimizers import *
import src.counter
from src.tiling import GenericTile, GenericProblem
import time

src.counter.init_counter()
print(src.counter.counter)

FREQUENCY = 1e9

# architecture parameters
# sizes are in terms of total data capacity 
L1_SIZE = 10000000
# energy (whatever unit as long as used consistently)
L1_E = 0.0e-12

# Read / Write BW
L1_BW = (1, 1)

# PE array configuration
L1_PEX = 128
L1_PEY = 128

# LLB
L2_SIZE = 131072  # LLB 128 kiB
L2_E = 2.08e-12
L2_BW = (2048, 1024)

L2_PEX = 4
L2_PEY = 1

# GLB
L3_SIZE = 1572864  # GLB 1.5 MiB
L3_ENERGY = 2.0e-12
L3_BW = (1024, 512)

# DRAM
DRAM_ENERGY = 56.24e-12
DRAM_BW = (256, 256)

# First, create the problem
bounds = {"B": 16, "H": 32, "M": 16*1024, "P": 16*1024, "E": 128}
tens_desc = [["B", "H", "M", "E"], ["B", "H", "P", "E"], ["B", "H", "M", "P"]]
prob = GenericProblem(bounds, tens_desc)


# We can print out the problem to see the analysis done
# (formatting is still WIP)
print("\n\nProblem analysis")
print(prob)


def run_sunstone(l2_bypass, l3_bypass):
    start = time.time()
    # Now initialize the Mapper
    # Since we are optimizing the lowest level first, we have no tiles or ordering
    # to work off of so we first create a tile wtith "0-levels", and set the
    # ordering to None

    # Now we call the actual enumerate+ranking function to get the L1 tiles
    # The first arg is the L1 size
    # The second arg is a list of access energies, for each level, for each tensor
    # In this case, we will have 2 levels, so we would want 2 entries in the
    # access energy list, where each entry is a 3-entry tuple (one for each tensor)
    # x_axis and y_axis are optional to include spatial unrolling after L1
    # static arg indicates to use static bounds (Sunstone principle for tiling 
    # that only grows in certain dimensions depending on the order)
    access_energies = [(L1_E, L1_E, L1_E),
                    (L2_E, L2_E, L2_E)]
    bw = [L1_BW, L2_BW]
    bypass = [[False, True, False]]
    L1_tiles, L1_costs = tls_sptl_tmprl(
                tiles=[(GenericTile(prob), None)],
                prob=prob,
                mem_size=L1_SIZE, 
                access_energies=access_energies,
                x_axis=L1_PEX,
                y_axis=L1_PEY,
                static=False,
                bw=bw,
                threads=8,
                edp=True,
                sptl_cnstrnts=[{"E": 128}, {"P": 128}],
                bypass=bypass  # TODO: only weights are not bypassed in the registers per MAC
            )

    # ret[0] now contains a list of (GenericTile, ordering) tuples, where
    # GenericTile now has L1+spatial tiling, and ordering is a list of strings to
    # describe the best ordering at each temporal level (in this case, there is one
    # ordering). ret[0] can now be passed to the next Mapping class, and each
    # promising tile+ordering at the L1 level can now be used as starting point to # find the L2 tiling
    # ret[1] contains the number of tiles evaluated (this is not mappings evaluated
    # since for each tile, we evaluate all promising orderings)

    # Now for the L2 mapper, we can pass ret[0] directly in, as well as the generic
    # problem

    # After initialization, we call a solve function again
    # The interface is the same as the one with "with_ord", with the first arg
    # being the L2 size, and second arg being a list of access energies. Since we
    # will have 3 levels, we need to pass the DRAM energy as well (again, one value
    # per tensor)
    # prior is a optional argmuent for upper factors
    access_energies.append((L3_ENERGY, L3_ENERGY, L3_ENERGY))
    bw.append(L3_BW)
    bypass.append(l2_bypass)

    # we also get an upper estimate on the cost for each candidate which will 
    #help in alpha-beta pruning
    L2_candidates, L2_costs = tls_sptl_tmprl_alpha_beta(
                tiles=L1_tiles,
                prob=prob,
                mem_size=L2_SIZE,
                access_energies=access_energies,
                costs=L1_costs,
                x_axis=L2_PEX,
                y_axis=L2_PEY,
                static=False,
                prior=False,
                bw=bw,
                threads=8,
                bypass=bypass,
                )

    access_energies.append((DRAM_ENERGY, DRAM_ENERGY, DRAM_ENERGY))
    bw.append(DRAM_BW)
    bypass.append(l3_bypass)
    L3_tiles, L3_orders, L3_costs = tls_tmprl_alpha_beta_mlt_thrd(
                tiles=L2_candidates,
                prob=prob,
                mem_size=L3_SIZE,
                access_energies=access_energies,
                costs=[cost[0] for cost in L2_costs],
                x_axis=None,
                y_axis=None,
                static=False,
                prior=False,
                bw=bw,
                threads=8,
                bypass=bypass,
                )
    dur = time.time() - start
    return L3_costs/FREQUENCY, dur

# like get_outin, solve_alpha_beta_parallel returns a list of (GenericTile, ordering) tuples
# but only return the best mapping. The ordering will now
# contain 2 strings, one for each memory interaction.

from itertools import product

def run_all_bypasses():
    best_edp = float('inf')
    total_duration = 0
    for l2_bypass in product([True, False], repeat=3):
        for l3_bypass in product([True, False], repeat=3):
            try:
                edp, duration = run_sunstone(l2_bypass, l3_bypass)
                total_duration += duration
                best_edp = min(edp, best_edp)
            except:
                pass
    return best_edp, total_duration


edp, duration = run_all_bypasses()


# If we define a list with every memory level, indicating whether each one has
# split buffers or not, as well as a list with the names of the tensor
# (according to the definition order in the tensor description), we can
# pass those as well as the ordering to yaml function of the tile to get a
# dictionary ready to be dumped onto a yaml file that will be compatible with
# Timeloop
# Every memory level is unified in this example
# arch = [('L1', False), ('L2', False), ('DRAM', False)]
# tens = ["Inputs", "Weights", "Outputs"]
# yaml_dict = tile.yaml(order, arch, tens)

# print("\n\nPrinting the formatted result")
# print(yaml_dict)

print("\n Optimization time:")
print(f"Time elapsed: ", duration)
print("EDP (Js):", edp)
print("Evaluations:", src.counter.counter)

from paths import DATA_DIR
import csv

with open(DATA_DIR / "sunstone_mha.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['evaluations', 'mapper_time', 'edp'])
    writer.writerow([src.counter.counter, duration, edp])