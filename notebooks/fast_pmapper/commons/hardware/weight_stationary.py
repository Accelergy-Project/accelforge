import sys, os
sys.path.append(os.getcwd() + "/examples/eyeriss_like")
from hardware.accelerator import *
from hardware.eyeriss import size_in_bits, create_accelerator_from_mem_sizes

# ******************** Eyeriss Like ********************
def acc_weight_stationary():
     # ****************** Configuring Eyeriss v1 Accelerator ******************
    # *********** 16 x 16 PEs ***********
    spatial_dim1 = 16
    spatial_dim2 = 16
    # *********** 128 KB L2, 512KB L1  ***********
    L2_capacity = int(1024 * 128 / 2)
    L1_capacity = int(512 / 2) 
    accelerator = create_accelerator_from_mem_sizes([L1_capacity, L2_capacity], spatial_dim1, spatial_dim2, spatial_level=1)
    accelerator.memory_hierarchy[1].banks = 32

    accelerator.memory_hierarchy[0].read_bandwidth = 3
    accelerator.memory_hierarchy[0].write_bandwidth = 4
    #*********** 16b ifmap BW, 64b Weight & Ofmap BW ***********
    accelerator.bw.ifmap_bw = 1
    accelerator.bw.w_bw = 4
    accelerator.bw.ofmap_bw = 4
    accelerator.bw.refresh()
    accelerator.memory_hierarchy[1].read_bandwidth =  accelerator.bw.read_bw + accelerator.bw.write_bw  #9
    accelerator.memory_hierarchy[1].write_bandwidth = accelerator.bw.read_bw + accelerator.bw.write_bw  #9
    accelerator.NoC_type = NoC_type="eyerissv1"
    accelerator.access_energies = { 'L1': 2.05,
                                    'L2': 18.805, 
                                    'L3': 128.0, 
                                    'ifmap_NoC_check_tag': 0.131, 
                                    'ifmap_NoC_idle': 0.021, 
                                    'ofmap_NoC_check_tag': 0.238, 
                                    'ofmap_NoC_idle': 0.039, 
                                    'w_NoC_check_tag': 0.238, 
                                    'w_NoC_idle': 0.039, 
                                    'communication': 0.607
                                }
    return accelerator

