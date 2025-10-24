import sys, os
sys.path.append(os.getcwd() + "/examples/eyeriss_like")
from hardware.accelerator import *


def size_in_bits(size_in_bytes, factor):
    """
    converts size from bytes to bits

    Parameters
    ----------
    size_in_bytes : int
      size in bytes
    factor : str
      unit of size in bytes. One of "byte, K, M or G"

    Returns
    -------
    equivalent value in bits
    """

    multiplier = 8
    if factor == 'k' or factor == 'K':
        multiplier *= 1024
    elif factor == 'm' or factor == 'M':
        multiplier *= 1024**2
    elif factor == 'g' or factor == 'G':
        multiplier *= 1024**3
    return size_in_bytes * multiplier


def create_accelerator_from_mem_sizes(sizes, spatial_X=0, spatial_Y=0, spatial_level=-1):
    """
    creates an accelerator from list of memory sizes expressed in bytes.
    Automatically decides type of accelerator. If size > 512 bytes then uses SRAM
    Otherwise uses RF

    Parameters
    ----------
    sizes : list
      list of size of memory levels in bytes
      if first element is tuple, assume tensor specific buffers at L1
      tuple values will be (size of w buffer, size of ofmap buffer, size of ifmap buffer)

    Returns
    -------
    An accelerator with memory levels with specified sizes
    """

    memories = []
    levels = len(sizes)

    if type(sizes[0]) is tuple or type(sizes[0]) is list:
        L1_level = []
        L1_level.append(RF(name='L1_w',  size=size_in_bits(
            size_in_bytes=sizes[0][0] * 2,  factor="bytes"), width=16))
        L1_level.append(RF(name='L1_ofmap',  size=size_in_bits(
            size_in_bytes=sizes[0][1] * 2,  factor="bytes"), width=16))
        L1_level.append(RF(name='L1_ifmap',  size=size_in_bits(
            size_in_bytes=sizes[0][2] * 2,  factor="bytes"), width=16))
        memories.append(L1_level)
    else:
        memories.append(RF(name='L1',  size=size_in_bits(
            size_in_bytes=sizes[0] * 2,  factor="bytes"), width=16))

    for level in range(1, levels):
        size = sizes[level] * 2
        if size <= 2*256:
            if size < 16:
                size = 16
            memories.append(
                RF(name='L'+str(level+1),  size=size_in_bits(size_in_bytes=size,  factor="bytes"), width=16))
        else:
            num_banks = int(
                ceil(size_in_bits(size_in_bytes=size, factor="bytes") / (64*512)))
            if num_banks < 1:
                num_banks = 1
            if size < 16:
                size = 16
            memories.append(SRAM(name='L'+str(level+1),  size=size_in_bits(
                size_in_bytes=size,  factor="bytes"), width=64, banks=num_banks))
    memories.append(DRAM(name='DRAM', width=64, word=16))
    accelerator = Accelerator(
        technology='45nm', name='accelerator', memory_hierarchy=memories,
        spatial_X=spatial_X, spatial_Y=spatial_Y, spatial_level=spatial_level)
    return accelerator


# ******************** Eyeriss Like ********************
def acc_eyeriss_like():
    # ****************** Configuring Eyeriss v1 Accelerator ******************
    # *********** 14 x 12 PEs ***********
    spatial_dim1 = 14
    spatial_dim2 = 12
    # *********** 108 KB L2, 448 B Weight L1, 48 B Ofmap L1, 24B ifmap L1  ***********
    L2_capacity = int(1024 * 108 / 2)
    
    L1_w_buffer = 192
    L1_ofmap_buffer = 16
    L1_ifmap_buffer = 12

    L1_sizes = [L1_w_buffer, L1_ofmap_buffer, L1_ifmap_buffer]
    accelerator = create_accelerator_from_mem_sizes([L1_sizes, L2_capacity], spatial_dim1, spatial_dim2, spatial_level=1)
    #*********** 16b ifmap BW, 64b Weight & Ofmap BW ***********
    accelerator.bw.ifmap_bw = 1
    accelerator.bw.w_bw = 4
    accelerator.bw.ofmap_bw = 4
    accelerator.bw.refresh()
    accelerator.memory_hierarchy[1].banks = 32
    accelerator.memory_hierarchy[1].read_bandwidth =  accelerator.bw.read_bw + accelerator.bw.write_bw
    accelerator.memory_hierarchy[1].write_bandwidth = accelerator.bw.read_bw + accelerator.bw.write_bw
    accelerator.NoC_type = NoC_type="eyerissv1"
    accelerator.access_energies = {'L1': [1.59, 0.25, 0.24], 
                                    'L2': 17.565, 
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

