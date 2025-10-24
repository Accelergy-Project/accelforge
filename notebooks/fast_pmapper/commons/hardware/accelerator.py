import sys, os
sys.path.append(os.getcwd() + "/examples/eyeriss_like")
from math import ceil
from hardware.accelergy_components import *


class Component(ABC):
  """
  Abstract class for hardware component to be compatible with Timeloop

  Methods
  -------
  get_TL_style_dict()
    returns the dictionary in TL format that describes the component
  """

  @abstractmethod
  def get_TL_style_dict(self):
    pass



class Memory(Component):
  """
  A Memory component. Childeren of component.

  Attributes
  ----------
  type : str
    smartbuffer_SRAM or smartbuffer_RF
  subtype : str
    LPDDR4 for DRAM and "" otherwise
  name : str
  size : int
    total size in bits
  width : int
  depth : int
    size divided by width
  data_capacity : int
    total number of data with length words (e.g. total capacity for 16-bit data)
  read_bw : int
  write_bw : int
  word : int
    number of bits that data the component containts have (e.g. 16 or 32)
  """

  def __init__(self, type, subtype, name, size, width, read_bw, write_bw, word):
    """
    Parameters
    ----------
    type : str
      smartbuffer_SRAM or smartbuffer_RF
    subtype : str
      LPDDR4 for DRAM and "" otherwise
    name : str
    size : int
      total size in bits
    width : int
    read_bw : int
    write_bw : int
    word : int
    """
    self.type = type
    self.subtype = subtype
    self.name = name
    self.size = size
    self.width = width
    self.word = word
    self.block = int(self.width / self.word)
    self.depth = max(int(ceil(self.size / self.width)), 1)
    self.data_capacity = int(self.size / self.word)
    self.read_bandwidth = read_bw
    self.write_bandwidth = write_bw




class DRAM(Memory):
  """
  class for DRAM memory. Childeren of Memory.

  Methods
  -------
  get_TL_style_dict()
    returns the dictionary in TL format that describes the component
  """

  def __init__(self, name, width, word):
    super().__init__(type="DRAM", subtype="LPDDR4", name=name, size=0, width=width, read_bw=0, write_bw=0, word=word)


  def get_TL_style_dict(self):
    dict = {'name' : self.name,
            'class' : self.type,
            'attributes' : {
              'type' : self.subtype,
              'width' : self.width,
              'block-size' : self.block,
              'word-bits' : self.word,
            }}
    return dict




class SRAM(Memory):
  """
  Class for SRAM memory. Childeren of Memory.

  Methods
  -------
  get_TL_style_dict()
    returns the dictionary in TL format that describes the component
  """

  def __init__(self, name, size, width, banks, read_bw=32, write_bw=32, word=16):
    super().__init__("smartbuffer_SRAM", "", name, size, width, read_bw, write_bw, word)
    self.banks = banks


  def get_TL_style_dict(self):
    dict = {'name' : self.name,
            'class' : self.type,
            'attributes' : {
              'memory_width' : self.width,
              'n_banks' : self.banks,
              'memory_depth' : self.depth,
              'block-size': self.block,
              'word-bits' : self.word,
              'read_bandwidth' : self.read_bandwidth,
              'write_bandwidth' : self.write_bandwidth,
            }}
    return dict




class RF(Memory):
  """
  Class for RF memory. Childeren of Memory.

  Methods
  -------
  get_TL_style_dict()
    returns the dictionary in TL format that describes the component
  """

  def __init__(self, name, size, width, read_bw=2, write_bw=2, word=16):
    super().__init__("smartbuffer_RF", "", name, size, width, read_bw, write_bw, word)


  def get_TL_style_dict(self):
    dict = {'name' : self.name,
            'class' : self.type,
            'attributes' : {
              'memory_width' : self.width,
              'memory_depth' : self.depth,
              'block-size': self.block,
              'word-bits' : self.word,
              'read_bandwidth' : self.read_bandwidth,
              'write_bandwidth' : self.write_bandwidth,
            }}
    return dict




class MAC(Component):
  """
  Class for MAC. Childeren of Component..

  Methods
  -------
  get_TL_style_dict()
    returns the dictionary in TL format that describes the component
  """

  def __init__(self, type, name, datawidth, meshX=0):
    self.type = type
    self.name = name
    self.datawidth = datawidth
    self.meshX = meshX


  def get_TL_style_dict(self):
    dict = {'name' : self.name,
            'class' : self.type,
            'attributes' : {
              'datawidth' : self.datawidth,
            }}
    if self.meshX != 0:
      dict["attributes"]["meshX"] = self.meshX
    return dict




class Wires():

    def __init__(self, ifmap_bw=16, w_bw=16, ofmap_bw=16):
        self.ifmap_bw = ifmap_bw
        self.w_bw = w_bw
        self.ofmap_bw = ofmap_bw
        self.set_total_bw()

    def set_total_bw(self):
        self.read_bw= int((self.ifmap_bw + self.w_bw + (self.ofmap_bw/2)))
        self.write_bw= int(self.ofmap_bw / 2 )

    def refresh(self):
        self.set_total_bw()




class Accelerator():
  """
  Describes an accelerators.

  Attributes
  ----------
  technology : str
    e.g. '45nm'
  name : str
    name of the accelerator
  memory_hierarhcy : list
    list of memory components the accelerator has. L1 is at index 0
  levels : int
    total number of levels in the accelerator
  DRAM
  MAC

  Methods
  -------
  def get_level_as_dict(level)
    returns Timeloop style dictionary of memory level

  def get_memory_data_capacity(level)
    returns data capacity of memory level

  def get_last_level_memory_capacity()
    returns the memory capacity of the last level in memory (i.e. before DRAM)
  """

  def __init__(self, technology, name, memory_hierarchy, spatial_X=0, spatial_Y=0,
               spatial_level=0, ifmap_bw=4, ofmap_bw=4, w_bw=4, NoC_type="eyerissv1"):
    self.technology = technology
    self.name = name
    self.memory_hierarchy = memory_hierarchy
    self.unified_L1 = False if type(memory_hierarchy[0]) is list else True
    self.spatial_X = spatial_X
    self.spatial_Y = spatial_Y
    if self.spatial_X!=0 and self.spatial_Y!=0:
      self.PEs = self.spatial_X * self.spatial_Y
    elif self.spatial_X!=0:
      self.PEs = self.spatial_X
    else:
      self.PEs = 1
    self.spatial_level = spatial_level
    self.levels = len(self.memory_hierarchy) - 1 #DRAM excluded
    self.DRAM = self.memory_hierarchy[-1]
    if self.spatial_X!=0 and self.spatial_Y!=0:
      self.MAC = MAC(type="intmac", name="mac", datawidth=16, meshX=self.spatial_X)
    else:
      self.MAC = MAC(type="intmac", name="mac", datawidth=16)
    self.bw = Wires()
    self.NoC_type = NoC_type
    self.access_energies = None

  def configure_NoCs(self, tags=[6, 6, 6]):
    self.w_NoC = Eyeriss_NoC(vertical_mcus=self.spatial_Y, horizontal_mcus=self.spatial_X, name='eyeriss_noc', technology='45nm', voltage=1)
    self.w_NoC.add_component(component_name="mcu",
                                   component=MulticastController(tag_width=tags[0], data_width=self.bw.w_bw, tech=self.w_NoC.technology),
                                   repetition=self.spatial_Y * self.spatial_X)
    self.ofmap_NoC = Eyeriss_NoC(vertical_mcus=self.spatial_Y, horizontal_mcus=self.spatial_X, name='eyeriss_noc', technology='45nm', voltage=1)
    self.ofmap_NoC.add_component(component_name="mcu",
                                 component=MulticastController(tag_width=tags[1], data_width=self.bw.ofmap_bw, tech=self.ofmap_NoC.technology),
                                 repetition=self.spatial_Y * self.spatial_X)
    self.ifmap_NoC = Eyeriss_NoC(vertical_mcus=self.spatial_Y, horizontal_mcus=self.spatial_X, name='eyeriss_noc', technology='45nm', voltage=1)
    self.ifmap_NoC.add_component(component_name="mcu",
                                 component=MulticastController(tag_width=tags[2], data_width=self.bw.ifmap_bw, tech=self.ifmap_NoC.technology),
                                 repetition=self.spatial_Y * self.spatial_X)

  def get_level_as_dict(self, level):
    tensor_dedicated = False
    if not isinstance(level, str) and type(self.memory_hierarchy[level]) == list:
      tensor_dedicated = True

    if level == "dummy_buffer":
      dict = RF("dummy_buffer[0.." + str(self.spatial_X-1) + "]", size=256, width=16, read_bw=1, write_bw=1, word=16).get_TL_style_dict()
      dict["attributes"]["meshX"] = self.spatial_X
    else:
      if tensor_dedicated:
        dict1 = self.memory_hierarchy[level][0].get_TL_style_dict()
        dict2 = self.memory_hierarchy[level][1].get_TL_style_dict()
        dict3 = self.memory_hierarchy[level][2].get_TL_style_dict()
      else:
        dict = self.memory_hierarchy[level].get_TL_style_dict()
      if level < self.spatial_level and self.spatial_Y != 0:
        if tensor_dedicated:
          dict1["attributes"]["meshX"] = self.spatial_X
          dict2["attributes"]["meshX"] = self.spatial_X
          dict3["attributes"]["meshX"] = self.spatial_X
        else:
          dict["attributes"]["meshX"] = self.spatial_X

    if tensor_dedicated:
      return [dict1, dict2, dict3]
    else:
      return dict

  def get_level_as_dictv2(self, level):
    tensor_dedicated = False
    if not isinstance(level, str) and type(self.memory_hierarchy[level]) == list:
      tensor_dedicated = True

    if level == "dummy_buffer":
      dict = RF("dummy_buffer[0.." + str(self.spatial_X-1) + "]", size=256, width=16, read_bw=1, write_bw=1, word=16).get_TL_style_dict()
      dict["attributes"]["meshX"] = self.spatial_X
    else:
      if tensor_dedicated:
        dict1 = self.memory_hierarchy[level][0].get_TL_style_dict()
        dict2 = self.memory_hierarchy[level][1].get_TL_style_dict()
        dict3 = self.memory_hierarchy[level][2].get_TL_style_dict()
      else:
        dict = self.memory_hierarchy[level].get_TL_style_dict()
      if level < self.spatial_level and self.spatial_Y != 0:
        if tensor_dedicated:
          dict1["attributes"]["meshX"] = self.spatial_X
          dict2["attributes"]["meshX"] = self.spatial_X
          dict3["attributes"]["meshX"] = self.spatial_X
        else:
          dict["attributes"]["meshX"] = self.spatial_X

    if tensor_dedicated:
      return [dict3, dict2, dict1]
    else:
      return dict


  def get_memory_data_capacity(self, level):   #in number of words
    if self.unified_L1 or level > 0:
      return self.memory_hierarchy[level].data_capacity
    else:
      return [self.memory_hierarchy[level][tensor_type].data_capacity for tensor_type in range(3)]


  def get_last_level_memory_capacity(self):
    return self.memory_hierarchy[-2].data_capacity #-1 is DRAM


  def print_info(self):
    if self.spatial_level != 0:
      print("\n\nspatial accelerator with", self.spatial_X, "x", self.spatial_Y, "PEs")
    print(self.levels, "levels memory hierarchy")
    for level in range(self.levels, 0, -1):
      print("level", level, "has size ", self.get_memory_data_capacity(level-1))

  def get_as_txt(self):
    txt = ""
    txt += "L1: "
    if self.unified_L1:
      txt += str(self.memory_hierarchy[0].data_capacity)
    else:
      txt += str(self.memory_hierarchy[0][0].data_capacity) +\
             " - " + str(self.memory_hierarchy[0][1].data_capacity) +\
             " - " + str(self.memory_hierarchy[0][2].data_capacity)
    txt += "\nL2: " + str(self.memory_hierarchy[1].data_capacity)
    txt += "\nPEs: X:" + str(self.spatial_X) + " - Y:" + str(self.spatial_Y)
    txt += "\nR-BW:" + str(self.bw.read_bw)
    return txt
