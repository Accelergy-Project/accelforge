from abc import ABC, abstractmethod


class Primitive(ABC):

    def __init__(self, actions, attributes, type, name, tech="40nm"):
        self.compound = False
        self.actions = actions
        self.attributes = attributes
        self.type = type
        self.name = name
        self.tech=tech




class Compound(ABC):

    def __init__(self, attributes, actions_highlevel, actions_subcomponent_map, name, tech="40nm"):
        self.compound = True
        self.attributes = attributes
        self.actions_highlevel = actions_highlevel
        self.actions_subcomponent_map = actions_subcomponent_map
        self.tech=tech
        self.name = name
        self.type = self.name
        self.subcomponents = {}





class BitWise(Primitive):

    def __init__(self, datawidth, inputs, name, tech="40nm"):
        super().__init__(actions=['process', 'idle'],
                         attributes={"datawidth":datawidth, "num":inputs, "technology":tech},
                         type='bitwise',
                         name=name,
                         tech=tech)
        self.datawidth = datawidth
        self.inputs = inputs




class Comparator(Primitive):

    def __init__(self, datawidth, name, tech="40nm"):
        super().__init__(actions=['compare', 'idle'],
                          attributes={"datawidth":datawidth, "technology":tech},
                          type='comparator',
                          name=name,
                          tech=tech)
        self.datawidth = datawidth




class Wire(Primitive):

    def __init__(self, name, length="1um", tech="40nm"):
        super().__init__(actions=['transfer_random', 'transfer_repeated', 'idle'],
                         attributes={"length":length, "technology":tech},
                         type='wire',
                         name=name,
                         tech=tech)
        self.length = length




class Mux(Compound):

    def __init__(self, datawidth, inputs, tech="40nm"):
        super().__init__(actions_highlevel=['process', 'idle'],
                         attributes={'data_width':datawidth, 'inputs':inputs, 'technology':tech},
                         actions_subcomponent_map =
                         {
                            'NOT_gate':{'process':['process'], 'idle':['idle']},
                            "AND_gate":{'process':['process'], 'idle':['idle']}
                         },
                         name="mux",
                         tech=tech)
        self.address_bits = inputs.bit_length()
        self.inputs = inputs
        self.NOT_gates = [BitWise(1, 2, name=self.name+"_NOT_gate", tech=tech) for i in range(self.address_bits * inputs)]
        self.AND_gates = [BitWise(1, 1+self.address_bits, name=self.name+"_AND_gate", tech=tech) for i in range(self.inputs * datawidth)]
        self.subcomponents = {
            "NOT_gate" : self.NOT_gates,
            "AND_gate" : self.AND_gates
        }



class MulticastController(Compound):

    def __init__(self, tag_width, data_width, tech="40nm"):
        super().__init__(attributes={'data_width':data_width, 'tag_width':tag_width, 'technology':tech},
                         actions_highlevel=['check_tag', 'idle'],
                         actions_subcomponent_map =
                         {
                            'AND_gate' : {'check_tag':['process'], 'idle':['idle']},
                            'comparator' : {'check_tag':['compare'], 'idle':['idle']},
                            'mux' : {'check_tag':['process'], 'idle':['idle']},
                         },
                         name = "multicast_controller",
                         tech=tech)
        self.comparators = [Comparator(tag_width,  name=self.name+"_comparator", tech=tech)]
        self.AND_gates = [BitWise(datawidth=1, inputs=3, name=self.name+"_AND_gate", tech=tech)]
        self.mux = [Mux(datawidth=data_width, inputs=2, tech=tech)]
        self.subcomponents = {
            "comparator" : self.comparators,
            "mux" : self.mux,
            "AND_gate" : self.AND_gates
        }



class Accelergy_Hardware():
    def __init__(self, name, technology="45nm", voltage=1):
        #{name:{'component':component, 'repetitions':repetitions}}
        self.components = {}
        self.name = name
        self.technology = technology

    def add_component(self, component_name, component, repetition=1):
        self.components[component_name]={'component':component, 'repetition':repetition}




class Eyeriss_NoC(Accelergy_Hardware):
    def __init__(self, vertical_mcus, horizontal_mcus, name, technology, voltage=1):
        super().__init__(name=name, technology=technology, voltage=voltage)
        self.vertical_mcus = vertical_mcus
        self.horizontal_mcus = horizontal_mcus
        self.attributes = {'technology':technology, 'voltage':voltage}
