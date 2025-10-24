import sys, os
sys.path.append(os.getcwd())
from src.tiling import Problem

class NN():

    def __init__(self):
        self.L1_optimals = []
        self.L2_optimals = []
        self.layers = []
        self.L1_average = 0
        self.L2_average = 0

    def get_layers(self):
        return self.layers

    def get_layers_backward_pass(self):
        backward_pass_layers = []
        for layer in self.layers:
            backward_pass_layer = Problem(n=layer.n, c=layer.c, m=layer.m, r=layer.r, s=layer.s,
                                          p=(layer.p + layer.r - 1), q=(layer.q + layer.s - 1), duplicate=layer.duplicate)
            backward_pass_layers.append(backward_pass_layer)
        return backward_pass_layers

    def get_layers_weight_update(self):
        weight_update_layers = []
        for layer in self.layers:
            weight_update_layer = Problem(n=layer.c, c=layer.n, m=layer.m, r=layer.p, s=layer.q,
                                          p=layer.r, q=layer.s, duplicate=layer.duplicate)
            weight_update_layers.append(weight_update_layer)
        return weight_update_layers



#***************************************************************************
#********************************** AlexNet ********************************
class Alex_Net_Problem(NN):

    def __init__(self, n=1):
      super().__init__()
      self.layers = [Problem(n=n,  c=96, m=256, r=5, s=5, p=28, q=28),
                     Problem(n=n, c=256, m=384, r=3, s=3, p=13, q=13),
                     Problem(n=n, c=384, m=384, r=3, s=3, p=13, q=13),
                     Problem(n=n, c=384, m=256, r=3, s=3, p=13, q=13)
                    ]




#***************************************************************************
#********************************** VGG-16 *********************************
class VGG16(NN):

    def __init__(self, n=1):
        super().__init__()
        self.layers = [Problem(n=n,  c=3, m=64, r=3, s=3, p=112, q=112),
                       Problem(n=n,  c=64, m=64, r=3, s=3, p=112, q=112),

                       Problem(n=n,  c=64, m=128, r=3, s=3, p=64, q=64),
                       Problem(n=n,  c=128, m=128, r=3, s=3, p=64, q=64),

                       Problem(n=n,  c=128, m=256, r=3, s=3, p=32, q=32),
                       Problem(n=n,  c=256, m=256, r=3, s=3, p=32, q=32),
                       Problem(n=n,  c=256, m=256, r=3, s=3, p=32, q=32),

                       Problem(n=n,  c=256, m=512, r=3, s=3, p=16, q=16),
                       Problem(n=n,  c=512, m=512, r=3, s=3, p=16, q=16),
                       Problem(n=n,  c=512, m=512, r=3, s=3, p=16, q=16),

                       Problem(n=n,  c=512, m=512, r=3, s=3, p=8, q=8),
                       Problem(n=n,  c=512, m=512, r=3, s=3, p=8, q=8),
                       Problem(n=n,  c=512, m=512, r=3, s=3, p=8, q=8),

                       Problem(n=n,  c=512, m=4096, r=1, s=1, p=1, q=1),
                       Problem(n=n,  c=4096, m=4096, r=1, s=1, p=1, q=1),
                       Problem(n=n,  c=4096, m=1024, r=1, s=1, p=1, q=1),
                      ]




#***************************************************************************
#********************************* Resnet-18 *******************************
class ResNet18(NN):
    def __init__(self, n=1):
        super().__init__()
        self.layers = [Problem(n=n,  c=3, m=64, r=7, s=7, p=112, q=112),

                       Problem(n=n, c=64, m=64, r=3, s=3, p=56, q=56),
                       Problem(n=n, c=64, m=64, r=3, s=3, p=56, q=56, duplicate=True),
                       Problem(n=n, c=64, m=64, r=3, s=3, p=56, q=56, duplicate=True),
                       Problem(n=n, c=64, m=64, r=3, s=3, p=56, q=56, duplicate=True),

                       Problem(n=n, c=64, m=128, r=3, s=3, p=28, q=28),
                       Problem(n=n, c=128, m=128, r=3, s=3, p=28, q=28),
                       Problem(n=n, c=128, m=128, r=3, s=3, p=28, q=28, duplicate=True),
                       Problem(n=n, c=128, m=128, r=3, s=3, p=28, q=28, duplicate=True),

                       Problem(n=n, c=128, m=256, r=3, s=3, p=14, q=14),
                       Problem(n=n, c=256, m=256, r=3, s=3, p=14, q=14),
                       Problem(n=n, c=256, m=256, r=3, s=3, p=14, q=14, duplicate=True),
                       Problem(n=n, c=256, m=256, r=3, s=3, p=14, q=14, duplicate=True),

                       Problem(n=n, c=256, m=512, r=3, s=3, p=7, q=7),
                       Problem(n=n, c=512, m=512, r=3, s=3, p=7, q=7),
                       Problem(n=n, c=512, m=512, r=3, s=3, p=7, q=7, duplicate=True),
                       Problem(n=n, c=512, m=512, r=3, s=3, p=7, q=7, duplicate=True),

                       Problem(n=n, c=1, m=1, r=7, s=7, p=1, q=1, depth=512, depthwise=True),
                       Problem(n=n, c=512, m=1000, r=1, s=1, p=1, q=1),
                      ]




#***************************************************************************
#****************************** Bottleneck Block ***************************
class Bottleneck():

    def __init__(self, n, c, m, expansion, p, q, r=3, s=3, stride=1, name="", duplicate=False, duplicate_tail=False):
        local_duplicate = (duplicate_tail or duplicate)
        self.layers = [ Problem(n=n, c=c, m=c*expansion, r=1, s=1, p=p*stride, q=q*stride, duplicate=duplicate),
                        Problem(n=n, c=1, m=1, r=r, s=s, p=p, q=q, depthwise=True, depth=c*expansion, duplicate=local_duplicate),
                        Problem(n=n, c=c*expansion, m=m, r=1, s=1, p=p, q=q, duplicate=local_duplicate)
                      ]




#***************************************************************************
#****************************** MobileNet v2 ***************************
class MobileNet_V2(NN):

    def __init__(self, n=1):
        super().__init__()

        self.blocks = [Bottleneck(n=n, c=32, m=16, expansion=1, p=112, q=112, stride=1, name="B1"),
                   Bottleneck(n=n, c=16, m=24, expansion=6, p=56, q=56, stride=2, name="B2"),
                   Bottleneck(n=n, c=24, m=24, expansion=6, p=56, q=56, stride=1, name="B3", duplicate_tail=True),
                   Bottleneck(n=n, c=24, m=32, expansion=6, p=28, q=28, stride=2, name="B4"),
                   Bottleneck(n=n, c=32, m=32, expansion=6, p=28, q=28, stride=1, name="B5", duplicate_tail=True),
                   Bottleneck(n=n, c=32, m=32, expansion=6, p=28, q=28, stride=1, name="B6", duplicate=True),
                   Bottleneck(n=n, c=32, m=64, expansion=6, p=14, q=14, stride=2, name="B7"),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=14, q=14, stride=1, name="B8", duplicate_tail=True),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=14, q=14, stride=1, name="B9", duplicate=True),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=14, q=14, stride=1, name="B10", duplicate=True),
                   Bottleneck(n=n, c=64, m=96, expansion=6, p=14, q=14, stride=1, name="B11"),
                   Bottleneck(n=n, c=96, m=96, expansion=6, p=14, q=14, stride=1, name="B12", duplicate_tail=True),
                   Bottleneck(n=n, c=96, m=96, expansion=6, p=14, q=14, stride=1, name="B13", duplicate=True),
                   Bottleneck(n=n, c=96, m=160, expansion=6, p=7, q=7, stride=2, name="B14"),
                   Bottleneck(n=n, c=160, m=160, expansion=6, p=7, q=7, stride=1, name="B15", duplicate_tail=True),
                   Bottleneck(n=n, c=160, m=160, expansion=6, p=7, q=7, stride=1, name="B16", duplicate=True),
                   Bottleneck(n=n, c=160, m=320, expansion=6, p=7, q=7, stride=1, name="B17"),
                  ]

        for block in self.blocks:
            for layer in block.layers:
                self.layers.append(layer)
        self.layers.append(Problem(n=n, c=320, m=1280, r=1, s=1, p=7, q=7))
        self.layers.append(Problem(n=n, c=1280, m=1008, r=1, s=1, p=1, q=1))




class MobileNet_V2_cifar10(NN):

  def __init__(self, n=1):
    super().__init__()

    self.blocks = [Bottleneck(n=n, c=32, m=16, expansion=1, p=32, q=32, stride=1, name="B1"),
                   Bottleneck(n=n, c=16, m=24, expansion=6, p=32, q=32, stride=2, name="B2"),
                   Bottleneck(n=n, c=24, m=24, expansion=6, p=32, q=32, stride=1, name="B3"),
                   Bottleneck(n=n, c=24, m=32, expansion=6, p=16, q=16, stride=2, name="B4"),
                   Bottleneck(n=n, c=32, m=32, expansion=6, p=16, q=16, stride=1, name="B5"),
                   Bottleneck(n=n, c=32, m=32, expansion=6, p=16, q=16, stride=1, name="B6"),
                   Bottleneck(n=n, c=32, m=64, expansion=6, p=16, q=16, stride=2, name="B7"),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=16, q=16, stride=1, name="B8"),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=16, q=16, stride=1, name="B9"),
                   Bottleneck(n=n, c=64, m=64, expansion=6, p=16, q=16, stride=1, name="B10"),
                   Bottleneck(n=n, c=64, m=96, expansion=6, p=16, q=16, stride=1, name="B11"),
                   Bottleneck(n=n, c=96, m=96, expansion=6, p=16, q=16, stride=1, name="B12"),
                   Bottleneck(n=n, c=96, m=96, expansion=6, p=16, q=16, stride=1, name="B13"),
                   Bottleneck(n=n, c=96, m=160, expansion=6, p=8, q=8, stride=2, name="B14"),
                   Bottleneck(n=n, c=160, m=160, expansion=6, p=8, q=8, stride=1, name="B15"),
                   Bottleneck(n=n, c=160, m=160, expansion=6, p=8, q=8, stride=1, name="B16"),
                   Bottleneck(n=n, c=160, m=320, expansion=6, p=8, q=8, stride=1, name="B17"),
                  ]

    for block in self.blocks:
      for layer in block.layers:
        self.layers.append(layer)
    self.layers.append(Problem(n=n, c=320, m=1280, r=1, s=1, p=7, q=7))
    self.layers.append(Problem(n=n, c=1280, m=1008, r=1, s=1, p=1, q=1))









#Inceptionv3 Architecture
class Inceptionv3(NN):

    def __init__(self, n=1):
        super().__init__()
        self.layers.append(Problem(n, 3, 32, 3, 3, 149, 149))
        self.layers.append(Problem(n, 32, 32, 3, 3, 149, 149))
        self.layers.append(Problem(n, 32, 64, 3, 3, 147, 147))

        self.layers.append(Problem(n, 64, 80, 1, 1, 73, 73))
        self.layers.append(Problem(n, 80, 192, 3, 3, 71, 71))

        module = InceptionModuleA(n, 192, 32, 35, 35)
        self.layers += module.layers
        module = InceptionModuleA(n, 256, 64, 35, 35, duplicate=True)
        self.layers += module.layers
        module = InceptionModuleA(n, 288, 64, 35, 35, duplicate=True)
        self.layers += module.layers

        module = InceptionModuleB(n, 288, 35, 35)
        self.layers += module.layers

        module = InceptionModuleC(n, 768, 128, 17, 17)
        self.layers += module.layers
        module = InceptionModuleC(n, 768, 160, 17, 17)
        self.layers += module.layers
        module = InceptionModuleC(n, 768, 160, 17, 17, duplicate=True)
        self.layers += module.layers
        module = InceptionModuleC(n, 768, 192, 17, 17)
        self.layers += module.layers

        module = InceptionModuleD(n, 768, 17, 17)
        self.layers += module.layers

        module = InceptionModuleE(n, 1280, 8, 8)
        self.layers += module.layers
        module = InceptionModuleE(n, 2048, 8, 8, duplicate=True)
        self.layers += module.layers

        self.layers.append(Problem(n, 2048, 1000, 1, 1, 1, 1,))



class InceptionModuleA():

    def __init__(self, n, in_ch, channel_pool, p, q, duplicate=False):
        self.n = n
        self.in_ch = in_ch
        self.channel_pool = channel_pool
        self.p = p
        self.q = q

        self.layers = [Problem(self.n, self.in_ch, 64, 1, 1, self.p, self.q),

                       Problem(self.n, self.in_ch, 48, 1, 1, self.p, self.q),
                       Problem(self.n, 48, 64, 5, 5, self.p, self.q, duplicate=duplicate),

                       Problem(self.n, self.in_ch, 64, 1, 1, self.p, self.q),
                       Problem(self.n, 64, 96, 3, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 96, 96, 3, 3, self.p, self.q, duplicate=duplicate),
                       #Ave pooling commented out
                       #Problem(self.n, self.in_ch, self.in_ch, 3, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, self.in_ch, self.channel_pool, 1, 1, self.p, self.q), #PW after ave. pooling
                      ]




class InceptionModuleB():

    def __init__(self, n, in_ch, p, q, duplicate=False):
        self.n = n
        self.p = p
        self.q = q
        self.in_ch = in_ch

        self.layers = [Problem(self.n, self.in_ch, 384, 3, 3, int(self.p/2), int(self.q/2), duplicate=duplicate),
                       Problem(self.n, self.in_ch, 64, 1, 1, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 64, 96, 3, 3, self.p, self.q, duplicate=duplicate),
                       #Max pooling commented out
                       #Problem(self.n, 96, 96, 3, 3, int(self.p/2), int(self.q/2), duplicate=duplicate),
                      ]




class InceptionModuleC():

    def __init__(self, n, in_ch, channles_7_by_7, p, q, duplicate=False):
        self.n = n
        self.p = p
        self.q = q
        self.in_ch = in_ch
        self.channles_7_by_7 = channles_7_by_7

        self.layers = [Problem(self.n, self.in_ch, 192, 1, 1, self.p, self.q, duplicate=duplicate),

                       Problem(self.n, self.in_ch, channles_7_by_7, 1, 1, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, channles_7_by_7, 1, 7, self.p, self.q+3, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, 192, 7, 1, self.p+3, self.q, duplicate=duplicate),

                       Problem(self.n, self.in_ch, channles_7_by_7, 1, 1, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, channles_7_by_7, 7, 1, self.p+3, self.q, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, channles_7_by_7, 1, 7, self.p, self.q+3, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, channles_7_by_7, 7, 1, self.p+3, self.q, duplicate=duplicate),
                       Problem(self.n, channles_7_by_7, 192, 1, 7, self.p, self.q+3, duplicate=duplicate),

                       #Ave pooling commented out
                       #Problem(self.n, self.in_ch, self.in_ch, 3, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, self.in_ch, 192, 1, 1, self.p, self.q, duplicate=duplicate), #PW after ave. pooling
                      ]


class InceptionModuleD():

    def __init__(self, n, in_ch, p, q, duplicate=False):
        self.n = n
        self.p = p
        self.q = q
        self.in_ch = in_ch

        self.layers = [Problem(self.n, self.in_ch, 192, 1, 1, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 192, 320, 3, 3, int(self.p/2), int(self.q/2), duplicate=duplicate),

                       Problem(self.n, self.in_ch, 192, 1, 1, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 192, 192, 1, 7, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 192, 192, 7, 1, self.p, self.q, duplicate=duplicate),
                       #Max pooling commented out
                       #Problem(self.n, 192, 192, 3, 3, int(self.p/2), int(self.q/2), duplicate=duplicate),
                      ]



class InceptionModuleE():

    def __init__(self, n, in_ch, p, q, duplicate=False):
        self.n = n
        self.p = p
        self.q = q
        self.in_ch = in_ch

        self.layers = [Problem(self.n, self.in_ch, 320, 1, 1, self.p, self.q),

                       Problem(self.n, self.in_ch, 384, 1, 1, self.p, self.q),
                       Problem(self.n, 384, 384, 1, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 384, 384, 3, 1, self.p, self.q, duplicate=duplicate),

                       Problem(self.n, self.in_ch, 448, 1, 1, self.p, self.q),
                       Problem(self.n, 448, 384, 3, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 384, 384, 1, 7, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, 384, 384, 7, 1, self.p, self.q, duplicate=duplicate),
                       #Ave pooling commented out
                       #Problem(self.n, self.in_ch, self.in_ch, 3, 3, self.p, self.q, duplicate=duplicate),
                       Problem(self.n, self.in_ch, 192, 1, 1, self.p, self.q), #PW after ave. pooling
                      ]
