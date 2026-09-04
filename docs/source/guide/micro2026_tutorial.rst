MICRO 2026 Tutorial
===================

.. _micro-tutorial:

We will be hosting a tutorial at MICRO 2026 on AccelForge, including how to use it and
its features.

Tutorial Content
----------------

The tutorial includes six main sections. The first five sections cover the design space
explored by AccelForge. The final section concludes with demonstrations of design space
exploration and co-design using AccelForge.

Workload: Expressing Tensor Algebra Workloads with Einsums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anyone trying to make sense of a DNN by reading through its PyTorch implementation will
quickly realize that it is a challenging, in part because the representation mixes
together the computation with how it is to be executed on accelerators.

This section introduces the Einsum notation, a concise and precise way to specify tensor
algebra workloads. The Einsum notation decouples what computations are performed from
how accelerators execute them, and lets you concisely describe tensor algebra workload
computations (including LLMs) in just a handful of lines. Finally, we describe how one
may transform Einsums to realize new optimizations (e.g., FlashAttention).

Mapping: Expressing Mappings with LoopTrees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a workload, we need a mapping that specifies how the workload is to be executed
on the accelerator. However, many works introduce mappings under varying and
often-limited vocabularies, making it challenging to understand or compare mappings
across works.

This section introduces LoopTrees, a concise and precise way to specify how workloads
are mapped onto accelerators. LoopTree’s flexibility enables exploration of a superset
of state-of-the-art mapping optimizations (e.g., fusion and uneven mappings), while
making data movement, parallelism, and memory usage trade-offs intuitive to understand
from the mapping. Using LoopTrees, we analyze key tradeoffs in mapping, including data
movement, memory usage, and parallelism. We then look at state-of-the-art mapping works,
analyze their tradeoffs using LoopTrees, and show how to increase their performance by
leveraging what we've learned.

Mapping: Finding Optimal Mappings with TCM and FFM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating an optimal mapping is challenging, even state-of-the-art mappings that took
domain experts months to design and optimize often fall short of optimal. Moreover, the
optimal mapping changes drastically with even small changes to the workload or
architecture, making it necessary to automate mapping generation.

This section introduces TCM and FFM, the first mappers to find optimal mappings for a
wide range of workloads and architectures and a comprehensive space of mapping choices.
This section shows you how to use these mappers to generate optimal mappings given a
wide range of workload and architectures. We then give a deep dive to show how TCM and
FFM quickly find these mappings, then show how to use these mappers to improve
accelerator performance over human-expert-optimized mappings.

Components: Building and Incorporating Components with HWComponents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An accelerator needs to be built from components, including devices and circuits.
However, without a library of pre-made components, works will re-implement components or
introduce different components (with different assumptions), making it challenging to
compare works. The opposite is also true: Devices and circuits must be evaluated in the
context of an accelerator. However, without a framework to easily incorporate components
into accelerators, works may evaluate components in isolation, which fails to reveal
their true performance impacts.

This section describes how to use HWComponents to incorporate component (device and
circuit) designs into accelerators. We show how to leverage the included HWComponents
library of components to quickly model new accelerator designs. We also show how to
modify component parameters to realize new components, and how to create fully-new
component designs and evaluate them in accelerators.

Architecture: Expressing and Exploring Key Architecture Decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accelerator architectures are complex, and many accelerator works describe individual
architectures with different vocabularies, making it challenging to understand or
compare architectures across works.

This section describes the AccelForge architecture representation, which concisely and
precisely describes key accelerator properties, including memory hierarchy, parallelism,
and ability to capture reuse. We show how to interperet the tradeoffs of different
architecture decisions, including per-component benefits and costs, to find bottlenecks
and improve performance.

Using AccelForge for Design Space Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we show that AccelForge is unique in its design space exploration ability
because it incoporates all of the above sections into one framework. We show how to use
AccelForge to explore accelerators, showing how to leverage tradeoffs and synergies
across the design space to find better accelerator designs.

Tutorial Abstract
-----------------

Deep neural networks (DNNs) are energy-intensive workloads in modern datacenter and edge
deployments, making accelerators necessary to achieve energy efficiency and high
throughput. To quickly evaluate and iterate on accelerator designs, we need an
accelerator modeling framework that captures salient attributes of devices, circuits,
architectures, workloads, as well as optimizing the mapping of the workload onto the
hardware.

In this tutorial, we introduce AccelForge, which improves upon existing accelerator
modeling framework in both capabilities and ease-of-use. AccelForge unifies and extends
eight (and counting) published and in-progress works into one framework, and it includes
(1) composable user-defined and user-modifiable models of devices, circuits, and
architectures, (2) fast mappers that enable accurate evaluation in orders of magnitude
less (computer and human) time, (3) easy-to-use and easy-to-extend Python
implementations of both the model and mapper to enable rapid research and extension to
novel optimizations. The AccelForge framework has been used in various projects,
including teaching Hardware Architecture for Deep Learning (6.5930/1) at MIT.

First, we describe how to specify and evaluate a combination of: hardware design
(including device, circuit, and architecture); a workload; and the mapping of the
workload to the architecture . As part of this tutorial, we also discuss key works that
we have integrated into AccelForge. These works include: CiMLoop [ISPASS’24], a model
that allows for device and circuit exploration and integration into architectures; the
extended Einsum notation, a concise representation of DNN workloads (and tensor algebra
workloads generally); and LoopTree [TCASAI’24], a clear, concise representation of the
ways workloads may be mapped onto accelerators.

However, manual exploration of mappings is infeasible, so we also provide a deeper dive
into how AccelForge automatically optimizes the mapping. We demonstrate why a fast and
optimal mapper is essential to perform accelerator design space exploration, including
common pitfalls seen in both prior works, such as misusing hardware resources (e.g.,
mappers using too-large tiles, causing accelerator designers to over-provision buffer
sizes) or optimizing for a narrow use case (e.g., forcing a computation order that
causes re-fetches of a weight tensor is bad when weight tensors are large). We then show
how to use AccelForge’s Fast and Fusiest and Turbo-Charged mappers, which guarantee
finding optimal mappings and therefore guarantee well-informed design space exploration.
We demonstrate how to use these mappers to perform mapping search, avoid the shown
pitfalls, and explore the accelerator design space to find architectures better in
throughput, area, and energy.

Finally, we demonstrate how to use AccelForge to perform design space exploration with
case studies that show the benefits of co-design. We focus on new studies enabled by
AccelForge, including (but not limited to): an exploration of the tradeoffs around
operator fusion and how it interacts with accelerator design; a demonstration of the
importance of comprehensive searches; and a demonstration of co-design that reveals new
synergies.

Presenter Bios
--------------

Tanner Andrulis is a PhD candidate at MIT working on building energy-efficient deep
neural network accelerators through full-stack codesign. His research focuses on
building modeling tools that integrate all levels of the stack and allow for rapid
design space exploration and evaluation of accelerator designs. Tanner received B.S.
degrees in Computer Engineering and Math from Purdue University in 2021, and his
Master’s from MIT in 2023.

Michael Gilbert is a PhD candidate at MIT working on energy-efficient accelerator
architectures and tools to enable early-stage architecture design space exploration. His
current interests include optimization of workload-to-architecture mapping, and
energy-efficient architectures for the data center. Michael received his M.Eng in
computer science from MIT in 2023, and his B.S. in computer science from MIT in 2022.

Joel S. Emer is a professor in the Electrical Engineering and Computer Science
Department (EECS) at MIT and a member of the Computer Science and Artificial
Intelligence Laboratory (CSAIL). He is also a Senior Distinguished Research Scientist at
Nvidia in Westford, MA, where he is responsible for exploration of future architectures
as well as modeling and analysis methodologies.

Vivienne Sze is a professor in the Electrical Engineering and Computer Science
Department (EECS) at MIT. She works on computing systems that enable energy-efficient
machine learning, computer vision, and video compression/processing for a wide range of
applications, including autonomous navigation, digital health, and the internet of
things.
