Frequently Asked Questions
==========================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

What unit is ... specified in?
------------------------------
We use un-prefixed units for all values. Joules, seconds, meters, square meters, bits,
etc. If you're working internally in the code, you may see reservations as well; these
are a porportion of maximum, from 0 to 1.

In what order are area, energy, latency, and leak power parameters and names?
-----------------------------------------------------------------------------
Wherever applicable, we follow alphabetical order of area, energy, latency, and leak
power. For example, the main function for calculating these attributes is
:py:meth:`~accelforge.frontend.spec.Spec.calculate_component_area_energy_latency_leak`.
The same convention is followed in `hwcomponents
<https://github.com/accelergy-project/hwcomponents>`_.
