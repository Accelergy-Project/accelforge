How FFM Works
=============

Calling the Mapper
------------------

FFM consists of two stages: (1) exploring single-Einsum pmappings and (2) joining the pmappings.
You can run these stages using functions from the ffm package.

::

    from fastfusion.mapper.ffm import make_pmappings, join_pmappings

    pmappings = make_pmappings(spec)
    mappings = join_pmappings(spec, pmappings)

More information about these functions (*e.g.*, call signature, detailed documentation) can be
found here (TODO).

Interpreting the Output
-----------------------

There are three important attributes in the resulting ``mapping``:
- ``data`` is a ``pandas.DataFrame`` following a convention described below.
- ``total_mappings`` are the number of mappings that the mapper considered.
- ``valid_mappings`` are the number of valid mappings that the mapper considered.


Configuring the Mapper
----------------------

There are several knobs that can be used to control the mapspace explored by the mapper.