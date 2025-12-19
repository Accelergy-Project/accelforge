..
  TODO: there is actually a syntax for defining terms, something like
    term-name
      definition

    other-term
      more-definition

  We should use that so that the terms can be referenced using the :term: directive.

..
  I would do it myself, but I'm not sure what the :label: directives are doing, and I don't
  want to break anything by editing it.


Definitions
===========
- **Action**: :label:`Action`, :label:`Actions` An action is something performed by a
  hardware unit. For example, a read or a compute.

- **Mapping**: :label:`Mapping`, :label:`Mappings` A *mapping* is a schedule that maps
  operations and data movement onto the hardware.

- **Component**: :label:`Component`, :label:`Components` A component is a hardware unit
  in the architecture. For example, a memory or a compute unit.

- **Dataflow**: :label:`Dataflow`, :label:`Dataflows` The order in which a mappings
  :ref:`Mapping` iterates over tiles, noting that tiles may be abstract before the
  mapping is fully defined. :ref:`Tile`.

- **Dataplacement**: :label:`Dataplacement`, :label:`Dataplacements` Which tile(s) are
  stored in each memory level of the accelerator, and for what time period, noting that
  tiles and time periods may be abstract before the mapping is fully defined. :ref:`Tile`.

- **Pmapping**: :label:`Pmapping`, :label:`Pmappings` A *partial mapping*, or
  *pmapping*, is a mapping of a subset of the workload to the hardware.

- **Tile**: :label:`Tile`, :label:`Tiles`