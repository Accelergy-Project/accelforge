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

- **Mapping**: :label:`Mapping`, :label:`Mappings` A *mapping* is a schedule that maps
  operations and data movement onto the hardware.

- **Component**: :label:`Component`, :label:`Components` A component is a hardware unit
  in the architecture. For example, a
  memory or a compute unit.

- **Action**: :label:`Action`, :label:`Actions` An action is something performed by a
  hardware unit. For example, a read or a compute.

- **Pmapping**: :label:`Pmapping`, :label:`Pmappings` A *partial mapping*, or *pmapping*, is a mapping of a
  subset of the
  workload to the hardware.
