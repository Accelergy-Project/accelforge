Frequently Asked Questions
==========================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

What unit is ... specified in?
------------------------------
We use un-prefixed units for all values. Joules, seconds, meters, square meters, bits,
etc.

Why are some attributes underscored?
------------------------------------

.. _underscore-discussion:

Underscore prefixes are used to indicate that a value is recognized by the frontend.
They are used in places where there may be a mix of recognized and unrecognized values,
such as in a :py:class:`~fastfusion.frontend.arch.Component` ``attributes`` dictionary,
where ``attributes`` may contain recognized fields (such as
:py:obj:`~fastfusion.frontend.arch.ComponentAttributes.energy`) and
unrecognized fields (a field that may be used by `hwcomponents
<https://github.com/Accelergy-Project/hwcomponents>`_, but not this package).

When a value is underscored, this package will check whether it is recognized and raise
an error if it is not. In places where fields may or may not be recognized (e.g.,
:py:class:`~fastfusion.frontend.arch.ComponentAttributes`,
:py:class:`~fastfusion.frontend.arch.ActionArguments`), we recommend
underscore-prefixing all fields that are going to be used by this package.

As a result, you may see attributes dictionaries that have a mix of underscored and
non-underscored fields. The underscored fields will be used by this package, and the
non-underscored fields will only be used by other parsers of the object (such as
`hwcomponents <https://github.com/Accelergy-Project/hwcomponents>`_).

When an object is initialized with underscore-prefixed fields, all underscores are
dropped after checking validity.