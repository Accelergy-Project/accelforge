Parsing YAML Files
==================

YAML objects can include expressions that are parsed when they are loaded into Python.
To-be-parsed expressions can include Python code, and supported operations include many
standard library functions (*e.g.,* `range`, `min`) and functions from the `math`
standard library (*e.g.,* `log2`, `ceil).`

The scope available for parsing includes the following in order of increasing
precedence:

- Variables defined in a top-level :py:class:`~fastfusion.frontend.variables.Variables`
  object.
- Variables defined in outer-level YAML objects. Dictionary keys can be referenced by
  names, and list entries by index. The dot syntax can be used to access dictionaries;
  for example, `x.y.z` is equivalent to `outer_scope["x"]["y"]["z"]`.
- Variables defined in the current YAML object. Dictionary keys may reference each other
  as long as references are not cyclic.

The following is an example of valid parsed data:

.. code-block:: yaml

  variables:
    a: 123
    b: a + 5
    c: min(b, 3)
    d: sum(y for y in range(1, 10))

  # In some later scope
  ... outer_scope:
    x: 123
    y: a + x # Reference top-level variables
    inner_scope:
        a: 3 # Override outer scope
        b: outer_scope.x
        # Statements can be out-of-order if not cyclic referencing
        firt_item: second_item
        second_item: 3


The following are available expressions. In addition to the below, Python keywords that
are available witout import (*e.g.,* `min`) are also available

- `ceil`: `math.ceil``
- `comb``: `math.comb``
- `copysign``: `math.copysign``
- `fabs``: `math.fabs``
- `factorial``: `math.factorial``
- `floor``: `math.floor``
- `fmod``: `math.fmod``
- `frexp``: `math.frexp``
- `fsum``: `math.fsum``
- `gcd``: `math.gcd``
- `isclose``: `math.isclose``
- `isfinite``: `math.isfinite``
- `isinf``: `math.isinf``
- `isnan``: `math.isnan``
- `isqrt``: `math.isqrt``
- `ldexp``: `math.ldexp``
- `modf``: `math.modf``
- `perm``: `math.perm``
- `prod``: `math.prod``
- `remainder``: `math.remainder``
- `trunc``: `math.trunc``
- `exp``: `math.exp``
- `expm1``: `math.expm1``
- `log``: `math.log``
- `log1p``: `math.log1p``
- `log2``: `math.log2``
- `log10``: `math.log10``
- `pow``: `math.pow``
- `sqrt``: `math.sqrt``
- `acos``: `math.acos``
- `asin``: `math.asin``
- `atan``: `math.atan``
- `atan2``: `math.atan2``
- `cos``: `math.cos``
- `dist``: `math.dist``
- `hypot``: `math.hypot``
- `sin``: `math.sin``
- `tan``: `math.tan``
- `degrees``: `math.degrees``
- `radians``: `math.radians``
- `acosh``: `math.acosh``
- `asinh``: `math.asinh``
- `atanh``: `math.atanh``
- `cosh``: `math.cosh``
- `sinh``: `math.sinh``
- `tanh``: `math.tanh``
- `erf``: `math.erf``
- `erfc``: `math.erfc``
- `gamma``: `math.gamma``
- `lgamma``: `math.lgamma``
- `pi``: `math.pi``
- `e``: `math.e``
- `tau``: `math.tau``
- `inf``: `math.inf``
- `nan``: `math.nan``
- `abs``: `abs``
- `round``: `round``
- `pow``: `pow``
- `sum``: `sum``
- `range``: `range``
- `len``: `len``
- `min``: `min``
- `max``: `max``
- `float``: `float``
- `int``: `int``
- `str``: `str``
- `bool``: `bool``
- `list``: `list``
- `tuple``: `tuple``
- `enumerate``: `enumerate``
- `getcwd``: `os.getcwd``
- `map``: `map``


Jinja expressions are also available. (COPY TUTORIAL FROM BEFORE)