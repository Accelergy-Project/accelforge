Contributing
============

We welcome contributions to AccelForge! This guide outlines the standards and practices
for contributing to the project.

Code Formatting
---------------

All Python code should be formatted using `Black <https://black.readthedocs.io/>`_ with
a line length of 88 characters (Black's default).

To format your code, install Black and run:

.. code-block:: bash

   pip install black
   black .

You can also set up your editor to format on save, or add a pre-commit hook to
automatically format your code before committing.

Type Hints
----------

All functions and methods should include type hints for parameters and return values.
This improves code clarity and enables better IDE support and static analysis.

Example:

.. code-block:: python

   def calculate_energy(
       operations: int,
       energy_per_op: float,
       voltage: float = 1.0
   ) -> float:
       """Calculate total energy consumption.

       Args:
           operations: Number of operations performed
           energy_per_op: Energy consumed per operation (pJ)
           voltage: Operating voltage (V)

       Returns:
           Total energy consumed (pJ)
       """
       return operations * energy_per_op * (voltage ** 2)

Documentation
-------------

All public functions, classes, and modules should include docstrings that clearly
explain their purpose, parameters, and return values. We use
`Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

Documentation should include:

- **Module docstrings**: Brief description of the module's purpose
- **Class docstrings**: Purpose of the class and any important attributes
- **Function/method docstrings**:

  - Brief description
  - Args section describing each parameter
  - Returns section describing the return value
  - Raises section for any exceptions (if applicable)
  - Examples section for complex functionality (if helpful)

Building the Documentation
---------------------------

Before submitting a pull request, ensure that the documentation builds without errors.

To build the documentation:

.. code-block:: bash

   cd docs
   make html

The documentation will be built in ``docs/build/html/``. Open ``docs/build/html/index.html``
in your browser to view the results.

**Important**: The build should complete with zero errors. Warnings should also be
minimized where possible. If you add new modules or modify docstrings, verify that:

- All cross-references resolve correctly
- Code examples render properly
- API documentation is complete
- No broken links exist

Testing
-------

Before submitting changes, run the test suite to ensure your modifications don't break
existing functionality:

.. code-block:: bash

   python3 -m unittest ./tests/*.py

Add tests for any new functionality you introduce.

Pull Request Process
--------------------

1. Fork the repository and create a new branch for your changes
2. Make your changes following the guidelines above
3. Format your code with Black
4. Build the documentation and verify it's error-free
5. Run the test suite
6. Submit a pull request with a clear description of your changes

Your pull request should:

- Have a descriptive title
- Include a summary of the changes and motivation
- Reference any related issues
- Include tests for new functionality
- Pass all CI checks
