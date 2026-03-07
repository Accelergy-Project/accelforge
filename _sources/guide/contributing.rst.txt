Contributing
============

We welcome contributions to AccelForge! This guide outlines the standards and practices
for contributing to the project.

- **Formatting**: All Python code should be formatted using `Black
  <https://black.readthedocs.io/>`_ with a line length of 88 characters (Black's
  default).
- **Type Hints**: All functions and methods should include type hints for parameters and
  return values.
- **Documentation**: All public functions, classes, and modules should include
  docstrings that clearly explain their purpose, parameters, and return values. We use
  `Google-style docstrings
  <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

Building the Documentation
---------------------------

Before submitting a pull request, ensure that the documentation builds without errors.

To build the documentation:

.. code-block:: bash

   make generate-docs

Your terminal will output a link that can be used to view the generated documentation
site. The build should complete with zero errors. Warnings should also be minimized
where possible. If you add new modules or modify docstrings, verify that:

- All cross-references resolve correctly
- Code examples render properly
- API documentation is complete
- No broken links exist

Testing
-------

Before submitting changes, run the test suite to ensure your modifications don't break
existing functionality:

.. code-block:: bash

   python3 -m unittest discover -s tests -p "*.py" -v

Add tests for any new functionality you introduce. Tests should cover both valid inputs
and error cases with appropriate error messages.

Pull Request Process
--------------------

1. Fork the repository and create a new branch for your changes
2. Make your changes following the guidelines above
3. Format your code with Black
4. Build the documentation and verify it's error-free
5. Run the test suite
6. Submit a pull request with a clear description of your changes
