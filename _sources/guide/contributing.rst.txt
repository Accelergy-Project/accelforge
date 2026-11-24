Contributing
============

We welcome contributions to AccelForge! This guide outlines the standards and practices
for contributing to the project.

The Goals of AccelForge
-----------------------

In our website, we define AccelForge to be "a framework to model and design tensor
algebra and deep neural network accelerators."

We take this identity seriously. The primary goal of AccelForge is to enable people
(with or without AI assistance) to model and design accelerators. When contributing to
AccelForge, please consider whether your contributions furthers this goal.

The goals of AccelForge can be summarized as follows:

- Be fast
- Be accurate
- Be easy to use and extend
- Be applicable to a wide variety of accelerator designs

Your contributions should be evaluated against these goals, both in terms of how well
they are met *and* how well an ideal version of your contribution would meet them. It is
rare that a contribution is strictly positive on all fronts; adding a feature or making
the mapper faster will likely add code complexity and make it more challenging to extend
the codebase in the future. Some examples of contributions that degrade goals are:

- Adding a feature that is only applicable to one accelerator design
- Adding code that is complex when a simpler implementation is possible
- Adding code that will be challenging for future developers to understand and extend
- Speeding up the mapper for a subset of accelerator designs while slowing it down for
  others

Fast and Accurate Accelerator Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can think of this goal as having these components.

AccelForge is designed for *early*-design, so it trades off some fidelity for speed by
using analytical performance models. It is important to judge what kind of fidelity is
appropriate for modeling.

For example, AccelForge does not do cycle-by-cycle simulation because we deem it
inappropriate for early design. However, AccelForge comes with a mapper because mappings
have been shown to affect latency and energy by wide margins (see Timeloop, LoopForest,
Fast & Fusiest, and Turbo-Charged Mapper).

Please make it clear how your contribution affects the fidelity of the model (e.g., you
are enabling analysis of bank conflicts), the impact of that fidelity on accelerator
design, and the impact on accelerator evaluation speed (including running the mapping).

Excellent User and Developer Experience
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We want AccelForge to be a tool that others use *and* extend. That makes it crucial for
us to consider the impact of the contributions not just in architecture design terms but
also in terms of how it is implemented.

Before we even discuss code style, there is an even more important element of an
excellent user and developer experience: conceptual integrity. In Fred Brooks's *The
Mythical Man Month*, he stressed the importance of having a cohesive set of ideas drive
the design of a software system. In practical terms, **please avoid adding uncoordinated
features**.

For example, we have taken great care in making sure that a wide variety of
architectures can be expressed in the language that AccelForge has. If you are adding a
feature to that language, consider whether you are enabling modeling a class of
accelerators or the idiosyncrasies of one accelerator. The latter will make it harder
for you to justify your PR acceptance.

That said, AccelForge is a research-oriented project. We welcome discussions for new
features even when the idea is half-baked as long as you are open to taking the time and
effort to care for that idea and its implementation to make it excellent. Think of it
this way, our support for a PR takes time and effort for us to provide. An implicit
social contract (which we make explicit here) is that we expect you to appreciate our
time and effort with yours.

AI Use
------

While we welcome the use of AI tools to assist developers, we expect that all
contributions include a thoughtful human confirming that the contribution respects this
document. We also expect humans to write PR request text and all communications with us.
Please respect our time, and we will respect yours. If it appears that little human
thought has gone into a contribution, then little human thought will go into reviewing
it.

Formatting And Style
--------------------

- **Formatting**: All Python code should be formatted using `Black
  <https://black.readthedocs.io/>`_ with a line length of 88 characters (Black's
  default). For convenience, you can run `make format`.
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
