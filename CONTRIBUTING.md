The following contribution guideline is work-in-progress.

# The Goals of AccelForge
In our website, we define AccelForge to be "a framework to model and design tensor algebra and deep neural network accelerators."

We take this identity seriously. The primary goal of AccelForge is to enable people (with or without AI assistance) to model and design accelerators.
When contributing to AccelForge, please consider whether your contributions furthers this goal.

You can think of this goal as having these components.

## Fast and Accurate Accelerator Modeling

AccelForge is designed for *early*-design, so it trades off some fidelity for speed by using analytical performance models.
It is important to judge what kind of fidelity is appropriate for modeling.

For example, AccelForge does not do cycle-by-cycle simulation because we deem it inappropriate for early design.
However, AccelForge comes with a mapper because mappings have been shown to affect latency and energy by wide margins
(see Timeloop, LoopForest, Fast & Fusiest, and Turbo-Charged Mapper).

Please make it clear how your contribution affects the fidelity of the model (e.g., you are enabling analysis of bank
conflicts), the impact of that fidelity on accelerator design, and the impact on accelerator evaluation speed (including
running the mapping).

## Excellent User and Developer Experience

We want AccelForge to be a tool that others use *and* extend. That makes it crucial for us to consider the impact of the
contributions not just in architecture design terms but also in terms of how it is implemented.

Before we even discuss code style, there is an even more important element of an excellent user and developer experience: conceptual integrity.
In Fred Brooks's *The Mythical Man Month*, he stressed the importance of having a cohesive set of ideas drive the design of a software system.
In practical terms, **please avoid adding uncoordinated features**.

For example, we have taken great care in making sure that a wide variety of architectures can be expressed in the language
that AccelForge has. If you are adding a feature to that language, consider whether you are enabling modeling a whole class of accelerators
or the idiosyncrasies of one accelerator. The latter will make it harder for you to justify your PR acceptance.

That said, AccelForge is a research-oriented project. We welcome discussions for new features even when the idea is half-baked as long as
you are open to taking the time and effort to care for that idea and its implementation to make it excellent. Think of it this way, our
support for a PR takes time and effort for us to provide. An implicit social contract (which I am making explicit here) is that we expect you
to appreciate our time and effort with yours.


# Reviewing Your PR
You should always review your own pull request first, before marking it as ready for review by others.

Make sure that you:
- Confirm that the changes meet the user experience and goals outlined above.
- Ensure that all your code and documentation achieves what the PR says they will achieve.
- Check documentation for clarity (including grammar and spelling).
- Please adhere to the style of existing code. We do not currently have a formal style guide, but there is enough existing code for you to
peruse. Style guides exist to make code and comments clear; if you strive for clarity, you should be fine.
