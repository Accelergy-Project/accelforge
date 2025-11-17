---
marp: true
---
<style>section{font-size:18px;}</style>

# One More Pmapper Optimization

Even with storage-order first pmapping generation, we have a lot of resulting pmappings because there's a lot of ways to choose tile shapes for the loops.

For TPU GPT2 (batch=64, seq=8192) unfused: $4\times10^{14}$ pmappings

Prior works, including Fast \& Fusiest, have relied on user-defined constraints to shrink the pmapping space.
<!-- - In Fast \& Fusiest, we constrained pmappings to fully utilize hardware resources.
- Was this optimal? -->

- Show a simple exmple that shows human-written constraints are difficult to write and/or suboptimal and/or unintuitive
- Find some human-written constraints that were suboptimal
- COSA, ZigZag use min utilization as heuristics, and the ZigZag authors have a short two-pager showing that's suboptimal

**Can we be the first work to explore the pmapping space without any constraints?**

Idea: Pareto-optimal pmappings are rare; in these experiments, only $\approx 100,000$ pmappings (one in a million).

---

# Setting up the problem
Our dataflow-generation code is going to give us a *pmapping template* to fill out. To simplify the problem, we'll only include input storage nodes and loops for rank $N$.

```
[Store input in DRAM]
for n2 in [0...N2):
  [Store input in GLB]
  for n1 in [0...N1):
    [Store input in LLB]
    for n0 in [0...N0):
```

Our job is to select `N0`, `N1`, `N2` to fill out the template. We'd like to generate all Pareto-optimal combinations.

Observations:
- There are not many choices for any one $N0$. For a rank bound of 4096, there's only 14 prime factors to pick from.
- The tile shape for $N(i-1)$ constrains $Ni$ because the tile shape for $Ni$ must be a multiple of the tile shape for $N(i-1)$.

---

This looks like the problem $FFM$ targets.
- We'd like to find all Pareto-optimal combinations of $Ni$ choices
- The choices at a given step constrain future choices.

We can follow the procedure:
- Enumerate choices for $n0$: $n0\in{2,4,8,16 \ldots}$
- Prune suboptimal choices: $n0\in{2,\cancel{4}, 8, \cancel{16}, \ldots}$
- Enumerate choices for $n1$: $(n0,n1)\in{(2,2), (8,2), (2,4), (8,4), \ldots}$
- ...

A choice is only known to be suboptimal if a different choice is better in every way. Need criteria to define "better".

---

# Compatibility Within a Pmapping

For a given pmapping template with $n$ loops, our pmapping space is defined by a set of choices $t_1, t_1, \ldots, t_{n}$.

```
Valid:
for n1 tile shape 128
  [Store input in LLB]
  for n0 tile shape 64  # Inner loop subtiles the outer loop.

Invalid:
for n1 tile shape 128
  [Store input in LLB]
  for n0 tile shape 256  # Inner tile shape can't be larger than the outer tile shape.
```

**Compatibility**: Say that the loops $i$ and $j$ with tile shapes for $t_i$ and $t_j$ apply to the same rank. Assume $t_i$ applies to the more-inner loop. Loop $i$ will subtile the tiles from loop $j$, so $t_i$ is compatible with $t_j$ iff $t_j$ is a multiple of $t_i$*.

If we're constructing pmappings starting with the innermost loop, we can define *compatibility* as the outermost tile shape for each rank.

---

<!-- # Example

We'd like to make choices for tile shapes $a,b,c,d\in{1,2}$ that minimize two objective functions:

$f(a,b,c)=a+ab-bc$

$g(a,b,c)=c*d$

We start by enumerating choices for $a$, getting us pmappings $(1,?,?,?),(2,?,?,?)$. To compare these pmappings, we can see that reducing $a$ will help us minimize $f$, but it has no effect on $g$. So we should take only pmapping $(1,?,?,?)$.

Next, we enumerate choices for $b$ with our Pareto-optimal choicefor $a$, getting us $(1,1,?,?),(1,2,?,?)$. Looking at $f$, we can see that we still want to minimize $a$, but the effect of $b$ depends on the $ab-bc$ term, which depends on $c$. We therefore can't compare pmappings with different $b$.

Next, we enumerate choices for $c$, getting us $(1,1,1,?),(1,2,1,?),(1,1,2,?),(1,2,2,?)$. Now that we know all the values in $f$, we can evaluate $f$ directly as $a+ab-bc$. We can't evaluate $g$ directly, but we know that lower $c$ will lead to lower $g$, so we minimize two criteria: $f(a,b,c)$ and $c$. We'll get two Pareto-optimal pmappings $(1,1,1,?),(1,2,2,?)$

Finally, we enumerate choices for $d$, getting us pmappings $(1,1,1,1),(1,2,2,1),(1,1,1,2),(1,2,2,2)$. We can fully calculate both objective functions, so we can pick any combinations with Pareto-optimal $f$ and $g$.

--- -->

# Making Criteria

Describe objectives and resource limits as functions $f_i$ that depend on the tile shapes used in our pmapping. We'd like to turn them into criteria, but we can't calculate all objectives directly because they depend on tile shapes that we haven't enumerated yet.

**Challenge**: Objectives may depend on tile shapes that we haven't enumerated yet.

**Solution**: If we can't calculate an objective, break it down into sub-objectives that we can evaluate.

*Example*: $Energy = E_{DRAM} + E_{GLB} + E_{LLB} + E_{MAC}$

We first check if we can calculate energy directly. If energy is a function only of the tile shapes we know, we're done. Otherwise, calculate the energy of each sub-component directly and use as multiple sub-objectives.

If a sub-objective can't be calculated, we break it down further: $E_{DRAM} = E_{DRAM,Input} + E_{DRAM,Output} + E_{DRAM,Weight}$

We're able to break down multiple types of expressions to realize many different objectives:
- **Sum**: Total energy, total memory usage
- **Product**: Size of a particular tensor tile, spatial fanout
- **Max**: Latency


---

# Additional notes on criteria making

The previous slide was brief, but it's important to note that using the simplest criteria possible is essential for effective pruning. Therefore, I put together a solver that performs simplification of complex algebreic expressions. The solver uses all of the following rules:

- Ignore terms that don't depend on the tile shapes we've chosen
- If a term is monotonic with respect to a non-enumerated tile shape, we can ignore the non-enumerated tile shape because it won't effect betterness. We also do this for enumerated tile shapes if we can't compare them.
- If a term is monotonic with respect to an enumerated tile shape, we can take the derivative of the term with respect to that tile shape. If increasing a tile shape always leads to a worse value for the term, we should minimize the tile shape. If decreasing a tile shape always leads to a worse value for the term, we should maximize the tile shape.
- Tile shapes are always $\geq 1$.
- In a multiply, if any terms are negative, we need to invert our betterness. If any terms have an undetermined sign, we can't compare pmappings with different values for the other terms, since we don't know if our betterness is inverted.
- If there's disagreements about a term between different objectives (*e.g.,* one term says minimize tile shape $T$ and another term says maximize tile shape $T$), we can't compare pmappings with different values for $T$.
- After splitting a term into multiple sub-terms, we recombine sub-terms that depend on the same non-enumerated tile shapes. This minimizes the number of sub-terms that need to be included in the Pareto criteria.
- Several of these rules require that we look at the terms from multiple objectives. We will look at the terms, apply these simplifying changes, then repeat until we can't simplify any more.

---

# Results

The pmapper can generate $\approx 7\times10^{10}$ pmappings per second for TPU, $>7\times10^7$ faster than Timeloop.

Now we're severely bottlenecked by the fusion stage because:
- Unconstrained searches lead to many unique compatibilities
- For each one, we partition a large dataframe and create metadata to track compatibiliity
- Overlapping lifetimes $\rightarrow$ we need to track usage of any memories that may be shared
- This overwhelms the pmapping generation time

I've got more ideas to fix the fusion stage bottleneck; but

<!-- # Defining Objective Metrics and Resource Usage

Treat objective metrics and resource usage of the pmapping as functions of the tile shapes $f_i(t_1, \ldots, t_n)$. Without loss of generality, assume we'd like to minimize all of them. We can use this for resource usage by eliminating all pmappings with usage over some threshold.

Assume, at a given step, we've enumerated the choices for $t_1, \ldots, t_i$. We've not yet enumerated choices for $t_{i+1}, \ldots, t_n$, so these are treated as unknowns. Let's look at one of the objective functions $g=f_i$. We'd like create comparable criteria for $g$.

If $g$ is a function of $t_1, \ldots, t_i$, then we can use $g$ directly as a metric by plugging in the values we've enumerated for each pmapping.

If $g$ is a  -->





<!-- # Defining Objective Metrics and Resource Usage

Treat objective metrics and resource usage of the pmapping as functions of the tile shapes $f_i(t_1, \ldots, t_n)$. Without loss of generality, assume we'd like to minimize all of them. We can use this for resource usage by eliminating all pmappings with usage over some threshold.

Assume, at a given step, we've enumerated the choices for $t_1, \ldots, t_i$. We've not yet enumerated choices for $t_{i+1}, \ldots, t_n$, so these are treated as unknowns. Let's look at one of the objective functions $g=f_i$.

We'd like to compare one set of tile shape choices $t_1=u_1, \ldots, t_i=u_i$ to another set of tile shape choices $t_1=v_1, \ldots, t_i=v_i$. We can say that $t_1=u_1, \ldots, t_i=u_i$ is better than $t_1=v_1, \ldots, t_i=v_i$ if:

$$
g(u_1, \ldots, u_i, t_{i+1}, \ldots, t_n) < g(v_1, \ldots, v_i, t_{i+1}, \ldots, t_n)
$$

We can use an analytic solver for this function and categorize it into one of three cases:
- **Evaluatable**: $g(t_1, \ldots, t_i, t_{i+1}, \ldots, t_n)$ depends only on $t_1, \ldots, t_i$, the values we've already enumerated. We can use $g$ directly as a metric by plugging in the values we've enumerated for each pmapping (set of choices for $t_1, \ldots, t_i$).
- **Irrelevant**: $g(t_1, \ldots, t_i, t_{i+1}, \ldots, t_n)$ depends only on $t_{i+1}, \ldots, t_n$, the values we'll choose later. We can ignore it when comparing pmappings.
- **Comparable**: $g(t_1, \ldots, t_i, t_{i+1}, \ldots, t_n)$ depends both on $t_1, \ldots, t_i$ and $t_{i+1}, \ldots, t_n$, but the result of the inequality does not depend on $t_{i+1}, \ldots, t_n$. We can use $g$ directly as a metric by plugging in the values we've enumerated for each pmapping and using dummy values for the unknown choices, which won't affect the inequality.
- **Uncomparable**: The inequality $g(u_1, \ldots, u_i, t_{i+1}, \ldots, t_n) < g(v_1, \ldots, v_i, t_{i+1}, \ldots, t_n)$ depends on $t_1, \ldots, t_i$ and $t_{i+1}, \ldots, t_n$. We'll need to handle these cases specially.



---

# Uncomparable Criteria

If we have an uncomparable criterion $g(t_1, \ldots, t_i, t_{i+1}, \ldots, t_n)$, then the betterness of one pmapping over another depends on some combination of enumerated and non-enumerated tile shapes.

We'll figure out how each of the tile shapes we've picked so far affects the criterion, then use those tile shapes to build criteria that are comparable.

For each of the tile shapes we've enumerated so far, we can check $\frac{dg}{dt_i}$ and create several cases:
- $\frac{dg}{dt_i} > 0$: Larger $t_i$ is worse (*e.g.,* makes buffer usage larger). We want to minimize $t_i$.
- $\frac{dg}{dt_i} < 0$: Larger $t_i$ is better (*e.g.,* improves reuse). We want to maximize $t_i$.
- $\frac{dg}{dt_i} = 0$: $t_i$ doesn't affect this criterion, so we can ignore it for this case.
- $\frac{dg}{dt_i}$ depends on $t_{j\neq i}$: There are complex interactions that we can't reason about, so we can only compare pmappings with the same values for $t_i$.

--- -->