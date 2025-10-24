---
marp: true
---
<style>section{font-size:18px;}</style>

# FFM vs. SET (Sim. Anneal) and TileFlow (Genetic Algorithms)

<!-- Technical constraints:
- Holding a reasonably-large mapspace will OOM the machine. Holding only Pareto-optimal mappings is OK.
- Our pmapper must be pmapping-first. Implementing another one would take a very long time, and any other would be much slower $\rightarrow$ unfair anyway. -->

How the Fast and Fusiest paper compared:
- **Pmapping exploration:** Exhaustively explored pmapping space and returned Pareto-optimal pmappings. Recorded the porportion of pmappings that were Pareto-optimal.
- **Run baselines:** In place of pmapping model evaluations, access the cached pmappings
  - Generously assume that if $1/N$ of the evaluated pmappings** are Pareto-optimal, then the baseline can find a Pareto-optimal mapping in $N$ evaluations. This is generous; requires some magic function that can tell if a pmapping is Pareto-optimal.
  - Before, $N$ used to be very large, but now $N$ is much smaller because the pmapper avoids evaluating suboptimal mappings.


Feasible runtime for evaluating baselines depends on $N$ being very large; large $N$ means that we can charge the baselines for many evaluations, while only reading the cache once.

But the new pmapper changes things...

<!-- **Note: Evaluated pmappings is the number of pmappings evaluated by the pmapper, not the full size of the mapspace. Pareto-optimal mappings are about $1/1e6$ of the total pmappings, but between $1/5$ and $1/100$ of the evaluated pmappings (depending on constraints). -->

---

# FFM vs. SET (Sim. Anneal) and TileFlow (Genetic Algorithms)
$1/N$ of the evaluated pmappings are Pareto-optimal.

Nominally, $N$ would equal the proportion of Pareto-optimal mappings in the full mapspace size.

However, our new pmapper explores many fewer pmappings. This yields a higher density of Pareto-optimal pmappings (larger $N$).

Larger $N$ $\rightarrow$ baselines became intractible to run. Solution was simple:
- Removed some mapspace constraints (assertion that spatial arrays are fully utilized)
- Larger mapspace $\rightarrow$ Pareto-optimal mappings became more rare $\rightarrow$ larger $N$

---

# Simulated Annealing Comparison

Challenge: In a realistic use case, the new pmapper be $>1000\times$ faster for FFM than it would be for SET/TileFlow.

Key difference:
- FFM is *pmapping-first*: Generates all pmappings, then puts them together to create full mappings.
- SET/TileFlow are *fused-part-of-the-mapping* first: Generates backing storage nodes and fused loops, then fills out the lower-level loops and storage nodes.

*Fused-part-of-the-mapping* first means that any pmapping search will only consider pmappings with a given set of fused loops and backing storage nodes. This blocks key optimizations!


---

# Example

Option A underutilizes the spatial array, while option B fully utilizes it. Both will lead to the same choices for future fused loops, so the pmapper can prune the suboptimal option A.

```
for n0 in ? # <- Fused loop
  for n1 in [0..2): # Shape = 128

    # Option A: Underutilized spatial array
    S-for n2 in [0..1):

    # Option B: Fully-utilized spatial array
    S-for n2 in [0..128):
```

We can apply this pruning before enumerating $n0$ choices because we're pmapping-first. However, SET/TileFlow would generate $n0$ loop choices first, and would have to make this check for every $n0$ choice.

This results in significant speedup:
- $\sim200\times$ speedup: Prune pmappings before enumerating fused loops. Each pruned pmapping will affect $\sim200$ compatibilities
- $\sim80\times$ speedup: If pmappings differ in nothing except the permutation of fused loops, we copy the same pmapping & use different compatibilites

<!-- ---

# Pareto-Optimal Pmappings

The pmapper outputs only Pareto-optimal mappings. This benefits FFM linearly, but SET/TileFlow exponentially.

FFM: Reduce the number of pmappings by $N\times$ has an $N\times$ affect on the pmapping generation stage, but no effect on the joining stage because the joining stage will immediately prune suboptimal pmappings

SET/TileFlow:
- $N\times$ reduction in the pmappings for each Einsum
- $N^{\#Einsums}\times$ reduction in the overall mapspace size
- Challenging trade-offs for the pmapper; which choice does it make?
  - Find a better pmapping for the current fusion choice?
  - Find a better fusion choice?
- The previous is extra challenging because a new fusion choice may appear worse if the found pmapping is suboptimal. -->


---

# How to Compare
- Compare FFM vs Sim. Anneal / Genetic Algorithms, ignore exploration order
  - Generously use best-case pmapper speed for all baselines
  - Generously assume that if $1/N$ of the evaluated pmappings are Pareto-optimal, then the baseline can find a Pareto-optimal mapping in $N$ evaluations.

- Compare FFM + Pmapping-First vs. Sim. Anneal / Genetic Algorithms Fused-Part-of-the-Mapping-First
  - Generously assume that if $1/N$ of the evaluated pmappings are Pareto-optimal, then the baseline can find a Pareto-optimal mapping in $N$ evaluations.
