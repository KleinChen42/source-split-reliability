# Innovation and operational-value extension

Specified on 6 November 2026 before inspecting the new experiment outcomes.
The existing H/H2 cohorts have been inspected in prior work; these are
exploratory extensions on the same real prediction banks.

The central question is whether an information-preserving change to an
encoder's representation can change a source-split comparison. A useful
reliability implementation should also support a declared answer-release
capacity without another answer-generation pass.

1. **Mechanism.** ActivityNet-QA, the existing 30 paired development seeds,
   both development flows, both holdouts, and all four frozen model/prompt
   candidates. Transform the visual block by identity, four identical copies,
   or a fixed signed orthogonal Walsh-Hadamard transform. Compare coordinate
   standardization, coordinate standardization with dimension-balanced
   blocks, and centering with per-block empirical energy normalization.
   Use fixed alpha=1 within each model/prompt and a pooled alpha grid
   {0.01, 0.1, 1, 10, 100, 1000} with native confidence eligible.
   Repeated columns are fitted using their exact compressed ridge equivalent
   (sqrt(4) multiplication); an explicit-column calculation checks this
   equivalence on real fitting records. The orthogonal transform is computed
   on the actual 2,048-dimensional visual vectors. All normalization fits F.
2. **Operational selection.** Use F-fitted native, ridge and boosted-tree
   candidates to select a policy on V at q in {.1,.3,.5,.7,.9}. Compare
   capacity-matched selection with the same menu selected at q=.7. Menus
   distinguish native, text/native trees, visual features, question features,
   and combined menus. No holdout labels enter policy selection.
3. **Finite queues.** Replay selected q=.7 policies in common source-ordered
   arrival streams, grouping 10, 50, 100 or 1,000 sources per queue (100,
   500, 1,000 or up to 10,000 records), with 20 fixed random orderings.
   The smaller final batch uses its actual size. These are benchmark workload
   replays, not observations from a deployed service.
4. **Actual compute cost.** On the two authorized RTX 5090 GPUs, time frozen
   answer generation, ResNet features and capture of the pre-answer Qwen
   state on a fixed, outcome-independent set of 64 real video questions.
   Separate model loading and warm-up; synchronize GPU timing boundaries.
   Describe whether features are shared within a source and whether Qwen
   features require a separate pass for the other generator.

Ridge solves use double precision on the GPUs. Compare one full-feature
fit with a CPU reference. Retain the existing label-free ranking priorities.
Report accepted loss, split contrast, and encoding sensitivity. Source draws
are shared across compared policies and seeds and rebuild each queue.
Any new intervals use a separately declared finite comparison family and
are labelled as approximate conditional bootstrap intervals. Capacity and
finite-queue curves are descriptive unless included explicitly in that family.

The extension interval family uses 15 pooled panels at q=.7: all nine
normalization/encoding combinations with expanded ridge selection, native,
trees, the two question menus, and the two combined menus. It includes each
panel and every within-cohort pair for J_R, J_S and delta on H and H2:
2 * 3 * (15 + choose(15, 2)) = 720 endpoints, with 1,999 source draws.
Fixed-candidate interventions and additional capacities are descriptive.

The contribution is an invariance criterion, its controlled evaluation and
its operational consequences. Centering, block normalization and ridge
regression themselves are established techniques and are not claimed as
new algorithms.
