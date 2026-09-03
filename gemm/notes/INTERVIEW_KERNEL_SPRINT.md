# Kernel interview sprint: learn by typing, not by reading

The minimum interview target is CUDA C++, not handwritten PTX. Use one
25-minute loop per session:

1. **Predict (3 min):** say the tensor shape, one-thread responsibility, and
   memory path out loud.
2. **Type (12 min):** fill one missing 10--25 line region without copying the
   whole solution.
3. **Run (5 min):** compile, run one odd-size correctness case, then one normal
   benchmark case.
4. **Explain (5 min):** close the source and explain why every barrier and
   reduction step is needed.

Stop after one loop. Starting tomorrow is more valuable than finishing a long
reading session today.

## The six drills

| Drill | What must be hand-written | What may stay conceptual first |
|---|---|---|
| 1. Sum reduce | grid-stride load, shared-memory tree, multi-block finish | vector load tuning |
| 2. Generic reduce | identity + associative op; max/min; `{value,index}` argmax | cooperative groups |
| 3. Row softmax | row max reduce, `exp(x-max)`, row sum reduce, normalize | online/streaming softmax |
| 4. RMSNorm | sum of squares, `rsqrt`, scale; one row/block | fused residual variants |
| 5. GEMM | naive/coalesced, SMEM tile, register micro-tile, boundary checks | `cp.async`, Tensor Core PTX |
| 6. Attention | causal bounds and online-softmax recurrence | production FlashAttention scheduling |

## Closed-book checks

You are ready to move on when you can answer these without opening the source:

- Which global addresses do lanes 0--31 access in one instruction?
- Which values are reused from shared memory, and by how many threads?
- What is the operator identity? Is the operator associative?
- Why is each `__syncthreads()` legal and necessary?
- What happens for an odd input size or a row shorter than the block?
- Is the timed region allocation-free and synchronization-correct?

The advanced Tensor Core track is for reading and profiling. In an interview,
being able to explain `GMEM -> SMEM -> registers -> MMA -> epilogue`, pipeline
stages, bank-conflict-free layouts, and the occupancy/register tradeoff is a
strong answer even when the exact `mma.sync` register mapping is not memorized.
