"""Multi-turn RL on ColBench Backend-Programming.

ColBench is Meta's Sweet-RL Collaborative Agent Bench.

Phase-1: train a solver against a FROZEN user simulator. The solver extracts
hidden requirements from the simulator via clarification dialogue, then submits
code graded by functional equivalence against a ground-truth function. Ported
from ``sweet_rl`` (env + reward semantics) into our own verl stack; mirrors the
``codecontest`` package layout and reuses its sandboxed exec sidecar for
grading.
"""
