# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ColBench (Meta Sweet-RL Collaborative Agent Bench, Backend-Programming) multi-turn RL.

Phase-1: train a solver against a FROZEN user simulator. The solver extracts hidden
requirements from the simulator via clarification dialogue, then submits code graded by
functional equivalence against a ground-truth function. Ported from ``sweet_rl`` (env +
reward semantics) into our own verl stack; mirrors the ``codecontest`` package layout and
reuses its sandboxed exec sidecar for grading.
"""
