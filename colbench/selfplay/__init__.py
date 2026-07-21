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
"""ColBench self-play spec setting (Phase 0: offline spec authoring + diagnostic).

A CLEAN, SELF-CONTAINED alternative to the current GT-code-conditioned simulator. Instead of
handing the frozen user simulator the hidden ground-truth *code* (which leaks), we author a
natural-language **spec** (persona + scenario + complete requirements) offline and will later
condition the simulator on that prose. This subpackage builds ONLY the Phase-0 pieces:

  * ``generate_specs``  -- author one spec per task from (public problem + hidden GT code),
    with a pluggable backend: ``strong`` (external teacher) or ``selfplay`` (the trained
    model's frozen base checkpoint; no external model in the loop).
  * ``diagnose_specs``  -- the full-spec solve-rate diagnostic: hand a solver the ENTIRE spec
    (single turn, no dialogue), grade against the UNCHANGED GT ``test_cases``, and report the
    solve rate. Run it on strong-gen and self-gen specs to see whether self-authored specs are
    good enough before committing to Phase-1 training plumbing.

Grading and reward stay exactly as in the main package (``colbench.reward`` -> the codecontest
exec sidecar); the GT code is the reward, the spec only governs conditioning. Nothing here
touches the existing GT-code-conditioned training path.
"""
