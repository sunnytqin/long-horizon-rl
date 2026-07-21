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
"""CPU tests for colbench.selfplay.spec_templates (parsing + prompt building). No server."""

from colbench.selfplay import spec_templates as st


def test_parse_clean_json():
    raw = ('{"persona": {"who": "a bookseller", "domain": "publishing", '
           '"python_skill": "non-coder", "communication_style": "casual"}, '
           '"scenario": "She tracks a press.", "requirements": "Subtract the 126 laid off."}')
    spec = st.parse_spec(raw)
    assert spec["ok"] is True
    assert spec["scenario"] == "She tracks a press."
    assert "126" in spec["requirements"]
    assert spec["persona"]["python_skill"] == "non-coder"


def test_parse_json_wrapped_in_prose_and_fence():
    raw = ("Sure! Here is the spec:\n```json\n"
           '{"persona": "an analyst", "scenario": "s", "requirements": "do X then Y"}\n'
           "```\nHope that helps.")
    spec = st.parse_spec(raw)
    assert spec["ok"] is True
    assert spec["requirements"] == "do X then Y"
    assert spec["persona"] == "an analyst"


def test_parse_malformed_falls_back_to_raw_requirements():
    raw = "I could not follow the format but the function should add two numbers."
    spec = st.parse_spec(raw)
    assert spec["ok"] is False
    assert spec["requirements"] == raw  # nothing lost


def test_persona_object_renders_to_text():
    persona = {"who": "a teacher", "domain": "education", "python_skill": "hobbyist", "communication_style": "chatty"}
    text = st._persona_to_text(persona)
    assert "a teacher" in text and "hobbyist" in text and "education" in text


def test_author_messages_contain_both_public_and_gt():
    msgs = st.build_author_messages("PUBLIC ASK", "def f(): return 126")
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "PUBLIC ASK" in user and "def f(): return 126" in user
    # The instruction must forbid pasting the reference code.
    assert "Describe intent" in user or "never code" in user or "not code" in user.lower()


def test_full_spec_solver_messages_include_signature_and_requirements():
    spec = {"persona": "an analyst", "scenario": "quarter close", "requirements": "return the exact numbers"}
    msgs = st.build_full_spec_solver_messages(
        "Create def calculate_stats(a, b). The signature is def calculate_stats(a, b)", spec)
    user = msgs[1]["content"]
    assert "calculate_stats" in user            # signature carried from the public request
    assert "return the exact numbers" in user   # requirements carried
    assert "an analyst" in user                 # persona flavor present


# ── plot variant ──────────────────────────────────────────────────────────────
def test_parse_plot_spec_ok_requires_requirements_and_plot():
    raw = ('{"persona": "a home user", "scenario": "netbook boot issue", '
           '"requirements": "if year>=1970 subtract 126 employees", '
           '"plot": "The user forgets the 126 layoff rule at first and only mentions it after a draft."}')
    spec = st.parse_plot_spec(raw)
    assert spec["ok"] is True
    assert "126" in spec["requirements"]
    assert "forgets" in spec["plot"]


def test_parse_plot_spec_missing_plot_not_ok():
    raw = '{"persona": "x", "scenario": "y", "requirements": "do the thing"}'
    spec = st.parse_plot_spec(raw)
    assert spec["ok"] is False and spec["plot"] == ""


def test_parse_plot_malformed_falls_back_to_requirements():
    raw = "not json but describes the behavior"
    spec = st.parse_plot_spec(raw)
    assert spec["ok"] is False and spec["requirements"] == raw


def test_plot_author_messages_ask_for_direction_not_script():
    msgs = st.build_plot_author_messages("PUBLIC ASK", "def f(): return 126")
    user = msgs[1]["content"]
    assert "PUBLIC ASK" in user and "def f(): return 126" in user
    assert '"requirements"' in user and '"plot"' in user
    # the plot must be a DIRECTION, explicitly not a turn-by-turn script
    assert "DIRECTION" in user and "turn-by-turn" in user.lower()
    # reveal mechanisms are CONDITIONAL on the assistant (never assume it asks); user can't test
    assert "if the assistant asks" in user.lower()
    assert "never assume" in user.lower()
    assert "never runs, tests, or executes" in user.lower()
    # requirements describe the user's needs in the third person (consistent with the plot)
    assert "third person" in user.lower() and "the user needs" in user.lower()
    # the plot is written as an instruction to the user-simulator ("the user would...")
    assert "instruction to the user-simulator" in user.lower()


def test_plot_solver_still_uses_requirements():
    # the diagnostic solver reads requirements (unchanged full-spec solver), not the plot
    spec = {"persona": "a home user", "scenario": "boot issue",
            "requirements": "return the exact numbers", "plot": "forgets an edge case"}
    msgs = st.build_full_spec_solver_messages("Create def diagnose(a, b)", spec)
    user = msgs[1]["content"]
    assert "diagnose" in user and "return the exact numbers" in user
    assert "forgets an edge case" not in user  # plot is not fed to the diagnostic solver


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all spec_templates tests passed")
