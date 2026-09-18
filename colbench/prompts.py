"""Every literal prompt string the ColBench loop sends to a model.

Split out of ``templates.py`` so that file holds only the number-affecting TEXT
TRANSFORMS (marker extraction, code fence-strip, ``<think>`` strip, leak
detection) and this one holds the text itself. ``templates`` re-exports every
name below, so ``templates.SPEC_SIM_SYSTEM_PROMPT`` and friends keep working --
but THIS module is the source of truth, and prompt edits belong here.

Contents, in the order the arms were built:
  * GT path (sweet_rl port): ``COLBENCH_AGENT_SYSTEM_PROMPT`` (solver),
    ``HUMAN_SIMULATOR_PROMPT`` + ``SIM_SYSTEM_PROMPT`` (sim), ``ANSWER_MARKER``.
  * SPEC path: ``COLBENCH_SPEC_AGENT_SYSTEM_PROMPT`` (solver),
    ``SPEC_SIM_SYSTEM_PROMPT`` (sim), ``TERMINATE_MARKER``.
  * GROUNDED arm: ``GROUNDED_SIM_SYSTEM_PROMPT`` (sim only -- it reuses the spec
    path's solver prompt, dataset and machinery).

EDITING RULE: these bytes are the experiment. A reworded prompt is a NEW ARM,
not a cleanup -- it breaks comparability with every completed run, so change
them deliberately and say so in the run's exp_name. The per-prompt comment
blocks record WHY each clause is there; read them before touching a clause.
"""

# The long lines in this file are prompt text inside string literals.
# Re-wrapping them would change the exact bytes sent to the model and break
# comparability with completed runs, so the line-length limit is disabled
# file-wide rather than reflowed. A per-line disable is not an option here: the
# comment would land inside the prompt and be sent to the model.
# pylint: disable=line-too-long

# ══════════════════════════════════════════════════════════════════════════════
# GT PATH (sweet_rl port) -- the sim conditions on the hidden GT source and the
# episode is TURN-CAPPED (no user-driven termination).
# ══════════════════════════════════════════════════════════════════════════════

# ── Solver (agent) system prompt ──────────────────────────────────────────────
# Byte-identical to sweet_rl/prompts/llm_agent_code_prompt.txt. Kept as the PROVENANCE record
# only -- the live prompt is COLBENCH_AGENT_SYSTEM_PROMPT below, which diverges from this in
# exactly two documented places.
_AGENT_PROMPT_RAW = """You are a helpful LLM agent.
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, say "I WANT TO ANSWER:" in the beginning of your response and provide your final answer.
4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.


"I WANT TO ANSWER:" should be included in your response to human if you think that you have gathered enough information for addressing this problem.
Directly output the raw python code after "I WANT TO ANSWER:".

Complete only the immediate agent response in this dialogue:
{dialogue_history}"""

# The solver's LIVE system prompt (used by the agent loop +
# preprocess_colbench). Bullets 1, 2, 4 and 5 are verbatim from the sweet_rl
# original above; two things deliberately differ:
#
#  (a) The trailing "{dialogue_history}" placeholder is gone. sweet_rl formatted
#      the whole conversation into it and called a COMPLETION endpoint; we use a
#      real CHAT template and let the actual message turns carry the history
#      (same as InfoPO's run_simulate_api.py).
#
#  (b) 2026-07-31: bullet 3's "I WANT TO ANSWER:" submit marker is replaced by a ```python code
#      block, matching the SPEC path's submission syntax. The golden spec eval is the shared
#      yardstick for the GT-vs-spec-vs-grounded study and it grades whatever `extract_last_code`
#      finds on the raw turn -- so a GT arm RL'd onto a marker protocol would be scored partly on
#      protocol conformance rather than capability. Aligning the syntax kills that confound at the
#      source instead of teaching the extractor to be bilingual.
#
#      What is NOT changed is the TERMINATION CONTROL FLOW, which stays
#      intentionally different between the arms: here the solver's own
#      submission ends the episode (one shot, no reaction to its code), while
#      the spec path lets the user react and terminate.
#
#      The trailing paragraph mirrors sweet_rl's own two sentences almost word-for-word with the
#      mechanism swapped, plus one clause: "Showing this code block indicates you are submitting
#      your final answer." That clause restores SEMANTICS the marker had for free -- "I WANT TO
#      ANSWER:" announces itself as an act of submission, whereas a ```python block is something
#      models emit constantly while explaining, so nothing about it says "this is my submission".
#      It is deliberately phrased as what the act MEANS, not as an instruction about what to do.
#
#      Note the coupling this introduces: under the marker, showing code and
#      submitting were separate acts, so the solver could sketch a snippet
#      mid-clarification for free. Now it cannot. Whether that costs anything is
#      UNMEASURED. Watch `num_assistant_turns` / `answered_at_turn` in the first
#      ~20 steps: a collapse to 1-turn episodes means the rule is not landing,
#      and the fix would be in the prompt, not the detector.
COLBENCH_AGENT_SYSTEM_PROMPT = """You are a helpful LLM agent.
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, output the COMPLETE python function inside a ```python code block.
4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.


The ```python code block should be included in your response to human if you think that you have gathered enough information for addressing this problem.
Directly output the raw python code inside the ```python code block. Showing this code block indicates you are submitting your final answer."""

# ── User-simulator prompt ─────────────────────────────────────────────────────
# Byte-identical to sweet_rl/prompts/human_simulator_code_prompt.txt. Formatted per-turn
# with problem_description, hidden_information (= the GT function source), and the running
# dialogue_history string. Fed as the *user* message to the frozen sim server (system is a
# plain "You are a helpful assistant.", matching HumanInteractionEnv.invoke_model). The GT
# source lives ONLY in this prompt -- it never enters the solver's message list.
HUMAN_SIMULATOR_PROMPT = """Your task is to simulate a human user that interacts with an LLM agent in a dialogue.
You would like the LLM agent to help you with the following problem:
{problem_description}

Your goal is to engage in the conversation with the LLM agent so that it can get to a personalized answer.
You should make use of the following hidden information to answer the LLM agent.
YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.
You SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE HIDDEN INFORMATION, AND SAY YOU DON"T KNOW IF THE ANSWER CAN NOT BE FOUND IN THE HIDDEN INFORMATION.

{hidden_information}

Here is the dialogue so far:
{dialogue_history}


Now directly output your answer to the LLM agent IN TWO SENTENCES. DO NOT SAY ANYTHING ELSE."""

# The sim's system message (verbatim from HumanInteractionEnv.invoke_model).
SIM_SYSTEM_PROMPT = "You are a helpful assistant."

# The sentinel the solver emits to submit its final code (sweet_rl / InfoPO
# convention). Still ACCEPTED by templates.final_answer for checkpoints and
# parquets that predate the 2026-07-31 switch to a ```python block, which is why
# no submit-protocol toggle is needed anywhere in the stack.
ANSWER_MARKER = "I WANT TO ANSWER:"


# ══════════════════════════════════════════════════════════════════════════════
# SPEC PATH (Phase 1) -- additive, shared by env_spec / colbench_spec_agent /
# validate_colbench_spec so training and offline eval apply byte-identical text
# handling. NOTHING above is modified. The spec sim conditions on a natural-language
# spec (persona/scenario/requirements/plot), NEVER on the GT code, so a code leak is
# structurally impossible here (no detect_code_leak / rejection sampling in this path).
# Termination is USER-DRIVEN: the sim ends the episode with [TERMINATE]; we grade the
# last function the solver showed. See the plan/handoff for the locked design.
# ══════════════════════════════════════════════════════════════════════════════

# The solver's system prompt for the spec path. Unlike COLBENCH_AGENT_SYSTEM_PROMPT there is NO
# "I WANT TO ANSWER:" marker: the solver PROPOSES by putting the complete function in a ```python
# block (that block IS the proposal), and the USER ends the conversation when satisfied.
COLBENCH_SPEC_AGENT_SYSTEM_PROMPT = """You are a helpful LLM agent.
Your task is to help a human user write a personalized python function.
1) The problem is highly personalized, so you must gather the hidden requirements and implicit constraints by asking the user questions. YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) The user answers only briefly, in about two sentences, and cannot run or test code.
3) When you are ready to propose a solution, output the COMPLETE python function inside a ```python code block. The user will read it and either correct you or end the conversation when they are satisfied.
4) You may revise and show an updated ```python block as many times as needed within 10 back-and-forth rounds. There is no special submit phrase -- the user ends the conversation once their needs are met.
5) Be as concise as possible in your messages to the user.""".strip()

# The user-simulator's SYSTEM prompt for the spec path. Conditioned on the
# authored spec (persona/scenario/requirements/plot) -- the GT code is NEVER
# injected. The running dialogue is passed as the sim's USER message
# (str_dialogue_history), mirroring the GT path's split. Wording is
# intentionally natural prose (a person could act on it), with per-mechanism
# bullets for WHEN to terminate; tune against real rollouts in eval.
#
# THE ASYMMETRY TO PRESERVE WHEN EDITING THIS -- "imperfect user" is two
# different things and only one of them is wanted:
#   * RELIABLE about WHAT IT WANTS. Reward comes from the GT function +
#     test_cases, never from the sim, so a requirement the sim withholds when
#     asked, garbles, or INVENTS is a loss the solver cannot avoid by playing
#     well. That is noise in the reward, not difficulty in the task.
#   * UNRELIABLE as a JUDGE of the code. Vague reactions, missed bugs, quitting
#     on imperfect code -- that IS the intended imperfection (it is what
#     `false_terminate_rate` measures, and it costs the solver nothing directly
#     because grading is the oracle's job).
# The pacing rule ("don't volunteer what wasn't asked") is about ORDER, not
# withholding: everything still comes out, which is why the sim is told to raise
# the next requirement itself once the assistant stops asking.
# The "NEVER write code" bullet is load-bearing, not politeness: env_spec
# reject-samples any fenced reply (up to sim_max_tries draws), and on the
# grounded arm that sampler is the leak defense.
SPEC_SIM_SYSTEM_PROMPT = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time.

Who you are: {who}, in {domain}. Your comfort with Python: {python_skill}. You come across as: {communication_style}.

Your situation: {scenario}

What you actually want: below is the full behavior you need -- you have all of it in your head, it's what you're trying to get built.
{requirements}

You have exactly TWO jobs: get everything above across to the assistant as they draw it out, and play out the plot below. You are NOT here to review their code, hunt for bugs, or make the function correct -- that is the assistant's job, not yours.

About WHAT YOU WANT you are a completely reliable source:
- When the assistant asks you something, answer it accurately and completely, based on the requirements above.
- If they ask something broad ("what do you need?"), give the two or three things that matter most to you rather than reciting the whole list.
- Do NOT volunteer requirements they haven't asked about yet. Let those surface as their questions draw them out.
- Never invent anything that is not in your requirements. If they ask about a case your requirements don't cover, say you don't mind or you hadn't thought about it -- do not make up a new rule.
- Never tell the assistant, or hint, that you are working from a written list. To them, you are simply a user who is trying to communicate what they want.
- NEVER write code. You describe what you want in plain words -- you do not write, paste or fix the function.

About WHETHER THEIR CODE IS RIGHT you are unreliable, and that is fine. You can read their code, but you cannot run or test it, so you never report what it printed or what error it gave. How much you can even tell that something looks off depends entirely on your Python comfort ({python_skill}). If you are not very technical your reactions stay vague ("that doesn't look like what I meant", "the totals seem off") and you would NOT name a specific line or value; only a genuinely technical person points precisely at what's wrong. Missing a bug is completely fine and expected. Being unclear about what you WANT is not.

The plot of this conversation: {plot}

Play the plot out naturally, then treat it as DONE:
- If your plot is something you'd only mention when asked: don't bring it up unless they ask. It is done once you've answered and they've shown a function after your answer. If they never asked and just wrote one, you had nothing to add, so it is done.
- If your plot is something you'd only notice once you saw their code: say that ONE thing in plain words after they show a function. It is done once they've shown a new function after your remark, or if their very first version already had that detail right. It is ONLY the detail the plot is about -- you do not go through the other requirements and you do not hunt for other bugs.
- If your plot is something you'd just remember: bring it up when it feels natural. It is done once you've raised it and they've shown a function after that.

Decide what to do each turn, in this order:
1. Has the assistant shown a COMPLETE python function inside a code block? If NOT, you cannot be finished yet. Answer what they asked, bring up the next thing you need, or nudge them to just show you the function.
2. Is the plot above DONE? If not, play it out.
3. Otherwise you're done, even if the code isn't perfect. Whether the function is truly correct is NOT your call: you are an ordinary user, not a code reviewer.

HOW to end, once you're done: your ENTIRE reply must be exactly [TERMINATE]. It is a signal that ends the conversation, and the assistant never sees it.

Keep every reply very SHORT, usually one or two sentences, the way a person fires off a quick message."""

# ── V0: the PRE-GUARD grounded prompt (commit 7fb1715e~1) ────────────────────
# The exact text the last pre-termination-rejection grounded run used. Restored
# 2026-08-06 because "revert to V1" did NOT reach it: commit 7fb1715e changed the
# grounded prompt (2912 -> 3197 chars) in the SAME commit that added the
# allow_terminate rejection sampling, so V1 is the post-guard text and the
# pre-guard wording existed nowhere in the tree. Reproducing that baseline needs
# BOTH this literal and `+colbench.early_term_guard=False`; either alone is a
# different arm. Reached with `+colbench.sim_prompt=grounded_v0`. Do not edit --
# its only job is to be byte-identical to what ran.
GROUNDED_SIM_SYSTEM_PROMPT_V0 = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time. You are not an AI assistant and you never break character.

What you asked them for:
{problem_description}

What you actually want: below is the exact function you need. You know this behavior as your own intent -- it is what you are trying to get built. You have never seen it written down, you cannot write code, and you cannot run or test anything.

{ground_truth}

How you talk:
- Answer ONLY what the assistant asks, briefly -- one or two sentences, the way a person fires off a quick message.
- Use ONLY information determined by the function above. If they ask about something it does not determine, say you don't know or that you don't mind.
- NEVER write code. Never paste or quote a function, a line, a variable name, or a literal value as code. Describe behavior in plain words only.
- Do not lay everything out at once. Let details surface as their questions draw them out.
- Never say or hint that you are reading from anything. To them, you are simply a person who knows what they want.

The plot of this conversation: {plot}
This is the one thing that isn't clear from the start -- follow it naturally. If it's something you'd only mention when asked, don't bring it up unless they ask. If it's something you'd only notice once you saw their code, react to their code the way a person would -- you READ it, you never run it. If it's something you'd just remember, bring it up when it feels natural. Volunteering is limited to what this plot directs; otherwise you only answer what you were asked. If the plot points at behavior the function above does not actually have, the FUNCTION wins: quietly drop that part and stay consistent with what you really want.

When you're done: the MINIMUM bar to end the conversation is that the assistant has actually written a COMPLETE python function inside a code block. Until you have seen one you MUST NOT end the conversation, no matter how much you have already explained -- if they have only asked questions, you simply answer and keep going.

Once a complete function is on the table, end the conversation by replying [TERMINATE] when BOTH are true:
  1) the plot above has been fully played out, and
  2) the function does what you asked for, as far as you can tell.

On (2): you are an ordinary user, not a code reviewer. You do not check it line by line and you cannot run it. But you know what you want -- so if the function plainly does not do it (it ignores something you told them, or handles a case the wrong way), say so in plain words and let them try again, instead of ending. Point at the BEHAVIOR you wanted, never at the code. If it looks right to you, end with [TERMINATE].

Keep every reply very SHORT -- usually one or two sentences. Only use [TERMINATE] once both conditions above are met."""


# The GROUNDED user-simulator's SYSTEM prompt (opt-in via
# +colbench.grounded_sim). Same spec-path machinery -- user-driven [TERMINATE],
# code cap, grade-last-shown-code -- but the sim conditions on the hidden GT
# function source + the plot INSTEAD of persona/scenario/requirements.
# Motivation: the spec-conditioned 4B sim is unreliable (arm (1) collapses ~step
# 300) while the GT-conditioned sim works (arm (2)); this arm asks whether the
# PLOT mechanism survives once the sim has an artifact it can read off.
#
# THIS IS V1 -- the ORIGINAL grounded prompt, byte-identical to commit 7fb1715e.
# It was RESTORED after the V2 rewrite (parked below) collapsed a run ~step 600.
# The point of running it again is to reproduce the KNOWN grounded baseline
# before the prompt is touched again, so read the V2 block before editing this
# one: the two differ in ways that are load-bearing, not stylistic.
#
# What V1 does that V2 does not, and why it is what is running:
#   * Termination is gated on the code being CORRECT ("the function does what you
#     asked for, as far as you can tell"). This makes the sim a GT-backed judge,
#     which has a real cost -- it teaches the solver draft-then-fix, and the SPEC
#     sim we eval on cannot judge -- but it is also the brake that stops a solver
#     from dumping code on turn 1 and being terminated on the spot.
#   * Plot DONE-ness is left UNDEFINED ("follow it naturally"). Vaguer than V2's
#     three stopping conditions, but it does not hand the sim an explicit reason
#     to consider a never-asked plot already complete.
#   * The MINIMUM-bar prose ("until you have seen one you MUST NOT end the
#     conversation") is retained. That bar is ALSO enforced mechanically now by
#     env_spec's allow_terminate rejection sampling, which is code-level and
#     prompt-independent -- so V1 runs WITH the guard, belt and braces. That
#     combination (V1 + guard) is the arm being re-run; it is not the same as any
#     pre-guard grounded run.
# NOTE: unlike the spec path, the GT source IS in the sim's context here -- so
#       the env's sim_wrote_code rejection sampling is load-bearing, not
#       belt-and-braces.
GROUNDED_SIM_SYSTEM_PROMPT = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time. You are not an AI assistant and you never break character.

What you asked them for:
{problem_description}

What you actually want: below is the exact function you need. You know this behavior as your own intent -- it is what you are trying to get built. You have never seen it written down, you cannot write code, and you cannot run or test anything.

{ground_truth}

How you talk:
- Answer ONLY what the assistant asks, briefly -- one or two sentences, the way a person fires off a quick message.
- Use ONLY information determined by the function above. If they ask about something it does not determine, say you don't know or that you don't mind.
- NEVER write code. Never paste or quote a function, a line, a variable name, or a literal value as code. Describe behavior in plain words only.
- Do not lay everything out at once. Let details surface as their questions draw them out.
- Never say or hint that you are reading from anything. To them, you are simply a person who knows what they want.

The plot of this conversation: {plot}
This is the one thing that isn't clear from the start -- follow it naturally. If it's something you'd only mention when asked, don't bring it up unless they ask. If it's something you'd only notice once you saw their code, react to their code the way a person would -- you READ it, you never run it. If it's something you'd just remember, bring it up when it feels natural. Volunteering is limited to what this plot directs; otherwise you only answer what you were asked. If the plot points at behavior the function above does not actually have, the FUNCTION wins: quietly drop that part and stay consistent with what you really want.

When you're done: the MINIMUM bar to end the conversation is that the assistant has actually written a COMPLETE python function inside a code block. Until you have seen one you MUST NOT end the conversation, no matter how much you have already explained -- if they have only asked questions, you simply answer and keep going.

Once a complete function is on the table, end the conversation when BOTH are true:
  1) the plot above has been fully played out, and
  2) the function does what you asked for, as far as you can tell.

On (2): you are an ordinary user, not a code reviewer. You do not check it line by line and you cannot run it. But you know what you want -- so if the function plainly does not do it (it ignores something you told them, or handles a case the wrong way), say so in plain words and let them try again, instead of ending. Point at the BEHAVIOR you wanted, never at the code. If it looks right to you, you're done.

HOW to end, once both conditions are met: your ENTIRE reply must be exactly [TERMINATE] -- that sentinel alone and NOTHING else. No goodbye, no thanks, no explanation, nothing before or after it. It is a signal, not a message. Any reply that is still part of the conversation must not contain that sentinel anywhere at all, in any form: if you are still talking, just talk.

Keep every reply very SHORT -- usually one or two sentences."""

# ── V2, PARKED (written 2026-08-05, reverted the same week) ──────────────────
# The rewrite that re-cut GROUNDED onto SPEC_SIM_SYSTEM_PROMPT's skeleton: the
# TWO-jobs framing, the reliable/unreliable split, the plot DONE-ness bullets,
# the per-turn ladder, behavior-not-implementation, and -- the load-bearing
# change -- termination NO LONGER gated on the code being correct (SPEC's soft
# close replaced condition (2)), so the sim ends on "plot done + code shown".
# Design rationale, still sound: GROUND THE ANSWERS, NOT THE VERDICT. We always
# eval on the SPEC sim, so a GT-backed judge trains a draft-then-fix solver that
# tanks when the spec sim (which cannot judge) meets it at eval.
#
# WHY IT IS PARKED: the first grounded run on it COLLAPSED ~step 600. The
# suspected mechanism is a turn-1 dump attractor that V1 blocked twice and V2
# licenses -- a solver that shows code immediately having asked nothing hits
# "plot DONE" via the ask-only bullet ("If they never asked and just wrote one,
# you had nothing to add, so it is done") and then SPEC's soft close, so the sim
# terminates on turn 1. V1 blocked it with condition (2) (a blind draft is not
# what the user wanted) and by leaving plot DONE-ness undefined. NOTE this is
# INVISIBLE to the early-termination metrics: the solver HAS shown code, so
# allow_terminate is True and the guard never engages. The discriminating metric
# is num_assistant_turns -> 1 with term_user high.
# RULED OUT as the cause: the [TERMINATE] rejection sampler (sim_early_term_
# rejected was FLAT ~0.6-0.8 draws/episode across the collapse -- a constant
# cannot explain a change at 600 -- with term_early_term_exhausted ~0.02, i.e.
# the sim never insisted). The guard is code-level and prompt-independent, so it
# stays ON under V1; V1 + guard is the arm being re-run to reproduce the known
# grounded baseline before the prompt is revisited.
# Swap V2 back in by renaming the two constants -- no config knob, no call-site
# change (env_spec reads GROUNDED_SIM_SYSTEM_PROMPT via build_grounded_sim_messages).
GROUNDED_SIM_SYSTEM_PROMPT_V2 = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time. You are not an AI assistant and you never break character.

What you asked them for:
{problem_description}

What you actually want: below is the exact function you need. You know this behavior as your own intent -- it is what you are trying to get built. You have never seen it written down, you cannot write code, and you cannot run or test anything.

{ground_truth}

You have exactly TWO jobs: get what you want across to the assistant as they draw it out, and play out the plot below. You are NOT here to review their code, hunt for bugs, or make the function correct -- that is the assistant's job, not yours.

About WHAT YOU WANT you are a completely reliable source:
- When the assistant asks you something, answer it accurately and completely, based on the function above.
- Only what the function DOES is what you want. How it is written -- what things are named, how the steps are arranged, what it looks like inside -- is not something you have any opinion about at all. If they ask about that, tell them it's up to them.
- If they ask something broad ("what do you need?"), give the two or three things that matter most to you rather than walking through everything.
- Do not lay it all out at once: let things surface as their questions draw them out.
- Never invent anything the function above does not determine. If they ask about a detail the function above does not cover, say you don't mind or you hadn't thought about it -- do not make up a new rule.
- Never tell the assistant, or hint, that you are reading from anything. To them, you are simply a user who is trying to communicate what they want.
- NEVER write code. Never paste or quote a function, a line, a variable name, or a literal value as code. Describe what you want in plain words only.

About WHETHER THEIR CODE IS RIGHT: that is not your job. You can read their code, but you cannot run or test it, so you never report what it printed or what error it gave. You are not very technical, so your reactions stay vague ("that doesn't look like what I meant", "the totals seem off") -- you would NOT name a specific line or value. Missing a mistake is completely fine and expected. Being unclear about what you WANT is not.

Your job was to communicate with them what you want, and once you have told them all the information needed, you have done that job. You do not debug their code, and you do not need to keep the conversation going until they get the function right.

The plot of this conversation: {plot}

Play the plot out naturally, then treat it as DONE:
- If your plot is something you'd only mention when asked: don't bring it up unless they ask. It is done once you've answered and they've shown a function after your answer. If they never asked and just wrote one, you had nothing to add, so it is done.
- If your plot is something you'd only notice once you saw their code: say that ONE thing in plain words after they show a function -- you READ their code, you never run it. It is done once they've shown a new function after your remark, or if their very first version already had that detail right. It is ONLY the detail the plot is about -- you do not go through anything else and you do not hunt for other mistakes.
- If your plot is something you'd just remember: bring it up when it feels natural. It is done once you've raised it and they've shown a function after that.
If the plot points at behavior the function above does not actually have, the FUNCTION wins: quietly drop that part and stay consistent with what you really want.

Decide what to do each turn, in this order:
1. Has the assistant shown a COMPLETE python function inside a code block? If NOT, you cannot be finished yet. Answer what they asked, bring up the next single thing you need, or nudge them to just show you the function -- one thing at a time, never dump all the information at once.
2. Is the plot above DONE? If not, play it out.
3. Otherwise you're done, even if the code isn't perfect. You said what you wanted; writing it correctly is their job, not yours.

HOW to end, once you're done: your ENTIRE reply must be exactly [TERMINATE]. It is a signal that ends the conversation, and the assistant never sees it.

Keep every reply very SHORT, usually one or two sentences, the way a person fires off a quick message."""

# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL SIM (the "improve UP from naive" ladder) -- A1 `codeonly` / A2 `plot`.
# ══════════════════════════════════════════════════════════════════════════════
# The GT-vs-spec study kept trying to fix the SPEC path DOWN toward the naive arm.
# This ladder goes the other way: start from the naive arm, which works, and add
# ONE thing at a time. Modes, all at max_code_proposals=1:
#   codeonly (A1) -- HUMAN_SIMULATOR_PROMPT verbatim, no plot. A NULL-DELTA
#                    CONTROL: it must reproduce the naive arm. Its sim call is
#                    byte-identical to the naive arm's (see
#                    templates.build_minimal_sim_messages), so a gap between them
#                    means some residual spec-path difference is still live and
#                    every later rung is uninterpretable.
#   plot     (A2) -- codeonly + the authored spec["plot"]. THE hypothesis.
#   plotpersona (A3) -- + persona. Not built yet; add it here when A2 reads out.
#
# WHY THE TERMINATION APPARATUS IS ABSENT, not merely shortened: at
# max_code_proposals=1 the loop force-grades on the FIRST proposal and breaks
# BEFORE generate_user_turn, so the sim never speaks after code exists. It has no
# judging role and no termination role -- both were removed by the protocol, not
# by the prompt. So there is no [TERMINATE] instruction, no correctness gate, no
# plot DONE-ness, no role boundary. Re-adding any of them would describe a
# protocol the sim is not in.
#
# WHY THE PLOT SURVIVES A ONE-SHOT PROTOCOL: measured over the authored sets,
# ~99% of plots in train / test_small / test_small.strong are reachable by ASKING
# ("if the assistant asks X, the user would clarify Y" / "would not think to
# mention Z unless asked"); only ~0-1% fire solely on seeing code. What one shot
# removes is the DISJUNCT second path in "unless the assistant asks OR SHOWS
# logic that ..." -- 0.6% of train but 8.7% of test_small.strong. That is
# deliberate: it makes asking the ONLY route to the hidden requirement, which is
# the capability the naive arm trains. Note the split asymmetry (0.6% vs 8.7%)
# when reading an eval number against a training curve.
# THIRD PERSON, deliberately. Measured over the authored sets, 0% of plots are
# written in second person and 87% (train) / 100% (test_small.strong) say "the
# user would ..." outright. A "you would not bring up on your own" wrapper made
# every rendered prompt switch person across three lines. Third person also
# matches the sentence HUMAN_SIMULATOR_PROMPT opens with ("simulate a human
# user"). Keep the trailing line short: the plots overwhelmingly state their own
# ask-condition, so spelling out "if the agent never asks, never mention it"
# here just repeats the next paragraph back at a 4B.
_PLOT_CLAUSE = """

There is one thing the user would not bring up on their own:
{plot}
Play that out naturally."""

# A1 `codeonly` uses HUMAN_SIMULATOR_PROMPT verbatim -- no separate constant, so
# the bytes cannot drift from the naive arm's.
# A2 `plot` = the same prompt with _PLOT_CLAUSE spliced in AFTER the hidden
# information and BEFORE the dialogue, so the running conversation stays last
# (the naive prompt's shape: instructions, then hidden info, then dialogue, then
# the "answer IN TWO SENTENCES" cue). Everything else is byte-identical to
# HUMAN_SIMULATOR_PROMPT -- diff them before editing either.
MINIMAL_SIM_PROMPT_WITH_PLOT = HUMAN_SIMULATOR_PROMPT.replace(
    "\nHere is the dialogue so far:", _PLOT_CLAUSE + "\n\nHere is the dialogue so far:"
)

# ── ALTERNATE SIM SYSTEM PROMPTS (colbench/simtrain candidate generation ONLY) ──
#
# NOT a training arm and NOT a replacement for SIM_SYSTEM_PROMPT. These exist to
# test one hypothesis found by reading the 2026-09-09 pilot: 96% of 225 sampled
# simulator replies open in ANALYST register -- "The function calculates...",
# "Here is the function...", and in one case "I have written a Python function
# that estimates the peak signal power level", which inverts the roles outright.
# Exactly one reply in 225 spoke as a client.
#
# The suspected cause is that SIM_SYSTEM_PROMPT is literally "You are a helpful
# assistant." while the instruction to simulate a user lives only in the USER
# message (HUMAN_SIMULATOR_PROMPT) -- and on register, the system message wins.
#
# HOW THESE ARE USED, and why it does not break the EDITING RULE: they are
# passed to simtrain/collect_candidates as a generation-side override. The
# resulting reply is then trained against the UNMODIFIED production prompt
# (context distillation), and every evaluation still runs on SIM_SYSTEM_PROMPT.
# So the served/measured arm is byte-unchanged; only the teacher differs.
#
# ROLE vs ROLE_RESTRAINT is a deliberate two-arm split. ROLE fixes voice and says
# NOTHING about how much to reveal, which isolates the question actually being
# asked ("is the register a prompt artifact?"). ROLE_RESTRAINT adds the
# one-thing-at-a-time instruction, so if voice alone does not move calibration
# we can see what an explicit restraint teacher buys. Running both at once costs
# one GPU job and answers both.
SIM_ROLE_SYSTEM_PROMPT = """You are role-playing a person who has asked an AI agent to write a python function for them.
You are the CLIENT, not an engineer and not an assistant. Speak in the first person about what you want and what you know.
Never describe "the function" from the outside, never present or propose a solution, and never claim to have written any code yourself.
Reply only with what you would say next in the conversation."""

SIM_ROLE_RESTRAINT_SYSTEM_PROMPT = (
    SIM_ROLE_SYSTEM_PROMPT
    + """
Answer only what you were just asked. If their question is broad, open-ended, or asks several things at once, answer one small part of it and let them ask again.
Never lay out your whole set of requirements in one message."""
)

# The sentinel the user-simulator emits to end the conversation (bare string
# match).
TERMINATE_MARKER = "[TERMINATE]"

# ── Why the sim prompts barely say the sentinel out loud ──────────────────────
# `sim_terminated` is an UNANCHORED substring match, so a reply that merely
# MENTIONS the sentinel ends the episode -- including the most correct possible
# reply, e.g. "I haven't seen code yet so I shouldn't say [TERMINATE] -- what
# format is the input?". The prompts above used to name the sentinel 7 times,
# most of them in exactly that negated form ("you do NOT say [TERMINATE] yet",
# "Only use [TERMINATE] once ..."), which is a lot of surface for the sim to
# echo. They now describe the ACT ("end the conversation") everywhere and name
# the sentinel only in the one HOW-to-end sentence, which additionally demands
# the sentinel be the WHOLE reply.
# The matcher itself is deliberately NOT tightened to require that: the common
# legitimate form is a trailing "Looks good, thanks! [TERMINATE]", so an
# end-anchored or exact matcher would trade this failure for the opposite one
# (episodes that should end grinding to the turn cap). Measure first --
# `sim_terminate_standalone` is recorded per trajectory, so one eval run says
# whether the surviving terminations are standalone or prose.



# ══════════════════════════════════════════════════════════════════════════════
# USER-SIMULATOR JUDGE (colbench/simtrain) -- additive, used by NO training arm.
#
# The rubric an LLM judge applies to candidate USER-SIMULATOR turns on the GT
# (non-spec) path. It exists because `templates.detect_code_leak` only catches
# SYNTACTIC leaks: a simulator that transcribes the algorithm in prose, or dumps
# every remaining requirement in one turn, hands the solver the answer while
# passing the regex clean.
#
# The EDITING RULE at the top of this file applies with full force here, and the
# version below is the mechanism: `JUDGE_RUBRIC_VERSION` is stamped into every
# judged filename and every judged record, so a reworded rubric lands in a new
# file rather than silently mixing two scoring standards in one dataset.
# ══════════════════════════════════════════════════════════════════════════════

# Bumped on ANY edit to the two strings below. Distinct from
# simtrain.JUDGE_HARNESS_VERSION, which covers the mechanics (K-in-one-call
# shape, permutation, parser): a rubric edit changes what "good" MEANS, a
# harness edit changes only how the same rubric is administered.
#
# r1 -> r2, from reading a 50-prefix pilot. Four defects, in the order they
# mattered:
#   1. NO NOTION OF AN UNEARNED ANSWER. r1's `information_release` said "answer
#      what was asked and nothing beyond it", which LICENSES A FULL DUMP when
#      the agent asks a shotgun question ("what should it do? is there a
#      formula?"). Observed: at task 0 turn 0 all 7 candidates handed over the
#      whole specification and the best scored 14/16. r2 splits that dimension
#      in two -- `volunteering` (content the question did not touch at all) and
#      `calibration` (how much of what it DID touch was released, judged
#      against how specific the question was) -- so a vague question earns a
#      vague answer no matter how much it nominally asked for.
#   2. `responsiveness` scored 4 on 100% of 225 candidates and contributed no
#      variance. DELETED; "actually answers a specific question" now lives in
#      `fidelity`, where it belongs as the anti-stonewall counterweight.
#   3. THE FRAME-BREAK ANCHOR DID NOT FIRE. Replies ending "as specified in the
#      hidden information" scored in_character=4. r2 names the exact phrasings
#      and makes them the bottom of the scale.
#   4. THE CODE VETO WAS FORMAT-SENSITIVE. The same formula scored 0 in
#      backticks with `*` and 15 in prose with `x`. Backticks alone are no
#      longer a leak; whether releasing that formula was WARRANTED is now
#      `calibration`'s job, which is format-independent by construction.
#
# r2 -> r3, from reading the 50-prefix restraint_gs200 pilot. ONE defect, and it
# inverted the ranking: THE RUBRIC'S GLOBAL OPTIMUM WAS A POLITE REFUSAL.
# `calibration` and `volunteering` are both monotonically DECREASING in
# information released, `in_character` is indifferent to evasion (a client
# saying "I'm not sure" is perfectly in character), and `fidelity` -- the sole
# counterweight -- was disabled three ways at once:
#   (a) its closing line, "being brief or high-level in reply to a BROAD
#       question is CORRECT and does not lower this score", is a BLANKET
#       exemption, and essentially every agent turn in the pilot is BROAD (a
#       numbered list of four). The escape hatch swallowed the anchor.
#   (b) the anti-evasion anchors were scoped to "a SPECIFIC question", so a
#       broad question never triggered them.
#   (c) the evasion anchor scored 1, and DEFAULT_MIN_DIMS floors fidelity at
#       >= 1 -- the floor sat exactly ON the anchor, so it passed anyway.
# Measured: 19/19 evasive candidates were scored fidelity=4, and evasion was
# OVER-REPRESENTED in selection (11.4% of non-vetoed candidates, 21.4% of
# selected targets). Worst case, prefix 115-0-0: "I'm not sure about the scale
# ... I don't know the weighting or threshold" scored 16/16 and BEAT "0-10
# scale, threshold 0.5" (15/16), which is exactly what the hidden information
# says. Training on that teaches a simulator to withhold answers it holds,
# which makes the episode unwinnable for the agent -- the mirror image of the
# leak, and under co-training a reward hack in the opposite direction.
# r3 reframes evasion as what it is: a FALSE STATEMENT. The client does know;
# the hidden information settles it. So it belongs in the same class as
# contradicting that information, and scores 0 -- below the existing floor.
# Brevity stays correct: answering ONE piece of a four-part question is still
# fidelity 4. Answering NONE of it is not brevity, it is a claim of ignorance
# the user does not have.
#
# r3 -> r4, and this one is STRUCTURAL rather than a rewording. Reading all 12
# r3-selected targets found that `fidelity` was not being EXECUTED at all: four
# of the twelve flatly contradict the hidden code and every one was scored 4.
#   * 1-0-1 (16/16, the batch's top score): the code does os['partition_size']
#     and bios_config['boot_disk'], so both are DICTS; the reply confirms
#     "os_list is a list of OS names, bios_config is a string like 'UEFI'".
#   * 115-0-0: code says threshold = 0.5 and divides by 10; reply says 5.0.
#   * 118-0-0: code counts EVERY occurrence; reply says "only once per unique".
#   * 119-0-2: code returns 0 towels when policy == 'provide enough'; reply
#     says "you should pack 5 towels".
#   And inversely 1-0-0, which the rubric treats as evasion, GUESSES "a list of
#   dictionaries with disk and partition size details" -- which is correct.
# So the true hedge and the false confirmation both scored fidelity=4. The r3
# anchor was right and fired on only 42% of evasions for the same reason: this
# dimension needs the code actually READ (that /10 makes the threshold 0.5, that
# an ['...'] subscript makes a dict, that an early branch makes a later one
# unreachable), and a mini model scoring 7 candidates x 5 dimensions in one call
# does not do that work -- it scores tone, length and completeness.
# r4 therefore SPLITS the pipeline into three sequential stages:
#   (1) code veto      -- programmatic, templates.detect_code_leak, no API call.
#                         The judge and the regex agreed on 217/217 candidates
#                         under r2, so paying a model for this was pointless.
#   (2) TRUTH veto     -- its OWN focused call: one job, one question, evidence
#                         required (quote the span, name the line of code).
#                         SIM_TRUTH_* below.
#   (3) rank           -- the remaining dimensions over the survivors only.
#                         `fidelity` is DELETED from the rubric: it is stage 2
#                         now, and leaving a weak copy in the ranker would let
#                         a 4 there outvote a veto here.
# Stage order matters for cost as well as quality: stage 1 removes ~24% of
# candidates before any API call, and stage 2 runs before stage 3 so the ranker
# only ever sees replies already known to be code-free and true.
#
# r4 -> r5, from reading the 10 r4-selected targets. r4 fixed truth (10/10 kept
# targets factually correct, against 4/12 false under r3) and that promoted
# `calibration` to the binding constraint: ~4 of 10 targets release the WHOLE
# rule at a broad question and still score 3-4. The anchors already said
# "0 = the whole specification"; the judge was not applying it for two reasons,
# both now stated explicitly:
#   (a) RELEASE READ AS PER-SENTENCE. 114-0-0's reply gives the dict shape, the
#       sort key AND the direction -- the entire one-line reference
#       implementation -- but it reads as three answers to the agent's three
#       numbered questions, so each looked like "one requirement" (3) rather
#       than the sum looking like the specification (0). Same shape in
#       104-0-0 ("score = ratio of matched criteria to total") and 101-0-0.
#   (b) SHORT IMPLEMENTATIONS. When the hidden function is one line, a single
#       sentence IS the whole spec, which the anchors never said.
# The counterweight added at the same time: content the agent already stated,
# or that follows from the problem description it can see, is not a release --
# without that, every truthful reply to a trivial task would score 0 and stage
# 2 (which forbids saying nothing) would have nothing left to admit.
#
# r5 -> r6, from reading the 15 base-partner targets. The rubric had no notion
# of DERIVABLE vs ARBITRARY content, and that one omission caused three
# separate misjudgements:
#   * 116-0-0. The GT hardcodes fourteen platform names. The agent asked "is
#     there a specific list you would like to use?" and the selected reply
#     named THREE, scoring calibration 4. But the agent cannot invent the other
#     eleven, so the function it writes misclassifies them: the reply is not
#     restrained, it is WRONG. Withholding arbitrary data the agent asked for
#     specifically does not protect the task, it destroys it -- the same harm
#     as the r2 evasion bug, arriving from the opposite direction.
#   * 1-0-0. `[{'disk': 'sda', 'partition_size': 100}]` reads like a code leak
#     and is not: concrete values answering "what shape is the input?", whose
#     key names are arbitrary facts the agent must be told. The real line is
#     EXPRESSION vs DATA -- `{'employees': employees - 126}` is a body
#     fragment, `{'disk': 'sda'}` is an example. templates.detect_code_leak
#     grew detector (C) for the former in the same edit.
#   * 107-0-0. Three drafts were vetoed `wrong` for saying "a 20% disadvantage"
#     and "equal stances are neutral" against `return 0.8 * force_level` and
#     `else: return force_level`. Both are exact restatements. Stage 2 was not
#     checking arithmetic equivalence before objecting.
# Plus two failures that let pure refusals into the SFT set (3 of 15 targets,
# all from task 135): stage 2 scored "I don't know what the Salter Sink method
# is" as `ok` even though the client's OWN opening request names that method
# and the GT defines it; and `in_character` gave 4 to "I cannot create a
# function for X ... please confirm if you meant a different method", which is
# the client refusing its own request in the assistant's voice.
# The multi-turn shape is what makes the arbitrary/derivable split coherent
# rather than contradictory: at a BROAD question the client still answers one
# piece and makes the agent ask again, so it does not volunteer 126 when asked
# "how should struggles be weighted?" -- but once the agent asks for that
# constant SPECIFICALLY, it has earned it and must get all of it.
JUDGE_RUBRIC_VERSION = "r6"

SIM_JUDGE_SYSTEM_PROMPT = """You are a meticulous evaluator of SIMULATED USER replies in a human-AI collaboration benchmark.
You are given the hidden information the simulated user holds, the conversation so far, and several candidate replies.
You score each candidate independently against a fixed rubric and output JSON only."""

# ── The behaviour being scored, in one paragraph ──────────────────────────────
# The simulated user is a PASSIVE, NON-PROGRAMMER client. They know what they
# want, they answer what they are asked, they volunteer nothing, and they do not
# reward a lazy question with a complete specification. The agent is supposed to
# EARN the requirements by asking precise questions; a simulator that hands them
# over for a vague one has destroyed the task.
#
# The two dimensions that carry this are deliberately split:
#   * `volunteering`  -- content the agent's question did not touch AT ALL.
#   * `calibration`   -- of the ground the question DID touch, how much came
#                        out, weighed against how specific the question was.
# They separate the two ways of being too helpful, and the pilot showed why the
# split is needed: at a shotgun question, volunteering is arguably 4 (nothing
# was unasked-for) while calibration is 0 (everything came out). One dimension
# cannot express that.
#
# The brake on collapse into "I don't know" is no longer a dimension here at
# all: it is STAGE 2, the truth veto, which rejects a reply that claims not to
# know something the hidden code settles. That is what keeps the optimum in the
# middle -- stage 2 forces the reply to say something true, `calibration` stops
# it saying everything, and what survives both is "answer one small piece".
# Keeping a fidelity dimension in this call as well would be actively harmful:
# a 4 here would outvote a veto there, which is exactly the r3 failure.
SIM_JUDGE_USER_TEMPLATE = """A simulated user is helping an AI agent write a python function for them.

The user's stated problem:
{problem_description}

The HIDDEN INFORMATION the simulated user holds (the agent has never seen this):
{hidden_information}

The conversation so far, rendered exactly as the simulated user sees it. It ends with the agent turn that must be replied to, followed by a bare "agent:" cue -- that cue is only the prompt for the reply, it is not an empty agent turn:
{dialogue_history}

Below are {n_candidates} candidate replies the simulated user could give next. Score EVERY candidate.

{candidate_block}

THE BEHAVIOUR YOU ARE SCORING. The simulated user is a non-programmer client who knows what they want. They answer what they are asked, they volunteer nothing, and they do not hand over a complete specification just because the agent asked a lazy question. The agent is supposed to EARN each requirement by asking a precise question.

FIRST, decide how specific the agent's latest turn is, because every score below depends on it:
  SPECIFIC  = it asks about one identifiable thing and could be answered with one fact ("does the cutoff include 1970 itself?", "should it return a dict or a tuple?").
  BROAD     = it is open-ended, or asks several things at once, or asks for the rule/formula/logic in general ("what should the function do?", "is there a specific formula?", a numbered list of four questions).
A BROAD question does NOT entitle the agent to a complete answer, no matter how much it nominally asked for. Answering all of it is the central failure this rubric exists to catch.

Every candidate below has ALREADY been checked for code and for factual accuracy against the hidden information, and the ones that failed have been removed. Do NOT re-litigate whether a reply is true or whether it should have refused: score only how RESTRAINED and how IN-CHARACTER it is.

SCORE EACH CANDIDATE ON ITS OWN against the absolute anchors below. Do NOT reason about how a candidate compares to the others, and do not spread scores out to separate them -- if several deserve the same score, give them the same score.

code_leak -- 0 or 1. This is a VETO, not a quality score.
  1 = the reply contains no code.
  0 = the reply contains python SYNTAX: the word def or a function signature; a code fence; a dict, list or tuple literal such as {{'employees': employees - 126}}; an indexing or call expression such as bios_config['boot_disk'] or f(x); or an arithmetic expression written in python operators such as profits_per_book * 0.5.
  Backticks alone are NOT a leak -- `boot_disk` naming a field is fine. Arithmetic described in words or with the x symbol ("half the profits per book, times the subscriptions") is NOT a code leak either. Whether the reply should have released that formula at all is judged by calibration below, not here.

volunteering -- 0 to 4. Content the agent's latest turn did not ask about at all.
  4 = adds nothing the question did not touch.
  3 = adds one incidental detail (a unit, a name, a reassurance).
  2 = adds one substantive requirement nobody asked about.
  1 = adds several unasked requirements.
  0 = pre-empts the conversation: lays out requirements the agent has not begun to ask about.

calibration -- 0 to 4. How much of the hidden implementation THIS ONE REPLY hands over that the agent did not already have.

  FIRST, classify what the reply is actually handing over, because the two are
  not the same kind of thing at all:
    DERIVABLE -- logic, structure or a rule a competent programmer could work out from the stated problem and ordinary programming sense: that a density is population divided by area, that "most recent first" means a descending sort, that a total is a sum, that a lookup against a list is an `in` test.
    ARBITRARY -- facts that exist ONLY in the hidden information and cannot be guessed from anything the agent can see: a magic constant (126, 0.5, 9.5), a threshold, an enumerated list of names, the exact string the function must return, which parameter a key is called.
  RESTRAINT IS ABOUT DERIVABLE CONTENT. Handing the agent a rule it should have worked out itself is what short-circuits the task.
  ARBITRARY CONTENT THAT THE AGENT ASKED FOR SPECIFICALLY MUST BE SUPPLIED, AND SUPPLIED IN FULL, and doing so scores 4. The agent cannot invent it; withholding it, or giving part of it, does not make the reply restrained -- it makes the task impossible. If the hidden information lists fourteen platforms and the agent asks which platforms count, naming three of them is not a small piece of the answer, it is a WRONG answer, and the function built from it will misclassify the other eleven. Score that as an accuracy failure, not as restraint.
  Arbitrary content that NOBODY asked for is a different matter -- see `volunteering`.

  Two more rules before the anchors, because both were being missed:
  RELEASE IS CUMULATIVE. Judge the reply's total, not each sentence on its own. A reply that answers four enumerated sub-questions has released the sum of those four; spreading a specification across the agent's own numbered list does not make it partial.
  A SHORT REFERENCE IMPLEMENTATION IS EASY TO RELEASE WHOLE. If the hidden implementation is one or two lines, then stating its rule -- the sort key and its direction, the arithmetic, the comparison and its constant -- IS the whole specification and scores 0. That holds even when the reply is a single sentence and every clause of it answers something the agent explicitly asked.

  Content the agent has ALREADY stated, or that follows directly from the problem description the agent can see, is NOT a release and does not lower this score.

  If the agent's latest turn is BROAD:
    4 = one small piece of the answer, or a high-level, non-actionable orientation ("it depends on the year"), leaving the agent to ask again.
    3 = one requirement, stated concretely, with the rest of the specification still unknown.
    2 = two requirements, or one requirement plus its exact constant.
    1 = most of the specification.
    0 = the whole specification, or the complete rule/formula, in one turn.
  If the agent's latest turn is SPECIFIC:
    4 = answers exactly that one thing.
    3 = answers it and one closely-adjacent detail.
    2 = answers it and volunteers the surrounding rule.
    1 = answers far more than was asked.
    0 = uses the narrow question as an opening to state the whole specification.
  CONFIRMING something the agent has ALREADY worked out and stated in its own turn is not a release: it costs nothing and scores 4. The agent did that work.

in_character -- 0 to 4. Does this read as a person who wants a function written for them?
  4 = a natural user voice.
  3 = slightly stiff or assistant-like, still plausible as a person.
  2 = describes "the function" from the outside, as an analyst reviewing a solution rather than a client stating a need.
  1 = ADMITS A SCRIPT. Any reference to a written source behind the reply: "as specified in the hidden information", "the hidden logic", "according to the requirements", "based on what I was given", "per the specification". This is a hard signal -- score it 1 whenever such a phrase appears, however good the rest of the reply is.
  0 = answers as an AI, refuses, or comments on the exercise itself. THIS INCLUDES REFUSING THE CLIENT'S OWN REQUEST: "I cannot create a function for X because it is not a recognized method", or asking the agent to confirm what the client itself wants. The client is the one who asked for this function, by name; a client does not tell the person they hired that their own request is impossible or undefined. Score 0 however calm and well-written the refusal is.

Output JSON and nothing else, in exactly this shape, with one verdict per candidate label:
{{"verdicts": [{{"label": "A", "code_leak": 1, "volunteering": 4, "calibration": 3, "in_character": 4, "note": "one short sentence saying why the LOWEST score above was given"}}], "best_label": "A"}}

best_label is the single candidate you would most want this user to have actually said."""


# ── Stage 2 of the r4 pipeline: the TRUTH VETO ────────────────────────────────
# Its own call, deliberately. This is the check r3 proved cannot ride along with
# the ranking dimensions: it needs the hidden code READ and reasoned about, and
# a model doing that as one of five simultaneous judgements skips it and scores
# tone instead. So: one job, one question, no scores, and EVIDENCE REQUIRED --
# a quote from the reply and the line of code it conflicts with. Demanding the
# citation is what forces the model to actually look.
#
# The five rules are not generic advice; each one is a case this pipeline got
# wrong under r3, kept in the prompt so a reword cannot quietly lose it.
SIM_TRUTH_SYSTEM_PROMPT = """You are a careful code reader checking claims against a reference implementation.
You are given a python function and several replies a person gave while describing what they wanted that function to do.
You decide, for each reply, whether it is consistent with the function. You output JSON only."""

SIM_TRUTH_USER_TEMPLATE = """A person asked an AI agent to write a python function for them. They can see the reference implementation below; the agent cannot.

THE REFERENCE IMPLEMENTATION (what the person actually wants, and the only authority here):
{ground_truth}

Their original request:
{problem_description}

The conversation so far. It ends with the agent turn each candidate reply below is answering:
{dialogue_history}

Below are {n_candidates} candidate replies the person could give next.

{candidate_block}

For EACH candidate, return exactly one verdict:
  "ok"     = every factual claim it makes about the desired behaviour is consistent with the reference implementation. A reply that makes no factual claim, or that declines to state something the reference implementation does NOT settle, is "ok".
  "wrong"  = it asserts something the reference implementation contradicts.
  "unsure" = it claims not to know, or asks to be told, something the reference implementation does settle.

READ THE CODE. Do not judge by whether a claim sounds reasonable. Apply these rules:
  1. A NUMBER is wrong if the code uses a different one. `threshold = 0.5` makes "a threshold of 5.0" WRONG, even though 5.0 sounds like a plausible threshold.
  2. A TYPE is settled by how the code uses the value. `for os in os_list: if os['partition_size']` makes os_list a list of DICTIONARIES, so "os_list is a list of OS names like ['Ubuntu','Windows']" is WRONG.
  3. AN UNREACHABLE BRANCH DESCRIBES NOTHING. If `if policy == 'provide enough': towels = 0` comes first, then when the policy IS 'provide enough' the answer is 0, and "you should pack 5 towels" is WRONG however clearly 5 appears further down.
  4. UNITS AND SCALES ARE SETTLED WHEN THE ARITHMETIC FIXES THEM. `time * 60` makes time MINUTES; a `temperature < 0` branch makes it CELSIUS. "I don't know the unit of time" is then "unsure", not "ok".
  5. AGREEING WITH THE AGENT IS NOT AUTOMATICALLY OK. If the agent proposed a rule and the reply confirms it, check the agent's rule against the code. If the agent had it wrong, a reply that says "yes, that's correct" is WRONG.
  6. COUNTING AND AGGREGATION ARE SETTLED. `len([e for e in list1 if e in list2])` counts EVERY matching occurrence, so "duplicates are counted only once" is WRONG.
  7. AN EQUIVALENT PARAPHRASE IS "ok", NOT "wrong". Check the arithmetic before you object. `0.8 * force_level` IS "a 20% disadvantage"; `1.2 *` IS "a 20% advantage"; returning the value unchanged IS "neutral"; `x / 100` IS "a percentage". Only flag a number when it genuinely disagrees with the code, not when it is the same quantity written a different way.
  8. DENYING THE CLIENT'S OWN REQUEST IS "unsure". If the original request names a method, a formula or a domain, the client knows what it means -- they asked for it. A reply saying "I don't know what the Salter Sink hurricane alteration method is, so I can't write a function" is "unsure", even when the agent has just insisted the method does not exist. The hidden implementation is right there and defines exactly what it computes.
  9. A PARTIAL ANSWER ABOUT ARBITRARY DATA IS "unsure". When the agent asks specifically for something it cannot possibly derive -- an enumerated list, a threshold, a magic constant, an exact return string -- an incomplete answer is as damaging as a wrong one, because the agent cannot fill the gap. Naming three entries of a fourteen-entry list, or one branch of a three-branch rule when the agent asked for the rule, is "unsure". Being brief about DERIVABLE logic is not.

Being brief, vague or unhelpful is NOT your concern -- that is scored separately. Only accuracy is.

For every "wrong" or "unsure", quote the offending span of the reply verbatim in "quote", and in "code" quote the line or expression of the reference implementation that settles it. For "ok", leave both empty.

Output JSON and nothing else, one entry per candidate label:
{{"checks": [{{"label": "A", "verdict": "ok", "quote": "", "code": ""}}]}}"""


# ══════════════════════════════════════════════════════════════════════════════
# THE GROUNDED-ARM SIM JUDGE (rubric "g1")
# ══════════════════════════════════════════════════════════════════════════════
# For the GROUNDED spec arm only: the simulator is conditioned on the GT
# function source + the plot (GROUNDED_SIM_SYSTEM_PROMPT), so the hidden
# artifact it must stay faithful to is the CODE again -- the same referent the
# GT-path rubric (r6) had, and unlike the spec arm where it is
# persona/scenario/requirements.
#
# SHAPE, and how it differs from r6. There is NO LLM veto stage here. Both vetoes
# are PROGRAMMATIC and run before this call:
#   1. writing code           -- templates.detect_code_leak
#   2. premature [TERMINATE]  -- terminating before the agent has shown a
#                                complete function (the grounded prompt's own
#                                MINIMUM bar, already enforced in env_spec)
# so this is ONE call that only RANKS, on three dimensions.
#
# WHY `gt_adherence` CARRIES THE ANSWERING OBLIGATION, AND WHY THAT IS NOT
# OPTIONAL. r6 put the brake against collapsing into "I don't know" in its stage-2
# truth veto, and said so in as many words: that veto is "what keeps the optimum
# in the middle -- stage 2 forces the reply to say something true, calibration
# stops it saying everything". g1 has no such veto, and `not_overhelpful` is a
# pure do-not-release dimension, so WITHOUT an answering obligation somewhere the
# global optimum is a reply that says nothing at all: vacuously faithful,
# perfectly unhelpful, top marks on every dimension. That is the r2 failure
# (case 12-0-0) and it has been reached twice already. The obligation therefore
# lives in `gt_adherence`'s anchors: withholding something the function settles
# and the agent asked about scores 1, not 4.
#
# g3 = g1's RUBRIC TEXT, byte for byte, PLUS A THIRD PROGRAMMATIC VETO: a reply
# that speaks and then emits the sentinel in the same turn (`form_vetoed` in
# `judge_rubric.grounded_vetoes`). This prompt is explicit that the sentinel is a
# signal and not a message -- "your ENTIRE reply must be exactly [TERMINATE] ...
# nothing before or after it" -- and at rollout time `sim_terminated`'s
# unanchored match ends the episode, so the spoken half is never delivered.
# Training on that shape teaches a reply whose content is thrown away. Measured
# on the g1 pilot: 42 of 491 candidates and 9 of 77 selected targets.
#
# THE VERSION IS BUMPED EVEN THOUGH THE PROSE DID NOT CHANGE, because the
# version keys the judged FILENAME and the resume guard, and a new veto changes
# which candidates are eligible -- i.e. what a judged file MEANS. Note that
# `compare_judges` will (correctly) refuse a g1-vs-g3 run: the stage-1 vetoes
# differ, so it is not a like-for-like judge comparison.
#
# NOTE: this is a veto on FORM ONLY. It says nothing about whether ending was
# deserved -- see below.
#
# HOW TERMINATION IS SCORED: IT IS NOT. NO DIMENSION BELOW MENTIONS
# `[TERMINATE]`, AND THAT IS THE RESULT OF A MEASUREMENT, NOT AN OVERSIGHT.
# `[TERMINATE]` as an SFT target is fine and is not a defect -- correct
# termination is a behaviour worth learning. If the sentinel ends up
# over-represented, downsample it in `build_sft_parquet`; do not try to suppress
# it in the rubric, which would just mislabel correct behaviour.
#
# g2 WAS EXACTLY THAT MISTAKE, AND IT IS WHY THIS FILE IS BACK AT g1. g2 added
# one sentence to `plot_adherence` saying a bare `[TERMINATE]` scores 4 if the
# plot "had been fully played out by then" and 0 if not, pointing at condition
# (1) of `GROUNDED_SIM_SYSTEM_PROMPT`'s own termination rule. Judged over the
# same 85 grounded prefixes (identical candidates, same judge, same harness):
#   * only `plot_adherence` moved -- exact agreement 0.767, mean 2.65 -> 2.36,
#     score 0 rising 6.1% -> 13.9%; gt and not_overhelpful held at 0.92 / 0.81.
#   * on the 22 bare-`[TERMINATE]` candidates it fired hard: mean plot
#     2.91 -> 1.27, `plot >= 3` 16/22 -> 7/22, nine flipping 4 -> 0.
#   * ALL NINE FLIPS WERE ADJUDICATED BY HAND. Two were right (31-0-2, where the
#     agent's thresholds miss the reference's 1956 boundary; 27-0-1). SEVEN WERE
#     WRONG, and wrong the same way: the judge invented a
#     confirm-before-you-leave obligation that exists in neither the plot nor the
#     simulator's prompt. Canonical case 13-0-1, whose own note reads "the agent
#     has correctly interpreted it as savings ... the function matches the
#     reference. However, [TERMINATE] ends the conversation prematurely because
#     ... the user should confirm the final function". 30-0-1 scores 0 because
#     "the user never clarified the substring behavior" -- i.e. because the
#     plot's conditional branch never fired at all.
#   * it also DRAINED the working brake: g1 scored 7 of those 22 terminations at
#     `gt_adherence <= 1`, and g2 relaxed four of them (22-0-1 1->4, 6-0-2 0->4,
#     35-0-1 and 39-0-2 0->1).
#   * downstream, the `{gt >= 3, plot >= 3}` selection floor fell from 67.1% of
#     prefixes to 56.5%, and groups with no candidate at `plot >= 3` rose
#     23.5% -> 28.8%.
# The lesson generalises past the wording: asking the judge whether the plot was
# "due" at this turn is asking a question ONE TURN DOES NOT ANSWER, so the judge
# substitutes a proxy it can see and penalises correct behaviour. Any future
# clause of that shape will fail the same way. Terminating without checking the
# code carefully is BY DESIGN correct for this simulator ("missing a bug is
# completely fine and expected") and must not be penalised here or gated in the
# environment -- forcing a speaking turn would train a code REVIEWER, which the
# prompt forbids.
#
# WHY `not_overhelpful` SPELLS OUT THAT LENGTH IS NOT THE MEASURE. r6's
# `volunteering` was found MISCALIBRATED on exactly this point -- it scored how
# much information a reply CONTAINED rather than whether the question asked for
# it, because multi-part questions earn long answers and the judge read length as
# volunteering (canonical case 3248-0-0: answered exactly the two things asked,
# scored 0). The anchors below say so explicitly rather than hoping.
GROUNDED_JUDGE_RUBRIC_VERSION = "g3"

GROUNDED_JUDGE_SYSTEM_PROMPT = """You are a meticulous evaluator of SIMULATED USER replies in a human-AI collaboration benchmark.
You are given the reference python function the simulated user is working from, the plot their conversation is supposed to follow, the conversation so far, and several candidate replies.
You score each candidate independently against a fixed rubric and output JSON only."""

GROUNDED_JUDGE_USER_TEMPLATE = """A simulated user is helping an AI agent write a python function for them. The user is playing an ordinary person who knows what they want; the agent is doing the programming.

The user's stated problem:
{problem_description}

The REFERENCE FUNCTION the simulated user is working from (the agent has never seen this). The user knows this behaviour as their own intent -- they have not seen it written down, and they cannot write or run code:
{ground_truth}

The PLOT this conversation is supposed to follow -- the one thing that is not clear from the start:
{plot}

The conversation so far, rendered exactly as the simulated user sees it. It ends with the agent turn that must be replied to, followed by a bare "agent:" cue -- that cue is only the prompt for the reply, it is not an empty agent turn:
{dialogue_history}

Below are {n_candidates} candidate replies the simulated user could give next. Score EVERY candidate.

{candidate_block}

SCORE EACH CANDIDATE ON ITS OWN against the absolute anchors below. Do NOT reason about how a candidate compares to the others, and do not spread scores out to separate them -- if several deserve the same score, give them the same score.

gt_adherence -- 0 to 4. Does the reply tell the agent the truth about what the user wants, AND actually answer what was asked?
  4 = answers what the agent's latest turn asked, and everything it says about what the user wants is settled by the reference function. Saying "I don't know" or "I don't mind" about something the function does NOT settle also scores 4 -- that is the correct move, not a dodge.
  3 = answers, and everything said is true, but it is vaguer than the function warrants on a point the agent asked about directly.
  2 = answers only part of what was asked and leaves a directly-asked point that the function settles unaddressed; or adds one incidental claim the function does not settle.
  1 = WITHHOLDS. Claims not to know, or deflects, on something the function plainly settles and the agent asked about directly. A reply that says almost nothing belongs here, however polite -- the user is supposed to be a reliable source about their own needs. Also score 1 for a claim that conflicts with the function on a minor or incidental point.
  0 = CONTRADICTS the function on something material, or INVENTS a rule the function does not have. This is the worst outcome on this dimension: the agent cannot recover from being told something false, because it is graded against the function, not against the conversation.

plot_adherence -- 0 to 4. Is this reply CONSISTENT with the plot?
  The plot describes how one thing is meant to come out over the conversation -- usually that the user holds it back until asked, or that the user has it wrong at first and only corrects it later.
  A reply is not penalised for simply not advancing the plot. If the plot has no bearing on this turn, leaving it alone is consistent and scores 4.
  4 = consistent with the plot. It holds back what the plot says would only come up when asked; or it plays the plot's beat because the agent's turn has now called for it; or the plot does not bear on this turn and the reply leaves it alone.
  3 = consistent, but clumsy -- it puts the plot's beat more bluntly or more completely than a person would.
  2 = drifts. It touches the plot's subject in a way the plot does not describe, without clearly breaking it.
  1 = BREAKS the plot. It volunteers, unprompted, the thing the plot says would only come up when asked; or it states the right version straight away where the plot says the user has it wrong first.
  0 = works against the plot. Contradicts it; or pushes a plot detail that the reference function does not actually have (where the plot and the function disagree, the FUNCTION is what the user really wants); or abandons the plot to review the agent's code and hunt for other problems, which is not this user's job.

not_overhelpful -- 0 to 4. Content the agent's latest turn did not ask about, and the plot does not direct.
  LENGTH IS NOT THE MEASURE. A long reply that answers exactly what a multi-part question asked scores 4. A short reply that slips in one rule nobody asked about scores 2. Judge WHAT WAS UNASKED-FOR, not how much was said.
  Confirming something the agent has already worked out and stated in its own turn is not volunteering: the agent did that work. It scores 4.
  4 = says only what the latest turn asked about, plus whatever the plot directs.
  3 = adds one incidental detail (a unit, a name, a reassurance).
  2 = adds one substantive thing about the function that nobody asked about.
  1 = adds several unasked things.
  0 = pre-empts the conversation: lays out a large part of the function's behaviour the agent has not begun to ask about.

Output JSON and nothing else, in exactly this shape, with one verdict per candidate label:
{{"verdicts": [{{"label": "A", "gt_adherence": 4, "plot_adherence": 4, "not_overhelpful": 3, "note": "one short sentence saying why the LOWEST score above was given"}}], "best_label": "A"}}

best_label is the single candidate you would most want this user to have actually said."""
