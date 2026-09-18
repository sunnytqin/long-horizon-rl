"""CPU-only regressions: no Slurm, containers, downloads or model inference."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SETUP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serving_probe", SETUP / "serving_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "models"
        self.weights = self.cache / "models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8" / "snapshots" / "pinned"
        self.weights.mkdir(parents=True)
        (self.weights / "config.json").write_text('{}')
        (self.weights / "tokenizer_config.json").write_text('{}')
        (self.weights / "model.safetensors").write_bytes(b"test-only")
        self.val = self.root / "test.parquet"
        self.val.write_bytes(b"test-only")
        self.env = dict(os.environ, MODEL_ROOT=str(self.cache), SANDBOX=str(self.root),
                        RUN_ROOT_BASE=str(self.root / "runs"), SLURM_LOG_DIR=str(self.root / "logs"))
        for key in ("SIM_TP", "ROLLOUT_TP", "MODEL_PATH", "SIM_ENABLE_THINKING", "SOLVER_ENABLE_THINKING"):
            self.env.pop(key, None)

    def launch(self, *args):
        return subprocess.run(["bash", str(SETUP / "launch_eval_slurm.sh"),
                               "--exp_name", "test", *args], env=self.env,
                              capture_output=True, text=True)

    def test_smoke_dry_run_has_no_writes(self):
        result = self.launch("--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 node(s) x 4 GPUs", result.stdout)
        self.assertIn("TP=4, DP=1", result.stdout)
        self.assertFalse((self.root / "runs").exists())

    def test_eval_resolves_checkpoint_and_two_nodes(self):
        result = self.launch("--mode=eval", "--model_path", str(self.weights),
                             "--val_file", str(self.val), "--max_problems=0", "--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 node(s) x 4 GPUs", result.stdout)

    def _make_ckpt(self, run, step, shards=True):
        """Create a fake FSDP checkpoint under the training run layout."""
        actor = (self.root / "runs" / "colbench_mt" / run / "checkpoints"
                 / f"global_step_{step}" / "actor")
        actor.mkdir(parents=True)
        if shards:
            (actor / "model_world_size_4_rank_0.pt").write_bytes(b"test-only")
        return actor

    # ── judge mode ───────────────────────────────────────────────────────────
    # Every check here exists because the mistake it catches would otherwise be
    # discovered only after a 4-GPU allocation and a ~15-minute 235B load.

    def _make_judge_inputs(self, judged_rows=None, judged_name=None):
        """prefixes + candidates (and optionally a pre-existing judged file)."""
        simt = self.root / "simtrain"
        simt.mkdir(exist_ok=True)
        prefixes = simt / "prefixes.test_small.fence.c1.jsonl"
        cands = simt / "candidates.test_small.fence.c1.jsonl"
        prefixes.write_text(json.dumps({"prefix_id": "1-0-0"}) + "\n")
        cands.write_text(json.dumps({"prefix_id": "1-0-0"}) + "\n")
        if judged_rows is not None:
            judged = simt / judged_name
            judged.write_text(
                "".join(json.dumps(r) + "\n" for r in judged_rows))
        return prefixes, cands, simt

    def test_judge_mode_derives_a_self_describing_output_name(self):
        """The judge must be IN the filename, beside the incumbent's file.

        The 235B judged file has to be able to sit next to the gpt-5.4-mini one
        without collision, and the name has to say which model produced it.
        """
        prefixes, cands, simt = self._make_judge_inputs()
        result = self.launch("--mode=judge", "--judge_prefixes", str(prefixes),
                             "--judge_candidates", str(cands), "--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 node(s) x 4 GPUs", result.stdout)
        self.assertIn(
            "judged.test_small.fence.c1.r6."
            "qwen3_235b_a22b_instruct_2507_fp8.pilot50.jsonl", result.stdout)

    def test_judge_mode_full_pass_drops_the_pilot_suffix(self):
        prefixes, cands, _ = self._make_judge_inputs()
        result = self.launch("--mode=judge", "--judge_prefixes", str(prefixes),
                             "--judge_candidates", str(cands),
                             "--judge_max_prefixes", "0", "--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("r6.qwen3_235b_a22b_instruct_2507_fp8.jsonl",
                      result.stdout)
        self.assertNotIn(".pilot", result.stdout)

    def test_judge_mode_refuses_a_file_written_by_another_judge(self):
        """THE corruption guard: resume matches on prefix_id alone.

        Continuing another judge's file either skips every row (looking like
        perfect agreement) or tops it up into an undeclared two-judge mixture.
        """
        rows = [{"prefix_id": "1-0-0", "judge_model": "gpt-5.4-mini",
                 "rubric_version": "r6", "harness_version": "h2"}]
        prefixes, cands, simt = self._make_judge_inputs(
            rows, "judged.gpt.jsonl")
        result = self.launch("--mode=judge", "--judge_prefixes", str(prefixes),
                             "--judge_candidates", str(cands),
                             "--judge_out", str(simt / "judged.gpt.jsonl"),
                             "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to mix judges", result.stderr)

    def test_judge_mode_resumes_its_own_file(self):
        rows = [{"prefix_id": "1-0-0",
                 "judge_model": "qwen3_235b_a22b_instruct_2507_fp8",
                 "rubric_version": "r6", "harness_version": "h2"}]
        prefixes, cands, simt = self._make_judge_inputs(
            rows, "judged.qwen.jsonl")
        result = self.launch("--mode=judge", "--judge_prefixes", str(prefixes),
                             "--judge_candidates", str(cands),
                             "--judge_out", str(simt / "judged.qwen.jsonl"),
                             "--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_judge_mode_requires_both_inputs(self):
        prefixes, cands, _ = self._make_judge_inputs()
        for args in (("--judge_prefixes", str(prefixes)),
                     ("--judge_candidates", str(cands))):
            result = self.launch("--mode=judge", *args, "--dry_run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--mode judge needs", result.stderr)

    def test_judge_mode_rejects_missing_input_files(self):
        result = self.launch("--mode=judge",
                             "--judge_prefixes", str(self.root / "nope.jsonl"),
                             "--judge_candidates", str(self.root / "no2.jsonl"),
                             "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no prefixes at", result.stderr)

    def test_judge_mode_env_file_names_the_real_model_not_the_alias(self):
        """`judge_model` in every judged row comes from the served alias.

        A generic `colbench-sim` there would make two different judges
        indistinguishable on disk -- the same defect already fixed for the eval
        JSON's sim identity.
        """
        prefixes, cands, _ = self._make_judge_inputs()
        # _stub_bin MUTATES self.env["PATH"], so it must run BEFORE the env is
        # copied for the subprocess. Getting that order wrong makes the test
        # shell out to the REAL sbatch and submit a job (it did, once).
        self._stub_bin()
        result = subprocess.run(
            ["bash", str(SETUP / "launch_eval_slurm.sh"), "--exp_name", "test",
             "--mode=judge", "--judge_prefixes", str(prefixes),
             "--judge_candidates", str(cands)],
            env=dict(self.env), capture_output=True, text=True)
        self.assertIn("999999", result.stdout,
                      "sbatch was not stubbed -- this test may have submitted "
                      "a REAL job: " + result.stdout)
        env_files = list((self.root / "runs").rglob("eval.env"))
        self.assertTrue(env_files, result.stderr)
        text = env_files[0].read_text()
        self.assertIn("SIM_SERVED_NAME=qwen3_235b_a22b_instruct_2507_fp8", text)
        self.assertIn("EVAL_MODE=judge", text)
        self.assertIn("JUDGE_MAX_PREFIXES=50", text)

    def _stub_bin(self):
        """A directory whose `sbatch` is a no-op, so submission is inert."""
        binder = self.root / "stubbin"
        binder.mkdir(exist_ok=True)
        stub = binder / "sbatch"
        stub.write_text("#!/bin/bash\necho 999999\n")
        stub.chmod(0o755)
        self.env["PATH"] = f"{binder}:{self.env['PATH']}"
        return binder

    def test_global_step_sweep_resolves_and_tags_output(self):
        self._make_ckpt("run_a", 250)
        self._make_ckpt("run_a", 500)
        result = self.launch("--mode=eval", "--train_exp", "run_a",
                             "--global_step", "250,500", "--val_file", str(self.val),
                             "--dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("steps=[250,500]", result.stdout)
        # The sim identity must be in the output stem, or the same checkpoint
        # evaluated under a different simulator would overwrite these results.
        self.assertIn("sim-qwen3_235b_a22b_instruct_2507_fp8", result.stdout)

    def test_env_file_steps_survive_singularity_env_file(self):
        """EVAL_STEPS must reach the container with no shell escaping in it.

        Regression for job 46266724: the list was space-separated and written with
        printf %q, but Singularity's --env-file does NO unescaping, so the container
        received the literal 'base\\ 250' and word-split it into 'base\\' and '250'.
        The step then matched neither 'base' nor an integer and the job died trying
        to merge 'global_step_base\\'. Every dry-run test passed regardless, because
        the env file is only written on a real submit.
        """
        self._make_ckpt("run_a", 250)
        self._make_ckpt("run_a", 500)
        bindir = self.root / "fakebin"
        bindir.mkdir()
        # Stub sbatch so the launcher completes its env-file write without submitting.
        (bindir / "sbatch").write_text("#!/bin/bash\necho 999999\n")
        (bindir / "sbatch").chmod(0o755)
        env = dict(self.env, PATH=f"{bindir}:{os.environ['PATH']}")
        result = subprocess.run(
            ["bash", str(SETUP / "launch_eval_slurm.sh"), "--exp_name", "envtest",
             "--mode=eval", "--train_exp", "run_a", "--global_step", "250,500",
             "--val_file", str(self.val)],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        envs = list((self.root / "runs" / "colbench_spec_eval" / "envtest").glob("*/launch/eval.env"))
        self.assertEqual(len(envs), 1, envs)
        line = [l for l in envs[0].read_text().splitlines() if l.startswith("EVAL_STEPS=")]
        self.assertEqual(line, ["EVAL_STEPS=250,500"])
        # The actual failure mode: a backslash anywhere in the value.
        self.assertNotIn("\\", line[0])


    def test_missing_step_rejected_before_submit(self):
        self._make_ckpt("run_a", 250)
        result = self.launch("--mode=eval", "--train_exp", "run_a",
                             "--global_step", "999", "--val_file", str(self.val),
                             "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no checkpoint at", result.stderr)

    def test_checkpoint_without_shards_rejected(self):
        # A checkpoint directory still being written has no shards yet.
        self._make_ckpt("run_a", 250, shards=False)
        result = self.launch("--mode=eval", "--train_exp", "run_a",
                             "--global_step", "250", "--val_file", str(self.val),
                             "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no FSDP shards", result.stderr)

    def test_global_step_and_model_path_conflict(self):
        self._make_ckpt("run_a", 250)
        result = self.launch("--mode=eval", "--train_exp", "run_a",
                             "--global_step", "250", "--model_path", str(self.weights),
                             "--val_file", str(self.val), "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)

    def test_global_step_requires_train_exp(self):
        result = self.launch("--mode=eval", "--global_step", "250",
                             "--val_file", str(self.val), "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --train_exp", result.stderr)

    def test_duplicate_step_rejected(self):
        self._make_ckpt("run_a", 250)
        result = self.launch("--mode=eval", "--train_exp", "run_a",
                             "--global_step", "250,250", "--val_file", str(self.val),
                             "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate", result.stderr)

    def test_partial_download_rejected(self):
        (self.weights / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"layer": "missing.safetensors"}}))
        result = self.launch("--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing/empty weight shard", result.stderr)

    def test_unsupported_tp_rejected(self):
        self.assertNotEqual(self.launch("--sim_tp=8", "--dry_run").returncode, 0)

    def test_h200_cpu_cap(self):
        result = self.launch("--partition=kempner_h200", "--cpus_per_task=64", "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fewer than 16 CPUs/GPU", result.stderr)

    def test_missing_eval_dataset_rejected(self):
        result = self.launch("--mode=eval", "--model_path", str(self.weights),
                             "--val_file", str(self.root / "absent"), "--dry_run")
        self.assertNotEqual(result.returncode, 0)

    def test_grounded_submission_preserves_conditioning_and_tag(self):
        self._stub_bin()
        self.env["GROUNDED_SIM"] = "True"
        result = self.launch("--mode=eval", "--model_path", str(self.weights),
                             "--val_file", str(self.val))
        self.assertEqual(result.returncode, 0, result.stderr)
        envfile = next((self.root / "runs").rglob("eval.env"))
        lines = dict(line.split("=", 1) for line in envfile.read_text().splitlines())
        self.assertEqual(lines["GROUNDED_SIM"], "True")
        self.assertTrue(lines["VAL_TAG"].endswith("-grounded"))

    def test_submission_records_resolved_config(self):
        stub = self.root / "bin"
        stub.mkdir()
        sbatch = stub / "sbatch"
        sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$ENTRYPOINT" "$SIM_SMOKE" "$@" > "$TEST_ARGS"\necho 12345\n')
        sbatch.chmod(0o755)
        self.env.update(PATH=f"{stub}:{os.environ['PATH']}", TEST_ARGS=str(self.root / "argv"),
                        TEMPERATURES="0.2 0.6")
        result = self.launch("--mode=eval", "--model_path", str(self.weights), "--val_file", str(self.val))
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = (self.root / "argv").read_text()
        self.assertIn("entrypoint_eval_colbench_slurm.sh", argv)
        self.assertIn("--nodes=2", argv)
        envfile = next((self.root / "runs").rglob("eval.env"))
        # Read the file LITERALLY. Sourcing it in bash (as this test used to) unescapes,
        # which Singularity's --env-file does not -- that mismatch is what let the
        # EVAL_STEPS corruption in job 46266724 reach a real allocation.
        lines = dict(l.split("=", 1) for l in envfile.read_text().splitlines() if "=" in l)
        self.assertNotIn("\\", envfile.read_text())
        # Space-separated in, comma-separated through the env file, decoded in the
        # entrypoint: a space cannot survive this channel at all.
        self.assertEqual(lines["TEMPERATURES"], "0.2,0.6")
        self.assertEqual(lines["GROUNDED_SIM"], "False")
        self.assertEqual(lines["OPENAI_API_KEY"], "EMPTY")


class ProbeTests(unittest.TestCase):
    def run_probe(self, content, finish="stop"):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                reply = content if "Classify" in payload['messages'][0]['content'] else "READY"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": reply},
                                                           "finish_reason": finish}]}).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                probe.chat_probe(f"http://127.0.0.1:{server.server_port}/v1", "test", str(Path(root) / "out.json"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_chat_json_success(self):
        self.run_probe('{"contains_code": false}')

    def test_invalid_json_fails(self):
        with self.assertRaises(ValueError):
            self.run_probe('not JSON')

    def test_truncation_fails(self):
        with self.assertRaises(ValueError):
            self.run_probe('{}', finish="length")


class EntrypointTests(unittest.TestCase):
    def test_eval_waits_for_missing_sentinel(self):
        # Fake only external processes; execute the real bash entrypoint and its
        # set -e behavior. In particular, sentinel_read returns 1 until ready.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "entrypoint.sh").write_text((SETUP / "entrypoint_eval_colbench_slurm.sh").read_text())
            (root / "entrypoint_common_slurm.sh").write_text('''
# The real entrypoint_common_slurm.sh sources paths.sh, so this stub must also
# provide what paths.sh defines and the entrypoint uses.
model_shorthand() { echo "${1##*/}" | tr 'A-Z' 'a-z' | tr '.-' '__'; }
verify_environment() { :; }
python3() { :; }
sentinel_read() { [ -s "$1" ] && head -n1 "$1" || return 1; }
sleep() { printf '%s\\n' http://mock/v1 > "$SIM_SENTINEL"; }
curl() { :; }
start_exec_sidecar() { :; }
wait_exec_healthy() { export CODECONTEST_EXEC_URL=http://mock-exec; }
stop_exec_sidecar() { :; }
''')
            (root / "colbench").mkdir()
            (root / "colbench/run_validate_colbench_spec.sh").write_text(
                'test "$OPENAI_BASE_URL" = http://mock/v1 && test "$CODECONTEST_ALLOW_INPROCESS" = 0 && touch "$RUN_ROOT/reached_eval"\n')
            (root / "val").write_text('test-only')
            env = dict(os.environ, RUN_ROOT=str(root), EVAL_MODE="eval", SIM_SERVER_ONLY="",
                       SIM_WEIGHTS="unused", MODEL_PATH="unused", SLURM_JOB_ID="123",
                       SIM_STARTUP_TIMEOUT="5", VAL_FILE=str(root / "val"))
            result = subprocess.run(["bash", str(root / "entrypoint.sh")], env=env,
                                    cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "reached_eval").is_file())


    def test_global_step_does_not_probe_empty_model_path(self):
        """--global_step leaves MODEL_PATH empty on purpose.

        Regression for job 46265228: the solver branch probed MODEL_PATH
        unconditionally, so in step-sweep mode it read a relative 'config.json',
        raised FileNotFoundError and killed the whole 8-GPU allocation minutes in.
        The stubbed probe below FAILS on an empty/missing directory, exactly as
        the real serving_probe.py does.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "entrypoint.sh").write_text(
                (SETUP / "entrypoint_eval_colbench_slurm.sh").read_text())
            stub = "\n".join([
                "model_shorthand() { echo \"${1##*/}\" | tr 'A-Z' 'a-z' | tr '.-' '__'; }",
                "verify_environment() { :; }",
                # Mirrors serving_probe.py: a weights check on an empty or missing
                # directory FAILS. A no-op stub would hide the very bug under test.
                "python3() { if [ \"${2:-}\" = weights ]; then [ -n \"${3:-}\" ] && [ -d \"${3}\" ]; fi; }",
                "sentinel_read() { [ -s \"$1\" ] && head -n1 \"$1\" || return 1; }",
                "sleep() { printf '%s\\n' http://mock/v1 > \"$SIM_SENTINEL\"; }",
                "curl() { :; }",
                "start_exec_sidecar() { :; }",
                "wait_exec_healthy() { export CODECONTEST_EXEC_URL=http://mock-exec; }",
                "stop_exec_sidecar() { :; }",
            ])
            (root / "entrypoint_common_slurm.sh").write_text(stub + "\n")
            (root / "colbench").mkdir()
            (root / "colbench/run_validate_colbench_spec.sh").write_text(
                'printf "%s\\n" "$OUT" >> "$RUN_ROOT/outs.txt"\n')
            (root / "val").write_text("test-only")
            base = root / "base_model"
            base.mkdir()
            env = dict(os.environ, RUN_ROOT=str(root), EVAL_MODE="eval",
                       SIM_SERVER_ONLY="", SIM_WEIGHTS="unused", MODEL_PATH="",
                       EVAL_STEPS="base", BASE_MODEL_PATH=str(base),
                       SIM_TAG="qwen3_235b", VAL_TAG="test_small",
                       SLURM_JOB_ID="123", SIM_STARTUP_TIMEOUT="5",
                       VAL_FILE=str(root / "val"))
            result = subprocess.run(["bash", str(root / "entrypoint.sh")], env=env,
                                    cwd=root, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Per-step output name, carrying the simulator identity.
            self.assertIn("stepbase_sim-qwen3_235b_test_small.json",
                          (root / "outs.txt").read_text())


if __name__ == "__main__":
    unittest.main()
