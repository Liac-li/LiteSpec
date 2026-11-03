"""CPU subprocess tests for dataset-per-GPU evaluation orchestration."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dlite_loop as loop


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def execute_queue(self, jobs, gpus, **kwargs):
        with redirect_stdout(io.StringIO()):
            return loop.run_evaluation_jobs(jobs, gpus, self.root, **kwargs)

    def test_two_gpus_overlap_and_each_device_is_exclusive(self):
        program = '''
import json,os,sys,time
from pathlib import Path
root,name=Path(sys.argv[1]),sys.argv[2]
gpu=os.environ['CUDA_VISIBLE_DEVICES']
lock=root/('gpu_'+gpu+'.lock')
with lock.open('x'): pass
(root/(name+'.started')).write_text(gpu)
if name in ('job0','job1'):
 deadline=time.monotonic()+8
 while not all((root/(n+'.started')).exists() for n in ('job0','job1')):
  if time.monotonic()>deadline: raise RuntimeError('workers failed to overlap')
  time.sleep(.02)
time.sleep(.05)
lock.unlink()
print(json.dumps({'dataset':name,'gpu':gpu}))
'''
        jobs = [(f"job{i}", [sys.executable, "-c", program, self.root, f"job{i}"]) for i in range(4)]
        states = self.execute_queue(jobs, ["4", "7"])
        self.assertTrue(all(row["status"] == "completed" for row in states))
        self.assertEqual([row["gpu"] for row in states[:2]], ["4", "7"])
        for row in states:
            record = json.loads((self.root/f"{row['dataset']}.log").read_text())
            self.assertEqual(record["gpu"], row["gpu"])
            self.assertEqual((self.root/f"{row['dataset']}.started").read_text(), row["gpu"])
        self.assertFalse(list(self.root.glob("*.lock")))
        self.assertEqual(json.loads((self.root/"queue_status.json").read_text())["jobs"], states)

    def test_single_gpu_runs_serially(self):
        program = '''
from pathlib import Path
import sys
root,index=Path(sys.argv[1]),int(sys.argv[2])
if index: assert (root/str(index-1)).is_file()
(root/str(index)).write_text('completed')
'''
        states = self.execute_queue([(str(i), [sys.executable, "-c", program, self.root, str(i)])
                                     for i in range(3)], ["6"])
        self.assertTrue(all(row["gpu"] == "6" and row["status"] == "completed" for row in states))

    def test_child_failure_cancels_peer_and_leaves_pending_work(self):
        fail = '''
from pathlib import Path
import sys,time
ready=Path(sys.argv[1])/'peer_ready'
deadline=time.monotonic()+8
while not ready.exists():
 if time.monotonic()>deadline: raise RuntimeError('peer not started')
 time.sleep(.02)
sys.exit(3)
'''
        peer = "from pathlib import Path;import sys,time;(Path(sys.argv[1])/'peer_ready').write_text('ready');time.sleep(30)"
        jobs = [("fail", [sys.executable, "-c", fail, self.root]),
                ("peer", [sys.executable, "-c", peer, self.root]),
                ("pending", [sys.executable, "-c", "raise RuntimeError('must not launch')"])]
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.execute_queue(jobs, ["0", "1"])
        self.assertEqual(error.exception.returncode, 3)
        states = json.loads((self.root/"queue_status.json").read_text())["jobs"]
        self.assertEqual([s["status"] for s in states], ["failed", "cancelled", "pending"])
        self.assertIsNotNone(states[1]["returncode"])
        self.assertFalse((self.root/"pending.log").exists())

    def test_interrupt_during_launch_reaps_already_running_child(self):
        real_popen = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            if children:
                raise KeyboardInterrupt()
            process = real_popen(*args, **kwargs)
            children.append(process)
            return process
        jobs = [(name, [sys.executable, "-c", "import time;time.sleep(30)"]) for name in ("one", "two")]
        with patch.object(loop.subprocess, "Popen", side_effect=launch), self.assertRaises(KeyboardInterrupt):
            self.execute_queue(jobs, ["0", "1"])
        self.assertIsNotNone(children[0].poll())
        states = json.loads((self.root/"queue_status.json").read_text())["jobs"]
        self.assertEqual(states[0]["status"], "cancelled")
        self.assertEqual(states[1]["status"], "failed")

    def test_spawn_failure_is_recorded(self):
        with self.assertRaises(FileNotFoundError):
            self.execute_queue([("bad", [self.root/"nonexistent_python"])], ["0"])
        state = json.loads((self.root/"queue_status.json").read_text())["jobs"][0]
        self.assertEqual(state["status"], "failed")
        self.assertIn("error", state)

    def prepare_evaluation(self):
        run = self.root/"trained"
        run.mkdir()
        (run/"experiment.json").write_text(json.dumps(dict(target_model="tiny")))
        for label, (layers, loops) in zip(loop.MODELS, ((5, 1), (1, 5), (1, 3))):
            checkpoint = run/label/"last"
            checkpoint.mkdir(parents=True)
            (checkpoint/"config.json").write_text(json.dumps(dict(num_hidden_layers=layers,
                dlite_config=dict(num_loops=loops))))
        text = self.root/"gsm8k.jsonl"
        text.write_text('{"turns":["question"]}\n')
        data = self.root/"data"
        data.mkdir()
        for name in ("charxiv_val.jsonl", "flickr30k_test.jsonl", "textvqa_test.jsonl",
                     "hr_bench_4k.jsonl", "hr_bench_8k.jsonl"):
            (data/name).write_text('{}\n')
        return ["eval", "--run-dir", str(run), "--text-data", str(text), "--data-dir", str(data)]

    def test_multigpu_dry_run_preserves_all_exits_and_batches(self):
        arguments = self.prepare_evaluation()
        with redirect_stdout(io.StringIO()):
            loop.main([*arguments, "--gpus", "0,1,2,3,4,5", "--dry-run"])
        result = self.root/"trained/evaluation"
        status = json.loads((result/"queue_status.json").read_text())
        self.assertEqual(status["gpus"], ["0", "1", "2", "3", "4", "5"])
        self.assertEqual([j["gpu"] for j in status["jobs"]], status["gpus"])
        self.assertEqual(len(status["jobs"]), 6)
        for job in status["jobs"]:
            command = job["command"]
            self.assertEqual(command.count("--checkpoint"), 9)
            self.assertEqual(command.count("--checkpoint-loops"), 9)
            start = command.index("--batch-sizes")
            self.assertEqual(command[start+1:start+3], ["1", "32"])
            self.assertEqual(command[command.index("--output-dir")+1], str(result/job["dataset"]))
        settings = json.loads((result/"experiment.json").read_text())
        self.assertEqual(len(settings["exits"]), 9)

    def test_old_single_gpu_option_remains_valid(self):
        arguments = self.prepare_evaluation()
        with redirect_stdout(io.StringIO()):
            loop.main([*arguments, "--gpu", "7", "--datasets", "gsm8k", "--dry-run"])
        status = json.loads((self.root/"trained/evaluation/queue_status.json").read_text())
        self.assertEqual(status["gpus"], ["7"])

    def test_invalid_gpu_lists_and_conflicting_options(self):
        for value in ("", "0,0", "0,00", "0,,1", "-1", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loop.evaluation_gpu_ids(value)
        for options in (["--gpu", "0", "--gpus", "1,2"], ["--gpu", "0,1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                loop.main(["eval", "--run-dir", "/not/read", *options])

    def test_report_ignores_incomplete_live_or_failed_summaries(self):
        completed = self.root/"gsm8k"
        completed.mkdir()
        summary = dict(model="ar", batch_size=1, mean_accepted_draft_tokens=None,
            mean_committed_length=None, output_tokens_per_second=100, raw_throughput_speedup_vs_ar=1,
            matched=1, requests=1, peak_allocated_gib=None)
        (completed/"summary.json").write_text(json.dumps([summary]))
        broken = self.root/"textvqa"
        broken.mkdir()
        (broken/"summary.json").write_text('{"half_written":')
        (self.root/"queue_status.json").write_text(json.dumps(dict(dry_run=False, jobs=[
            dict(dataset="gsm8k", status="completed"), dict(dataset="textvqa", status="failed")])))
        with redirect_stdout(io.StringIO()):
            loop.report(self.root)
        rows = json.loads((self.root/"comparison.json").read_text())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset"], "gsm8k")

    def test_evaluator_preserves_completed_report_when_another_dataset_fails(self):
        arguments = self.prepare_evaluation()
        def failed_queue(jobs, gpus, result_dir, **kwargs):
            self.assertEqual(gpus, ["4", "5"])
            self.assertEqual([name for name, _ in jobs], ["gsm8k", "textvqa"])
            complete = result_dir/"gsm8k"
            complete.mkdir()
            row = dict(model="ar", batch_size=1, mean_accepted_draft_tokens=None,
                mean_committed_length=None, output_tokens_per_second=100,
                raw_throughput_speedup_vs_ar=1, matched=1, requests=1, peak_allocated_gib=None)
            (complete/"summary.json").write_text(json.dumps([row]))
            failed = result_dir/"textvqa"
            failed.mkdir()
            (failed/"summary.json").write_text('{"partial":')
            (result_dir/"queue_status.json").write_text(json.dumps(dict(dry_run=False, jobs=[
                dict(dataset="gsm8k", status="completed"), dict(dataset="textvqa", status="failed")])))
            raise subprocess.CalledProcessError(3, ["test_worker"])
        with patch.object(loop, "run_evaluation_jobs", side_effect=failed_queue), redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            loop.main([*arguments, "--datasets", "gsm8k", "textvqa", "--gpus", "4,5"])
        rows = json.loads((self.root/"trained/evaluation/comparison.json").read_text())
        self.assertEqual([row["dataset"] for row in rows], ["gsm8k"])


if __name__ == "__main__":
    unittest.main()
