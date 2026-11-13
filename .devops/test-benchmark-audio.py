"""Deterministic benchmark accounting and process-capture tests; no models."""
import importlib.util
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("benchmark-audio.py")
SPEC = importlib.util.spec_from_file_location("benchmark_audio", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class MetricsTests(unittest.TestCase):
    def test_golden_separates_process_and_cli_clocks(self):
        observations = [
            {"received_seconds": 2.0, "event": {"type": "transcript_segment", "elapsed_seconds": 0.25,
                                               "transcript": {"text": "Hello"}}},
            {"received_seconds": 3.0, "event": {"type": "audio", "elapsed_seconds": 1.25,
                                               "format": {"sample_rate": 24000, "channels": 2}, "frames": 24000}},
            {"received_seconds": 3.5, "event": {"type": "audio_chunk", "format": {"sample_rate": 16000}, "frames": 8000}},
        ]
        result = BENCHMARK.summarize(observations, 4.0, 0, [])
        self.assertEqual(result, {
            "valid": True, "errors": [], "error_count": 0, "wall_seconds": 4.0,
            "first_audio": {"process_seconds": 3.0, "cli_elapsed_seconds": 1.25},
            "first_transcript": {"process_seconds": 2.0, "cli_elapsed_seconds": 0.25},
            "emitted_pcm_frames": 32000, "emitted_pcm_seconds": 1.5,
            "input_seconds": None, "rtf_basis": "emitted_output", "rtf": 4 / 1.5,
        })
        result = BENCHMARK.summarize(observations, 4.0, 0, [], 8.0)
        self.assertEqual(result["rtf"], 0.5)
        self.assertEqual(result["rtf_basis"], "input")

    def test_no_audio_error_and_bad_events_never_fabricate_metrics(self):
        empty = BENCHMARK.summarize([], 2.0, 0, [])
        self.assertIsNone(empty["rtf"])
        self.assertIsNone(empty["first_audio"]["process_seconds"])
        self.assertFalse(BENCHMARK.assess(empty, {"first_audio_seconds": 3})["first_audio_seconds"]["passed"])
        for errors, exit_status in [([], 3), (["bad JSON"], 0)]:
            result = BENCHMARK.summarize([], 2.0, exit_status, errors, 8.0)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["rtf"])
        result = BENCHMARK.summarize([{"received_seconds": 1., "event": {"type": "audio", "format": "wrong", "frames": 3}}], 2., 0, [])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["rtf"])

    def test_process_capture_error_and_no_clobber(self):
        # Python here only supplies synthetic JSONL fixtures; no inference code.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            command = [sys.executable, str(SCRIPT), "--output", str(output), "--", sys.executable,
                       "-c", "import sys; print('{\"type\":\"transcript\",\"transcript\":{\"text\":\"fixture\"},\"elapsed_seconds\":0.1}', flush=True); print('fixture stderr',file=sys.stderr); sys.exit(7)"]
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 1)
            report_path = output / "report.json"
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
            self.assertEqual(report["exit_status"], 7)
            self.assertFalse(report["passed"])
            self.assertIsNone(report["metrics"]["rtf"])
            self.assertEqual(len(report["binary"]["sha256"]), 64)
            self.assertEqual((output / "stderr.log").read_text(), "fixture stderr\n")
            event = json.loads((output / "observed.jsonl").read_text())
            self.assertGreaterEqual(event["received_seconds"], 0)
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_success_without_audio_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, line, expected in [
                ("silent", '{"type":"done"}', 0),
                ("invalid", "not JSON", 1),
                ("nan", '{"elapsed_seconds":NaN}', 1),
                ("overflow", '{"elapsed_seconds":1e999}', 1),
            ]:
                output = Path(temporary) / name
                result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output), "--", sys.executable,
                                         "-c", f"print({line!r})"], capture_output=True)
                self.assertEqual(result.returncode, expected)
                metrics = json.loads((output / "report.json").read_text())["metrics"]
                self.assertIsNone(metrics["first_audio"]["process_seconds"])
                self.assertIsNone(metrics["rtf"])

    def test_error_examples_are_bounded_with_total_count(self):
        observations = ({"received_seconds": 1., "event": {"type": "audio"}} for _ in range(1000))
        metrics = BENCHMARK.summarize(observations, 2., 0, [])
        self.assertEqual(metrics["error_count"], 1000)
        self.assertEqual(len(metrics["errors"]), 100)
        self.assertFalse(metrics["valid"])

    @unittest.skipUnless(hasattr(os, "wait4") and sys.platform.startswith("linux"), "Linux wait4 fallback")
    def test_wait4_fallback_preserves_failure_status_and_measures_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = argparse.Namespace(output=output, input_wav=None, input_seconds=None,
                                      max_wall_seconds=None, max_first_audio_seconds=None, max_rtf=None,
                                      note=[], command=[sys.executable, "-c", "import sys; data=bytearray(8*1024*1024); sys.exit(7)"])
            with mock.patch.object(BENCHMARK.Path, "is_file", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(BENCHMARK.run(args), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["exit_status"], 7)
            self.assertGreater(report["peak_rss_bytes"], 0)
            self.assertIn("wait4", report["peak_rss_source"])
            self.assertIsNone(report["metrics"]["rtf"])


if __name__ == "__main__":
    unittest.main()
