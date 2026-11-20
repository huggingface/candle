#!/usr/bin/env python3
"""Measure a Rust audio CLI process; never performs model inference in Python."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import wave


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    return value


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(observations, wall_seconds, child_status, parse_errors, input_seconds=None, parse_error_count=None):
    """Pure metric calculation, also used by deterministic fixtures."""
    first_audio = first_transcript = None
    frames = 0
    output_seconds = 0.0
    errors = list(parse_errors)[:100]
    error_count = len(parse_errors) if parse_error_count is None else parse_error_count
    for observation in observations:
        event = observation["event"]
        kind = event.get("type")
        if kind in ("audio", "audio_chunk"):
            count = event.get("frames")
            audio_format = event.get("format")
            rate = audio_format.get("sample_rate") if isinstance(audio_format, dict) else None
            if not (isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 2**64 - 1
                    and isinstance(rate, int) and not isinstance(rate, bool) and 0 < rate <= 2**32 - 1):
                error_count += 1
                if len(errors) < 100:
                    errors.append("audio event lacks a valid frame count/sample rate")
                continue
            frames += count
            output_seconds += count / rate
            if count and first_audio is None:
                first_audio = observation
        if kind in ("transcript", "transcript_segment"):
            transcript = event.get("transcript", {})
            if isinstance(transcript, dict) and isinstance(transcript.get("text"), str) and transcript["text"].strip():
                if first_transcript is None:
                    first_transcript = observation

    def latency(observation):
        if observation is None:
            return {"process_seconds": None, "cli_elapsed_seconds": None}
        elapsed = observation["event"].get("elapsed_seconds")
        return {"process_seconds": observation["received_seconds"],
                "cli_elapsed_seconds": elapsed if number(elapsed) and elapsed >= 0 else None}

    valid = child_status == 0 and error_count == 0
    denominator = input_seconds if input_seconds is not None else output_seconds
    return {
        "valid": valid,
        "errors": errors,
        "error_count": error_count,
        "wall_seconds": wall_seconds,
        "first_audio": latency(first_audio),
        "first_transcript": latency(first_transcript),
        "emitted_pcm_frames": frames,
        "emitted_pcm_seconds": output_seconds,
        "input_seconds": input_seconds,
        "rtf_basis": "input" if input_seconds is not None else "emitted_output",
        "rtf": wall_seconds / denominator if valid and denominator > 0 else None,
    }


def assess(metrics, budgets):
    values = {
        "wall_seconds": metrics["wall_seconds"],
        "first_audio_seconds": metrics["first_audio"]["process_seconds"],
        "rtf": metrics["rtf"],
    }
    return {name: {"maximum": maximum, "observed": values[name],
                   "passed": metrics["valid"] and values[name] is not None and values[name] <= maximum}
            for name, maximum in budgets.items() if maximum is not None}


def git_metadata():
    root = Path(__file__).resolve().parent.parent
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "dirty_paths": status.splitlines() if status else []}


def hardware():
    details = {"platform": platform.platform(), "machine": platform.machine(),
               "logical_cpus": os.cpu_count(), "cpu": platform.processor() or None}
    if sys.platform.startswith("linux"):
        try:
            lines = Path("/proc/cpuinfo").read_text().splitlines()
            details["cpu"] = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))
            details["memory_total"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
        except (OSError, StopIteration):
            pass
    return details


def run(args):
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("provide the Rust CLI command after --")
    binary = shutil.which(command[0])
    if binary is None:
        raise ValueError("command executable was not found")
    binary = Path(binary).resolve()
    command[0] = str(binary)
    input_seconds = args.input_seconds
    input_artifact = None
    if args.input_wav:
        with wave.open(str(args.input_wav), "rb") as audio:
            input_seconds = audio.getnframes() / audio.getframerate()
        if input_seconds <= 0:
            raise ValueError("input WAV must contain audio")
        input_artifact = {"path": str(args.input_wav.resolve()), "sha256": digest(args.input_wav)}
    binary_hash = digest(binary)
    metadata = {"git": git_metadata(), "hardware": hardware()}
    # An exclusive directory reserves all output paths before launching a child.
    args.output.mkdir(parents=False, exist_ok=False)
    time_binary = Path("/usr/bin/time")
    use_time = sys.platform.startswith("linux") and time_binary.is_file()
    launched = command
    if use_time:
        launched = [str(time_binary), "-f", "%M", "-o", str(args.output / "peak-rss-kib.txt"), "--", *command]
    parse_errors = []
    parse_error_count = 0
    peak = None
    peak_source = None
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    with (args.output / "stdout.jsonl").open("xb") as stdout, (args.output / "stderr.log").open("xb") as stderr, (args.output / "observed.jsonl").open("x", encoding="utf-8") as observed:
        try:
            process = subprocess.Popen(launched, stdout=subprocess.PIPE, stderr=stderr)
        except OSError as error:
            stderr.write(str(error).encode())
            status = None
            parse_errors.append(f"could not launch command: {error}")
            parse_error_count += 1
        else:
            # stdout is drained while the child runs; stderr goes directly to disk.
            for line_number, line in enumerate(process.stdout, 1):
                received = time.perf_counter() - started
                stdout.write(line)
                try:
                    event = json.loads(line, parse_constant=reject_constant, parse_float=finite_float)
                    if not isinstance(event, dict):
                        raise ValueError("expected a JSON object")
                except (UnicodeDecodeError, ValueError) as error:
                    parse_error_count += 1
                    if len(parse_errors) < 100:
                        parse_errors.append(f"stdout line {line_number}: {error}")
                    continue
                item = {"received_seconds": received, "event": event}
                observed.write(json.dumps(item, allow_nan=False) + "\n")
            process.stdout.close()
            if not use_time and hasattr(os, "wait4"):
                _, wait_status, usage = os.wait4(process.pid, 0)
                status = os.waitstatus_to_exitcode(wait_status)
                process.returncode = status
                if sys.platform.startswith("linux"):
                    peak = int(usage.ru_maxrss) * 1024
                    peak_source = "wait4 child ru_maxrss (Linux KiB converted to bytes)"
                elif sys.platform == "darwin":
                    peak = int(usage.ru_maxrss)
                    peak_source = "wait4 child ru_maxrss (Darwin bytes)"
            else:
                status = process.wait()
    wall = time.perf_counter() - started
    if use_time:
        try:
            last_line = (args.output / "peak-rss-kib.txt").read_text().splitlines()[-1]
            peak = int(last_line) * 1024
            peak_source = "GNU time maximum child RSS (KiB converted to bytes)"
        except (OSError, ValueError, IndexError):
            pass
    with (args.output / "observed.jsonl").open(encoding="utf-8") as stream:
        metrics = summarize((json.loads(line) for line in stream), wall, status,
                            parse_errors, input_seconds, parse_error_count)
    budgets = assess(metrics, {"wall_seconds": args.max_wall_seconds,
                              "first_audio_seconds": args.max_first_audio_seconds, "rtf": args.max_rtf})
    report = {"schema_version": 1, "started_at": started_at, "command": command,
              "cwd": str(Path.cwd()), "binary": {"path": str(binary), "sha256": binary_hash},
              **metadata, "notes": args.note, "input_artifact": input_artifact,
              "exit_status": status, "metrics": metrics, "peak_rss_bytes": peak,
              "peak_rss_source": peak_source,
              "budgets": budgets, "passed": metrics["valid"] and all(b["passed"] for b in budgets.values())}
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output / "report.json")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new run directory; never overwritten")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-wav", type=Path, help="derive ASR/pipeline RTF denominator from this PCM WAV")
    inputs.add_argument("--input-seconds", type=positive, help="explicit input duration for other input formats")
    parser.add_argument("--max-wall-seconds", type=positive)
    parser.add_argument("--max-first-audio-seconds", type=positive)
    parser.add_argument("--max-rtf", type=positive)
    parser.add_argument("--note", action="append", default=[], help="hardware/backend/cache/resident-process context; repeatable")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- /path/to/candle-audio [arguments]")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, wave.Error) as error:
        parser.exit(2, f"benchmark: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
