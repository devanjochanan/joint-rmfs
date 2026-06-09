#!/usr/bin/env python3
"""RMFS Simulator Golden Trace Baseline Harness.

Runs the simulator in a temporary copied workspace to prevent dirtying
the main checkout, capturing a stable, reproducible trace of setup and tick outputs.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

# Monitored simulator files (relative to repository root)
MONITORED_FILES = [
    "assign_order.csv",
    "pod_info.csv",
    "netlogo.state",
    "warehouse.db",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "generated_order.csv",
    "generated_pod.csv"
]


def get_git_info(repo_root):
    """Retrieves the current git commit SHA and branch name."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            text=True
        ).strip()
        return commit, branch
    except Exception:
        return "unknown", "unknown"


def get_file_info(filepath):
    """Computes file existence, size, and SHA256 digest."""
    if not os.path.exists(filepath):
        return {
            "exists": False,
            "size": 0,
            "sha256": None
        }
    try:
        size = os.path.getsize(filepath)
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return {
            "exists": True,
            "size": size,
            "sha256": h.hexdigest()
        }
    except Exception as e:
        return {
            "exists": True,
            "size": 0,
            "sha256": f"error: {str(e)}"
        }


def copy_workspace(src_dir, dest_dir):
    """Selectively copies the workspace to avoid dragging git and large run artifacts."""
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        parts = rel_path.split(os.sep) if rel_path != "." else []
        
        # Filter directories to copy/traverse
        keep_dirs = []
        for d in dirs:
            if rel_path == ".":
                if d in {".git", ".vscode", ".gemini"}:
                    continue
            elif len(parts) >= 1 and parts[0] == "data":
                if d == "runtime":
                    continue
            keep_dirs.append(d)
        dirs[:] = keep_dirs  # Prune directories in-place for recursion
        
        # Create directories in target
        for d in dirs:
            target_dir = os.path.join(dest_dir, rel_path, d) if rel_path != "." else os.path.join(dest_dir, d)
            os.makedirs(target_dir, exist_ok=True)
            
        # Copy files
        for f in files:
            src_file = os.path.join(root, f)
            target_file = os.path.join(dest_dir, rel_path, f) if rel_path != "." else os.path.join(dest_dir, f)
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(src_file, target_file)


def get_stable_digest(payload):
    """Generates a stable SHA256 signature for complex Python return payloads."""
    def sanitize(obj):
        if obj is None:
            return None
        elif isinstance(obj, (int, str, bool)):
            return obj
        elif isinstance(obj, float):
            return f"{obj:.6f}"
        elif isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, set):
            return sorted((sanitize(x) for x in obj), key=lambda x: json.dumps(x, sort_keys=True, default=str))
        elif isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        elif hasattr(obj, '__dict__'):
            return sanitize(obj.__dict__)
        elif hasattr(obj, 'x') and hasattr(obj, 'y'):
            # Specific coordinates mapping for NetLogoCoordinate classes
            return {"x": sanitize(obj.x), "y": sanitize(obj.y)}
        else:
            try:
                return str(obj)
            except Exception:
                return "unknown"

    sanitized = sanitize(payload)
    serialized = json.dumps(sanitized, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="RMFS Simulator Golden Trace Baseline Harness")
    parser.add_argument("--ticks", type=int, default=3, help="Number of ticks to simulate")
    parser.add_argument("--output", type=str, default="data/runtime/golden_trace/manual_smoke", help="Output directory path")
    args = parser.parse_args()

    # 1. Refuse negative tick counts
    if args.ticks < 0:
        parser.error("Tick count must be non-negative.")

    original_cwd = os.getcwd()
    # Script is in <root>/scripts/trace/golden_trace.py
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.isabs(args.output):
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.abspath(os.path.join(repo_root, args.output))

    # Retrieve Git commit and branch
    commit, branch = get_git_info(repo_root)

    print(f"Starting golden trace run:")
    print(f"  Branch: {branch}")
    print(f"  Commit: {commit}")
    print(f"  Ticks:  {args.ticks}")
    print(f"  Output: {output_dir}")

    temp_dir = None
    try:
        # 2. Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 3. Create temp copied workspace
        temp_dir = tempfile.mkdtemp()
        print(f"Copying workspace to temporary execution path: {temp_dir}")
        copy_workspace(repo_root, temp_dir)

        # Record pre-state of files
        pre_state = {}
        for filename in MONITORED_FILES:
            filepath = os.path.join(temp_dir, filename)
            pre_state[filename] = get_file_info(filepath)

        # Change CWD and Sys Path to temp workspace
        os.chdir(temp_dir)
        sys.path.insert(0, temp_dir)

        # 4. Import netlogo
        import netlogo

        # 5. Run setup
        print("Running simulator setup...")
        setup_result = netlogo.setup()
        
        # Check for setup error
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(f"Simulator setup failed: {setup_result}")

        setup_digest = get_stable_digest(setup_result)

        # 6. Run tick loops
        trace_rows = []
        per_tick_digests = []
        for t in range(1, args.ticks + 1):
            print(f"Executing tick {t}/{args.ticks}...")
            tick_result = netlogo.tick()
            
            # Check for tick error
            if isinstance(tick_result, str) and "An error occurred" in tick_result:
                raise RuntimeError(f"Simulator tick {t} failed: {tick_result}")
            
            tick_digest = get_stable_digest(tick_result)
            per_tick_digests.append(tick_digest)

            # Output metadata details
            output_type = type(tick_result).__name__
            output_len = len(tick_result) if hasattr(tick_result, '__len__') else None

            # Selected scalar metrics extraction
            metrics = {}
            if isinstance(tick_result, list) and len(tick_result) >= 6:
                metrics["total_energy"] = tick_result[1]
                metrics["job_queue_len"] = tick_result[2]
                metrics["stop_and_go"] = tick_result[3]
                metrics["total_turning"] = tick_result[4]

            trace_rows.append({
                "tick_index": t,
                "output_type": output_type,
                "output_length": output_len,
                "digest": tick_digest,
                "metrics": metrics
            })

        # Record post-state of files
        post_state = {}
        for filename in MONITORED_FILES:
            filepath = os.path.join(temp_dir, filename)
            post_state[filename] = get_file_info(filepath)

        # 7. Write outputs to original repo path
        print("Writing trace files...")
        
        # Manifest
        manifest = {
            "commit": commit,
            "branch": branch,
            "requested_ticks": args.ticks,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "working_directory": original_cwd,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Trace jsonl
        with open(os.path.join(output_dir, "trace.jsonl"), "w") as f:
            for row in trace_rows:
                f.write(json.dumps(row) + "\n")

        # Summary
        file_comparison = {}
        for filename in MONITORED_FILES:
            pre = pre_state[filename]
            post = post_state[filename]
            file_comparison[filename] = {
                "exists_before": pre["exists"],
                "exists_after": post["exists"],
                "size_before": pre["size"],
                "size_after": post["size"],
                "sha256_before": pre["sha256"],
                "sha256_after": post["sha256"],
                "changed": (pre["size"] != post["size"]) or (pre["sha256"] != post["sha256"])
            }

        summary = {
            "setup_output_digest": setup_digest,
            "ticks_completed": args.ticks,
            "per_tick_digests": per_tick_digests,
            "status": "success",
            "monitored_files": file_comparison
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("Trace completed successfully.")

    except Exception as e:
        print(f"Error during trace execution: {e}", file=sys.stderr)
        try:
            os.makedirs(output_dir, exist_ok=True)
            failure_summary = {
                "status": "failure",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            with open(os.path.join(output_dir, "summary.json"), "w") as f:
                json.dump(failure_summary, f, indent=2)
        except Exception as write_err:
            print(f"Failed to write failure summary: {write_err}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
