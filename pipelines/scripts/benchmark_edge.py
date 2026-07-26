import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time

import psutil
from codecarbon import EmissionsTracker


class SystemMonitor:
    def __init__(self):
        self.running = False
        self.cpu_measurements = []
        self.ram_measurements = []
        self.thread = None

    def start(self):
        self.running = True
        # Prime the cpu_percent
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def _monitor(self):
        while self.running:
            self.cpu_measurements.append(psutil.cpu_percent(interval=None))
            self.ram_measurements.append(psutil.virtual_memory().percent)
            time.sleep(0.1)

    def stop(self):
        if not self.running:
            return {}
        self.running = False
        if self.thread:
            self.thread.join()
        return {
            "avg_cpu": sum(self.cpu_measurements) / len(self.cpu_measurements) if self.cpu_measurements else 0,
            "max_cpu": max(self.cpu_measurements) if self.cpu_measurements else 0,
            "avg_ram": sum(self.ram_measurements) / len(self.ram_measurements) if self.ram_measurements else 0,
            "max_ram": max(self.ram_measurements) if self.ram_measurements else 0,
        }


def parse_duration(log_output):
    """
    Parses the output of the docker container to find the duration logged by UniversalTimer.
    """
    # 1. Try parsing structured JSON logs first
    for line in log_output.splitlines():
        try:
            data = json.loads(line.strip())
            msg = data.get("message", "")
            if "END" in msg and "[main]" in msg:
                match = re.search(r":\s+([\d\.]+)s", msg)
                if match:
                    return float(match.group(1))
        except json.JSONDecodeError:
            continue

    # 2. Fallback to standard plain text regex if logging is not JSON
    match = re.search(r"END\s+\[main\]\s+\[[a-f0-9]+\]\s+:\s+([\d\.]+)s", log_output)
    if match:
        return float(match.group(1))

    return None


def main():
    parser = argparse.ArgumentParser(description="Run Edge Benchmark")
    parser.add_argument("--image", type=str, default="tuh-edge-inference:amd64", help="The docker image to run")
    parser.add_argument("--model_type", type=str, required=True, choices=["tflite", "onnx"], help="Model type to test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file inside the container")
    parser.add_argument("--runs", type=int, default=10, help="Number of times to run the benchmark")
    args = parser.parse_args()

    durations = []
    print(f"Benchmarking {args.model_type.upper()} ({args.runs} runs)...")
    print(f"Using image: {args.image}")
    print("-" * 50)

    command = ["docker", "run", "--rm", args.image, "--model_type", args.model_type, "--model_path", args.model_path]

    tracker = EmissionsTracker(project_name=f"benchmark_{args.model_type}", log_level="error")
    tracker.start()

    monitor = SystemMonitor()
    monitor.start()

    for i in range(1, args.runs + 1):
        try:
            # Capture both stdout and stderr since logs might go to either
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout + result.stderr
            duration = parse_duration(output)

            if duration is not None:
                durations.append(duration)
                print(f"  Run {i:02d}: {duration:.4f} seconds")
            else:
                print(f"  Run {i:02d}: Failed to parse duration. Output snippet:")
                # Print the last 5 lines of output to help debug why parsing failed
                print("\n".join(output.splitlines()[-5:]))

        except subprocess.CalledProcessError as e:
            print(f"  Run {i:02d}: FAILED")
            print(e.stderr)

    if durations:
        sys_metrics = monitor.stop()
        emissions_data = None
        if tracker:
            emissions_data = tracker.stop()

        avg = statistics.mean(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
        print("-" * 50)
        print(f"RESULTS FOR {args.model_type.upper()}")
        print("-" * 50)
        print(f"Total Successful Runs: {len(durations)}")
        print(f"Average Time: {avg:.4f} seconds")
        print(f"Std Dev:      {std_dev:.4f} seconds")
        print(f"Min Time:     {min(durations):.4f} seconds")
        print(f"Max Time:     {max(durations):.4f} seconds")

        if sys_metrics:
            print("-" * 50)
            print("SYSTEM METRICS (Host)")
            print(f"Avg CPU Usage: {sys_metrics['avg_cpu']:.2f}%")
            print(f"Max CPU Usage: {sys_metrics['max_cpu']:.2f}%")
            print(f"Avg RAM Usage: {sys_metrics['avg_ram']:.2f}%")
            print(f"Max RAM Usage: {sys_metrics['max_ram']:.2f}%")

        if emissions_data is not None:
            print("-" * 50)
            print("CODECARBON EMISSIONS")
            print(f"Total Emissions: {emissions_data:.8f} kgCO2eq")
            if hasattr(tracker, "_total_energy"):
                energy = tracker._total_energy.kWh if hasattr(tracker._total_energy, "kWh") else float(tracker._total_energy)
                print(f"Total Energy:    {energy:.8f} kWh")

        print("-" * 50)


if __name__ == "__main__":
    main()
