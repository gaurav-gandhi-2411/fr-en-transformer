#!/usr/bin/env bash
# Run a command while sampling nvidia-smi every N seconds into a CSV (spec §1 pilot telemetry:
# clocks, temperature, power, utilization, memory and clock-event (throttle) reasons). The
# sampler is stopped when the command exits; the command's exit code is propagated.
#
# One nvidia-smi process per sample, appended: `nvidia-smi -l N -f FILE` buffers its output and
# killing it at the end lost every sample of a 15-minute pilot.
#
# Usage: scripts/run_with_gpu_monitor.sh <csv_out> <interval_s> -- <command...>
set -euo pipefail
out="$1"; interval="$2"; shift 2
[ "$1" = "--" ] && shift
mkdir -p "$(dirname "$out")"
fields=timestamp,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,memory.used,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,pstate
nvidia-smi --query-gpu="$fields" --format=csv > "$out"
(
  while true; do
    nvidia-smi --query-gpu="$fields" --format=csv,noheader >> "$out"
    sleep "$interval"
  done
) &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
"$@"
