#!/bin/bash
# =============================================================================
# Live Dashboard — monitors elastic serving cluster + eval job progress
#
# Usage:
#   bash scripts/dashboard.sh
#   bash scripts/dashboard.sh results/webshaper_full   # custom output dir
# =============================================================================

SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:8780}"
OUTDIR="${1:-results/webshaper_full}"
TRAJ_FILE="${OUTDIR}/trajectories.jsonl"

watch -n 5 -t bash -c '
SCHEDULER_URL="'"$SCHEDULER_URL"'"
TRAJ_FILE="'"$TRAJ_FILE"'"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             Elastic Inference — Live Dashboard              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date)"
echo ""

# ---- Cluster Status ----
STATUS=$(curl -s --connect-timeout 2 "${SCHEDULER_URL}/cluster_status" 2>/dev/null)
if [ -z "$STATUS" ]; then
    echo "  ⚠  Scheduler unreachable at ${SCHEDULER_URL}"
else
    MODEL=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"model\",\"?\").split(\"/\")[-1])" 2>/dev/null)
    READY=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"ready_workers\",0))" 2>/dev/null)
    LOADING=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"loading_workers\",0))" 2>/dev/null)
    PENDING=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"pending_slurm_jobs\",0))" 2>/dev/null)
    NODES=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"total_nodes_active\",0))" 2>/dev/null)

    echo "  ── Cluster ──────────────────────────────────────────────"
    echo "  Model:   $MODEL"
    echo "  Workers: $READY ready, $LOADING loading, $PENDING pending"
    echo ""

    # vLLM metrics per worker
    WORKERS=$(echo "$STATUS" | python3 -c "
import sys,json
for w in json.load(sys.stdin)[\"workers\"]:
    if w[\"status\"]==\"READY\":
        print(w[\"ip_address\"],w[\"port\"],w[\"hostname\"])
" 2>/dev/null)

    echo "  ── GPU Workers ──────────────────────────────────────────"
    printf "  %-20s %8s %8s %10s\n" "Host" "Running" "Waiting" "KV Cache"
    echo "  ─────────────────────────────────────────────────────────"
    while IFS=" " read -r ip port host; do
        [ -z "$ip" ] && continue
        METRICS=$(curl -s --connect-timeout 2 "http://${ip}:${port}/metrics" 2>/dev/null)
        if [ -n "$METRICS" ]; then
            RUN=$(echo "$METRICS" | grep "vllm:num_requests_running{" | grep -v "^#" | awk "{print \$2}" | head -1)
            WAIT=$(echo "$METRICS" | grep "vllm:num_requests_waiting{" | grep -v "^#" | awk "{print \$2}" | head -1)
            KV=$(echo "$METRICS" | grep "vllm:kv_cache_usage_perc{" | grep -v "^#" | awk "{printf \"%.1f%%\", \$2*100}" | head -1)
            printf "  %-20s %8s %8s %10s\n" "$host" "${RUN:-?}" "${WAIT:-?}" "${KV:-?}"
        else
            printf "  %-20s %8s %8s %10s\n" "$host" "?" "?" "?"
        fi
    done <<< "$WORKERS"
fi

echo ""

# ---- Eval Progress ----
if [ -f "$TRAJ_FILE" ]; then
    echo "  ── Eval Progress ($TRAJ_FILE) ──"

    python3 -c "
import json, sys, time
from collections import Counter
from pathlib import Path

traj_file = \"$TRAJ_FILE\"
trajs = []
with open(traj_file) as f:
    for line in f:
        try: trajs.append(json.loads(line))
        except: pass

n = len(trajs)
qids = set(t[\"qid\"] for t in trajs)
n_q = len(qids)

# Tool stats
tool_counts = Counter()
total_tools = 0
for t in trajs:
    for tc in t.get(\"tool_calls\", []):
        tool_counts[tc[\"tool\"]] += 1
        total_tools += 1

# Time stats
times = [t.get(\"latency_s\", 0) for t in trajs if t.get(\"latency_s\", 0) > 0]
avg_time = sum(times) / max(len(times), 1)

# Status
statuses = Counter(t.get(\"status\", \"?\") for t in trajs)

# Rate: trajs per minute based on file mtime
mtime = Path(traj_file).stat().st_mtime
# Estimate from first and last trajectory timestamps
if len(times) >= 2:
    total_elapsed = sum(times)  # rough
    rate = n / (total_elapsed / max(len(set(t.get(\"traj_idx\",0) for t in trajs)),1)) * 60
else:
    rate = 0

# Boxed answers
n_boxed = sum(1 for t in trajs if t.get(\"boxed_answer\"))

# Quick accuracy estimate (exact substring match)
correct = 0
by_qid = {}
for t in trajs:
    qid = t[\"qid\"]
    if qid not in by_qid:
        by_qid[qid] = {\"ref\": t.get(\"reference_answer\",\"\"), \"trajs\": []}
    by_qid[qid][\"trajs\"].append(t)

pass_count = 0
for qid, data in by_qid.items():
    ref = data[\"ref\"].strip().lower().replace(\",\",\"\").replace(\" \",\"\")
    any_match = False
    for t in data[\"trajs\"]:
        ans = (t.get(\"boxed_answer\") or t.get(\"answer\",\"\")).strip().lower().replace(\",\",\"\").replace(\" \",\"\")
        if ref and ans and (ref in ans or ans in ref):
            correct += 1
            any_match = True
    if any_match:
        pass_count += 1

print(f\"  Trajectories:  {n} / 2000  ({n/20:.0f}%)\")
print(f\"  Questions:     {n_q} / 500\")
print(f\"  Avg time/traj: {avg_time:.0f}s\")
print(f\"  Boxed answers: {n_boxed}/{n} ({n_boxed/max(n,1)*100:.0f}%)\")
print(f\"  Status:        {dict(statuses)}\")
print()
print(f\"  ── Tool Calls ({total_tools} total, {total_tools/max(n,1):.1f}/traj) ──\")
for tool, cnt in tool_counts.most_common():
    bar = \"█\" * min(int(cnt/max(total_tools,1)*40), 40)
    print(f\"  {tool:30s} {cnt:5d} ({cnt/max(total_tools,1)*100:4.0f}%) {bar}\")
print()
print(f\"  ── Quick Accuracy (substring match, not LLM judge) ──\")
print(f\"  Traj accuracy:  {correct}/{n} ({correct/max(n,1)*100:.1f}%)\")
print(f\"  pass@k (est):   {pass_count}/{n_q} ({pass_count/max(n_q,1)*100:.1f}%)\")
" 2>/dev/null
else
    echo "  No eval results yet at $TRAJ_FILE"
fi

echo ""
echo "  Press Ctrl+C to exit"
'

