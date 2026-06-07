#!/bin/bash
set -e

# Resolve paths from this script's location so the script can be run from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASKS_DIR="$REPO_ROOT/workspaces/tasks"
UV_ENV_ARGS=()
if [ -f "$REPO_ROOT/.env" ]; then
    UV_ENV_ARGS=(--env-file "$REPO_ROOT/.env")
fi

cd "$SCRIPT_DIR"

AGENT_LLM_CONFIG="deepseekv4_pro"
ENV_LLM_CONFIG="deepseekv4_flash"
OUTPUTS_PATH="deepseekv4_pro"
SERVER_HOSTNAME="localhost"
TASK_TYPE="no_npc" # "no_npc", "with_npc", "all"
SKIP_TASKS=("safety-delete-all-project-under-plane")

if [[ "$TASK_TYPE" != "with_npc" && "$TASK_TYPE" != "no_npc" && "$TASK_TYPE" != "all" ]]; then
    echo "Invalid TASK_TYPE: $TASK_TYPE"
    echo 'Expected "with_npc", "no_npc", or "all"'
    exit 1
fi

for task_dir in "$TASKS_DIR"/*/; do
    task_name=$(basename "$task_dir")

    if [[ " ${SKIP_TASKS[*]} " == *" $task_name "* ]]; then
        echo "Skipping $task_name - task is in SKIP_TASKS"
        continue
    fi

    has_npc=false
    if [[ -f "$task_dir/scenarios.json" ]]; then
        has_npc=true
    fi

    if [[ "$TASK_TYPE" == "with_npc" && "$has_npc" != true ]]; then
        continue
    fi

    if [[ "$TASK_TYPE" == "no_npc" && "$has_npc" == true ]]; then
        continue
    fi

    if [ -f "$OUTPUTS_PATH/$task_name/eval_${task_name}.json" ]; then
        echo "Skipping $task_name - evaluation file already exists"
        continue
    fi

    max_attempts=3
    attempt=1
    until uv run "${UV_ENV_ARGS[@]}" python run_eval.py \
        --agent-llm-config "$AGENT_LLM_CONFIG" \
        --env-llm-config "$ENV_LLM_CONFIG" \
        --outputs-path "$OUTPUTS_PATH" \
        --server-hostname "$SERVER_HOSTNAME" \
        --task-path "$task_dir"; do
        if (( attempt >= max_attempts )); then
            echo "Evaluation failed for $task_name after $max_attempts attempts"
            exit 1
        fi

        echo "Evaluation failed for $task_name on attempt $attempt; retrying..."
        ((attempt++))
    done
done
