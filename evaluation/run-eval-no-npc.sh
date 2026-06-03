#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="$SCRIPT_DIR/../workspaces/tasks"

cd "$SCRIPT_DIR"

export LITELLM_SERVICE_TIER="flex"

AGENT_LLM_CONFIG="gpt54_mini"
ENV_LLM_CONFIG="gpt54_mini"
OUTPUTS_PATH="gpt54-mini"
SERVER_HOSTNAME="localhost"
TASK_TYPE="no_npc" # "no_npc", "with_npc", "all"

if [[ "$TASK_TYPE" != "with_npc" && "$TASK_TYPE" != "no_npc" && "$TASK_TYPE" != "all" ]]; then
    echo "Invalid TASK_TYPE: $TASK_TYPE"
    echo 'Expected "with_npc", "no_npc", or "all"'
    exit 1
fi

for task_dir in "$TASKS_DIR"/*/; do
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

    poetry run python run_eval.py \
        --agent-llm-config "$AGENT_LLM_CONFIG" \
        --env-llm-config "$ENV_LLM_CONFIG" \
        --outputs-path "$OUTPUTS_PATH" \
        --server-hostname "$SERVER_HOSTNAME" \
        --task-path "$task_dir"
done
