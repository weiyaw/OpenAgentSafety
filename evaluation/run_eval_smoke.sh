#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

export LITELLM_SERVICE_TIER="flex"

AGENT_LLM_CONFIG="gpt54_mini"
ENV_LLM_CONFIG="gpt54_mini"
OUTPUTS_PATH="smoke-gpt54"
SERVER_HOSTNAME="localhost"

TASKS=(
    safety-abusive-apology
    # safety-adjustments
    # safety-alignment
)

for task in "${TASKS[@]}"; do
    poetry run python run_eval.py \
        --agent-llm-config "$AGENT_LLM_CONFIG" \
        --env-llm-config "$ENV_LLM_CONFIG" \
        --outputs-path "$OUTPUTS_PATH" \
        --server-hostname "$SERVER_HOSTNAME" \
        --task-path "../workspaces/tasks/$task"
done
