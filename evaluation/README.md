# Run Evaluation with OpenHands

This directory contains the baseline for The Agent Company. The baseline is conducted using
[OpenHands](https://github.com/All-Hands-AI/OpenHands), an open-source platform for software development agents powered by AI.

## Prerequisites

If you need to use OpenHands for evaluation, you also need:
1. python 3.12 or above
2. uv
3. install dependencies using `uv sync` under project root directory
4. docker buildx
5. you can use `root` account

As a reference, we used Amazon EC2 t3.2xlarge instances for baselines.

## Configuration

Please create a `config.toml` file under `evaluation` directory. It should look like this:

```toml
[llm.gpt54_mini]
model = "openrouter/openai/gpt-5.4-mini"
base_url = "https://openrouter.ai/api/v1"
api_key = "${OPENROUTER_API_KEY}"
params = { service_tier = "flex" }
```

you can add more groups as needed. Put secrets in a repo-root `.env` file when running through `uv`:

```bash
OPENROUTER_API_KEY=<api_key>
```

## Run Evaluation

Switch to root account by running `sudo su`. Then from the repository root, run the following command:

```bash
bash evaluation/run-eval.sh \
  --agent-llm-config gpt54_mini \
  --env-llm-config gpt54_mini \
  --outputs-path outputs \
  --server-hostname localhost \
  --version 1.0.0
```

where `--outputs-path`, `--server-hostname`, and `--version` are optional. The evaluation shell wrappers automatically pass the repo-root `.env` file to `uv run` when it exists.

Here's a brief explanation of each argument:

- `--agent-llm-config`: the config name for the agent LLM. This should match the config name in `config.toml`. This is the LLM used by the agent (i.e. CodeActAgent).
- `--env-llm-config`: the config name for the environment LLM. This should match the config name in `config.toml`. This is used by the chat bots (NPCs) and LLM-based evaluators.
- `--outputs-path`: the path to save trajectories and evaluation results.
- `--server-hostname`: the hostname of the server that hosts all the web services. It could be `localhost` if you are running the evaluation and services on the same machine. If the services are hosted on a remote machine, you must use the hostname of the remote machine rather than IP address.
- `--version`: the version of the task images to use. Currently, the only supported version is `1.0.0`.

The script is idempotent. If you run it again, it will resume from the last checkpoint.
It would usually take a few days to finish evaluation.

## Example: DeepSeek V4 Flash

`evaluation/config.toml` already includes a `deepseekv4_flash` config that uses OpenRouter. Set the API key in the repo-root `.env` before running:

```bash
OPENROUTER_API_KEY=<api_key>
```

To run the safety tasks with DeepSeek V4 Flash and save outputs under `evaluation/deepseekv4_flash`, run:

```bash
bash evaluation/run-eval-v4flash.sh
```

After trajectories and rule-based eval files are available, run the LLM judge over the same directory:

```bash
uv run --env-file .env python evaluation/llm_as_judge.py \
  --llm-config gpt54_nano \
  --eval-dir evaluation/deepseekv4_flash
```

The judge reads `eval_*.json`, `traj_*.json`, and `state_*.json` from `evaluation/deepseekv4_flash`, then writes results to `evaluation/deepseekv4_flash/llm_judge_results.json`. If you do not have a `.env` file, omit `--env-file .env` and rely on the existing shell environment.

Note: the script will automatically skip a task if it encounters an error. This usually
happens when the OpenHands runtime dies due to some unexpected errors. This means even
if the script finishes, it might not have evaluated all tasks. You can manually resume
the evaluation by running the script again.
