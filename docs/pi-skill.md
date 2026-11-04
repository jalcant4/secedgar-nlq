# Pi skill: `/secedgar`

This repository ships a Pi skill at `skills/secedgar/SKILL.md` and a prompt
template at `skills/prompts/secedgar.md` that exposes it as the
`/secedgar "<question>"` command. The skill preserves the direct CLI
invocation — `python ~/secedgar-nlq/src/secedgar_agent.py "<verbatim
user question>"` — as the execution path.

## Install

```bash
# Skill (gives /skill:secedgar and model-invoked routing)
ln -sfn ~/secedgar-nlq/skills/secedgar ~/.pi/agent/skills/secedgar

# Prompt template (gives the exact /secedgar command)
mkdir -p ~/.pi/agent/prompts
cp ~/secedgar-nlq/skills/prompts/secedgar.md ~/.pi/agent/prompts/
```

Run `/reload` in an active Pi session after installing.

## Usage

```text
/secedgar What is CoreWeave's net-income-to-debt ratio for its latest annual period?
```

The template routes the question to the `secedgar` skill, which runs the
agent CLI with a generously bounded wall-clock timeout, preserves
stdout/stderr and the exit code, and enforces the evidence rules (provisional
vs. verified results, accession citations, no relabeling).

The skill must not edit the repository or mutate the agent's persistent REPL
state from another process. `inspect()` is for the agent's own
model-generated code; if a run fails, ask the agent to inspect
`mode="last_error"` or `mode="validation"`. The agent's result validator
remains authoritative for acceptance; inspection output is diagnostic, not
financial evidence.
