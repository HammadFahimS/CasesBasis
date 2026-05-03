# CasesBasis Project Memory

## Core Philosophy
- This is a clinical simulation system with:
  - generator
  - patient agent
  - investigation system
  - evaluator
  - simulation runner
  - simulation auditor
- System must remain deterministic where possible
- LLM is used only where necessary

## Development Workflow (MANDATORY)
- Always run:
  - pytest
  - python -m compileall agents backend schemas
- Never skip tests

## Git Safety Rules (MANDATORY)
- Always commit before major changes
- Never force push
- Never overwrite history
- Never auto-push without explicit instruction
- Work in feature branches for risky changes

## Audit-Fix Loop Rules
- Simulation -> Auditor -> Fix -> Test -> Repeat
- Only fix:
  - confirmed deterministic bugs
- Do NOT fix:
  - model-quality issues automatically
  - subjective behavior without explicit instruction

## Auditor Behavior
- Must behave like a senior reviewer:
  - detect invalid diagnosis (symptom vs diagnosis)
  - detect duplicate findings
  - detect contradictions
  - detect scoring unfairness
- Must provide reasoning, not just classification

## Simulation Rules
- Student must NOT see:
  - hidden diagnosis
  - full presenting symptoms
- Investigations:
  - student sees raw results only
  - evaluator sees full structured data

## Diagnosis Evaluation
- Support:
  - alias matching
  - partial correctness
- Reject symptom-only guesses

## Performance Rules
- Use fast mode for testing loops
- Use small runs first (1-3)
- Avoid long runs unless stable

## Coding Rules
- Do not change architecture without explicit instruction
- Prefer minimal fixes
- Always add regression tests for bugs

## Goal
- Build a realistic, reliable clinical training simulator
- Prioritize correctness, realism, and safety over speed
