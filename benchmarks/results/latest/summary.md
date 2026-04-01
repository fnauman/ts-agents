# Refactor Benchmark Summary

This deterministic benchmark compares four assist levels on three representative tasks.

| Assist level | Success rate | Parse failure rate | Invalid tool calls | Avg artifact completeness | Avg duration (ms) | Retries | Recovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `plain_model` | 0.00 | 1.00 | 0 | 0.00 | 0.0 | 0 | 0.00 |
| `plain_tools` | 0.67 | 0.00 | 1 | 0.00 | 314.1 | 1 | 0.33 |
| `structured_discovery` | 1.00 | 0.00 | 0 | 0.00 | 62.6 | 0 | 0.00 |
| `skills_workflows` | 1.00 | 0.00 | 0 | 1.00 | 114.9 | 0 | 0.00 |

## Scenario Notes

- `plain_model` is deliberately freeform and non-machine-runnable, which gives it a schema/parse failure on every scenario.
- `plain_tools` uses raw tool access with no discovery or workflow bundling, so it tends to under-produce artifacts and occasionally guess the wrong contract.
- `structured_discovery` uses `tool search` / `tool show` before raw tool execution, which improves task completion but still lacks workflow artifact bundles.
- `skills_workflows` inspects the policy layer and then runs the workflow layer, which is why it should dominate artifact completeness.
