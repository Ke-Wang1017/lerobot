## Run read-only MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/`. All calls are pure reads — no subprocess is started, stopped, or signaled.

## get_run_status — is anything running?

_Intent: First call any AI agent makes when asked 'how's the robot doing'. Returns `{running: false, command: null}` when no subprocess is managed; the AI knows to suggest starting something rather than poll forever._

**→** `get_run_status()`

**←**

```json
{
  "running": false,
  "command": null
}
```

## get_run_output — what did the subprocess print?

_Intent: Snapshot of the last N captured stdout/stderr lines. With no active subprocess the buffer is empty; `truncated` remains false so the AI can branch on shape._

**→** `get_run_output(last_n=100)`

**←**

```json
{
  "lines": [],
  "total_buffered": 0,
  "truncated": false
}
```

## get_latency_metrics — performance of the active loop

_Intent: Latest atomic-replace snapshot for the requested source (default `teleop`). Empty-stub response when no run is active — `n_records=0` is the branch trigger._

**→** `get_latency_metrics()`

**←**

```json
{
  "n_records": 0,
  "dropped_records": 0,
  "overrun_ratio": 0.0,
  "stages": {},
  "series": {}
}
```

_Intent: Unknown source — returns the same empty-stub shape rather than a 404, so the AI doesn't have to special-case the 'wrong source name' path._

**→** `get_latency_metrics(source='made_up')`

**←**

```json
{
  "n_records": 0,
  "dropped_records": 0,
  "overrun_ratio": 0.0,
  "stages": {},
  "series": {}
}
```

## get_rlt_metrics — RLT training progress

_Intent: Same empty-stub pattern when no RLT session is active: `mode == 'IDLE'`. When a training run IS active, this returns the live metrics from the session's `metrics.json`._

**→** `get_rlt_metrics()`

**←**

```json
{
  "episode": 0,
  "step_count": 0,
  "buffer_size": 0,
  "total_updates": 0,
  "mode": "IDLE",
  "success_rate": 0,
  "total_successes": 0,
  "total_episodes": 0,
  "series": {}
}
```
