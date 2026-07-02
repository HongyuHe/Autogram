"""Concrete long-context subagent transport for SchemaSpec induction.

The transport shells out to an agentic coding CLI running in headless / non-interactive mode and
returns its raw response text.  Three harnesses are supported and selected by the
``AUTOGRAM_SUBAGENT_HARNESS`` environment variable (or the ``harness=`` argument):

* ``copilot`` (default) -> ``copilot -s --model <m> --context <ctx> ...`` (prompt on stdin)
* ``codex``             -> ``codex exec --model <m>``                     (prompt on stdin)
* ``claude``            -> ``claude -p --model <m>``                      (prompt on stdin)

Every harness reads the prompt from stdin, so only the argv differs.  Copilot is the default; the
other two provide drop-in compatibility with the OpenAI Codex and Anthropic Claude Code CLIs.  All
knobs (binary name, model, context, extra args, timeout) are overridable via environment variables
so no harness detail is hard-coded at the call site.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List


def _copilot_args(executable: str, model: str, context: str) -> List[str]:
    args = [executable, "-s"]
    if model:
        args += ["--model", model]
    if context:
        args += ["--context", context]
    args += [
        "--no-custom-instructions",
        "--disable-builtin-mcps",
        "--no-ask-user",
        "--no-color",
        "--available-tools=",
    ]
    return args


def _codex_args(executable: str, model: str, context: str) -> List[str]:
    # `codex exec` is the non-interactive mode; the prompt is read from stdin.
    args = [executable, "exec"]
    if model:
        args += ["--model", model]
    return args


def _claude_args(executable: str, model: str, context: str) -> List[str]:
    # `claude -p` prints the response non-interactively; the prompt is read from stdin.
    args = [executable, "-p"]
    if model:
        args += ["--model", model]
    return args


@dataclass(frozen=True)
class Harness:
    """A supported agentic-CLI harness: its default binary/model and how to build its argv."""

    name: str
    command: str
    model: str
    context: str
    build_args: Callable[[str, str, str], List[str]]


HARNESSES: dict[str, Harness] = {
    "copilot": Harness("copilot", "copilot", "gpt-5.5", "long_context", _copilot_args),
    "codex": Harness("codex", "codex", "gpt-5.5", "", _codex_args),
    "claude": Harness("claude", "claude", "sonnet", "", _claude_args),
}

DEFAULT_HARNESS = "copilot"


def _split_extra(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(shlex.split(value))


class AutogramSubagentRunner:
    """Invoke a real long-context agentic-CLI subagent and return its raw response text.

    The default harness is Copilot; pass ``harness="codex"`` or ``harness="claude"`` (or set
    ``AUTOGRAM_SUBAGENT_HARNESS``) to route to those CLIs instead.
    """

    _cache: dict[str, str] = {}

    def __init__(
        self,
        *,
        harness: str | None = None,
        command: str | None = None,
        model: str | None = None,
        context: str | None = None,
        timeout_s: float | None = None,
        log_path: str | None = None,
        extra_args: tuple[str, ...] | None = None,
    ):
        self.harness_name = (harness or os.environ.get("AUTOGRAM_SUBAGENT_HARNESS", DEFAULT_HARNESS)).strip().lower()
        spec = HARNESSES.get(self.harness_name)
        if spec is None:
            raise ValueError(
                f"unknown subagent harness {self.harness_name!r}; choose one of {sorted(HARNESSES)}"
            )
        self._spec = spec
        self.command = command or os.environ.get("AUTOGRAM_SUBAGENT_COMMAND", spec.command)
        self.model = model if model is not None else os.environ.get("AUTOGRAM_SUBAGENT_MODEL", spec.model)
        self.context = context if context is not None else os.environ.get("AUTOGRAM_SUBAGENT_CONTEXT", spec.context)
        self.timeout_s = float(timeout_s or os.environ.get("AUTOGRAM_SUBAGENT_TIMEOUT", "300"))
        self.extra_args = (
            extra_args if extra_args is not None
            else _split_extra(os.environ.get("AUTOGRAM_SUBAGENT_EXTRA_ARGS", ""))
        )
        self.log_path = Path(log_path or os.environ.get(
            "AUTOGRAM_SUBAGENT_LOG",
            r"artifacts\subagent_schema_induction.jsonl",
        ))

    def __call__(self, prompt: str) -> str:
        prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if os.environ.get("AUTOGRAM_SUBAGENT_CACHE") == "1" and prompt_id in self._cache:
            self._log({"event": "cache_hit", "prompt_id": prompt_id, "model": self.model})
            return self._cache[prompt_id]

        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                "Subagent schema induction requires a real subagent transport; "
                f"command {self.command!r} for harness {self.harness_name!r} was not found. "
                f"Install/authenticate the {self.harness_name} CLI or inject a responder."
            )
        args = list(self._spec.build_args(executable, self.model, self.context)) + list(self.extra_args)
        self._log({
            "event": "invoke",
            "prompt_id": prompt_id,
            "harness": self.harness_name,
            "model": self.model,
            "context": self.context,
            "command": executable,
        })
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            self._log({"event": "timeout", "prompt_id": prompt_id, "timeout_s": self.timeout_s})
            raise RuntimeError(
                f"Subagent schema induction timed out after {self.timeout_s:g}s "
                f"(harness {self.harness_name!r})"
            ) from exc

        if proc.returncode != 0:
            self._log({
                "event": "error",
                "prompt_id": prompt_id,
                "harness": self.harness_name,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
            })
            raise RuntimeError(
                "Subagent schema induction failed; no offline fallback is available. "
                f"harness={self.harness_name!r} stderr: {proc.stderr[-1000:]}"
            )

        output = proc.stdout.strip()
        self._log({
            "event": "complete",
            "prompt_id": prompt_id,
            "harness": self.harness_name,
            "returncode": proc.returncode,
            "stdout_chars": len(output),
        })
        if os.environ.get("AUTOGRAM_SUBAGENT_CACHE") == "1":
            self._cache[prompt_id] = output
        return output

    def _log(self, payload: dict) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
