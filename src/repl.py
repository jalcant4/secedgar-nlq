import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any


@dataclass
class REPL:
    namespace: dict[str, Any]= field(default_factory= dict)
    history: list[str]= field(default_factory= list)
    last_error: str | None = None
    last_failed: bool = False
    validation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.namespace["inspect"] = self.inspect_state

    def inspect_state(self, name: str | None = None, mode: str = "summary",
                      limit: int = 30, max_chars: int = 2000) -> Any:
        """Read-only, bounded inspection of persistent REPL state and feedback."""
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not isinstance(max_chars, int) or not 32 <= max_chars <= 10000:
            raise ValueError("max_chars must be between 32 and 10000")
        if mode == "last_error":
            return (self.last_error or "No execution error")[-max_chars:]
        if mode == "validation":
            return self.validation if self.validation is not None else {"status": "not_checked"}
        if name is None:
            if mode != "summary":
                return {"error": "Provide a variable name for this mode"}
            return {key: ("<func>" if callable(value) else type(value).__name__)
                    for key, value in list(self.namespace.items())
                    if not key.startswith("_")}
        if name not in self.namespace:
            return {"error": f"Unknown variable: {name}"}
        value = self.namespace[name]
        if mode == "type":
            return type(value).__name__
        if mode == "keys":
            return list(value.keys())[:limit] if isinstance(value, dict) else {"error": "Not a dictionary"}
        if mode == "summary":
            return repr(value)[:max_chars]
        if mode == "funcs":
            return [k for k, v in self.namespace.items()
                    if not k.startswith("_") and callable(v)]
        return {"error": f"Unsupported inspection mode: {mode}"}


    def _strip_code_fences(self, code: str) -> str:
        """Remove markdown code fences (```python ... ```) if present."""
        code = code.strip()
        if code.startswith("```"):
            # Remove opening fence and language tag
            code = code[3:]
            if "\n" in code:
                code = code.split("\n", 1)[1]
            # Remove closing fence
            code = code.rstrip()
            code = code.removesuffix("```")
            code = code.rstrip()
        return code

    def execute(self, code: str) -> str:
        # Write to output
        output = StringIO()
        self.last_failed = False

        try:
            # Strip markdown code fences (LLMs often wrap code in ```python ... ```)
            code = self._strip_code_fences(code)
            # Redirect to output
            """ TODO: redirect_stdout() and redirect_stderr() modify process-wide streams. 
            They're fine for initial single-threaded RLM development, 
            but they need attention before the Worker and Thinker execute Python concurrently."""
            with redirect_stdout(output), redirect_stderr(output):
                exec(code, self.namespace)  # noqa: S102
        # Catch exceptions with traceback
        except Exception:  # noqa: BLE001
            self.last_failed = True
            self.last_error = traceback.format_exc()
            output.write(self.last_error)

        result = output.getvalue()
        self.history.append(result)
        return result