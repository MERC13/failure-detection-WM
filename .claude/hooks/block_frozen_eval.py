#!/usr/bin/env python3
"""PreToolUse hook: refuse edits to `cfe/eval/**` (SPEC.md I3, CLAUDE.md rule 1).

CLAUDE.md shapes behavior; it doesn't block anything. For the two rules whose violation would
be unrecoverable, a PreToolUse hook is used instead — this is one of the two.

`cfe/eval/` holds the reported metrics and the replay protocol. Editing them rewrites every
result the project has ever produced, and the damage is silent — old numbers stay in
`results/runs/` looking valid while meaning something different. Instructions alone cannot
prevent that; a hook can.

Exits 2 to block, which surfaces stderr back to the model.

**To make a legitimate change**, per CLAUDE.md rule 1: stop and get the change approved, log it
in `CHANGELOG.md` with the reason and the rejected alternatives, then disable this hook for the
edit and re-enable it. The friction is the feature.
"""

import json
import os
import sys

FROZEN = os.path.join("cfe", "eval") + os.sep


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never block because the hook itself failed to parse

    path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not path:
        return 0

    normalised = os.path.normpath(os.path.abspath(path))
    if (os.sep + FROZEN) not in normalised + os.sep:
        return 0

    print(
        f"BLOCKED: {path} is under cfe/eval/, which is FROZEN after Phase 2.\n"
        "\n"
        "These files hold SPEC.md §6's metrics and the replay protocol. Changing them rewrites\n"
        "every number already in results/runs/, without those numbers looking any different.\n"
        "\n"
        "If you believe there is a genuine bug here: STOP and tell the user. Do not work around\n"
        "this hook, and do not reimplement the affected logic elsewhere to avoid it. An\n"
        "approved change is logged in CHANGELOG.md with its reason and rejected alternatives\n"
        "(CLAUDE.md rule 1), and the user disables this hook deliberately for that edit.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
