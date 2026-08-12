#!/usr/bin/env python3
"""PreToolUse hook for Bash: refuse `git push`, and refuse shell writes into `cfe/eval/`.

Two rules worth hard enforcement, not just CLAUDE.md convention: blocking edits to `cfe/eval/**`
and blocking `git push`. The file-tool half lives in `block_frozen_eval.py`; this covers the
shell, which can reach both.

The `cfe/eval/` check matters because a hook that only guards Edit and Write is trivially
sidestepped by `sed -i`, a heredoc, or a redirect — not maliciously, just by reaching for the
nearest tool. Freezing a path means freezing it against every route.

Exits 2 to block, which surfaces stderr back to the model.
"""

import json
import re
import sys

GIT_PUSH = re.compile(r"(?:^|[|;&]|\s)git\s+(?:-\S+\s+)*push\b")

# Shell constructs that could write into the frozen directory. Deliberately broad: a false
# positive costs one explanation, a false negative silently rewrites every prior result.
EVAL_WRITE = re.compile(
    r"(?:>>?\s*\S*cfe/eval/)"  # redirect into it
    r"|(?:\b(?:sed|perl|awk)\b[^|;&]*-i[^|;&]*cfe/eval/)"  # in-place edit
    r"|(?:\b(?:rm|mv|cp|truncate|tee|patch|dd)\b[^|;&]*cfe/eval/)"  # file manipulation
    r"|(?:\bgit\s+(?:checkout|restore|apply|revert)\b[^|;&]*cfe/eval/)"  # git-side rewrites
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command:
        return 0

    if GIT_PUSH.search(command):
        print(
            "BLOCKED: `git push` is disabled in this repo.\n"
            "\n"
            "Commits are fine; publishing is the user's call and is done by them. If a push is\n"
            "genuinely wanted, ask — do not route around this with a different remote command.",
            file=sys.stderr,
        )
        return 2

    if EVAL_WRITE.search(command):
        print(
            "BLOCKED: this command writes into cfe/eval/, which is FROZEN after Phase 2.\n"
            "\n"
            "The Edit/Write hook does not cover the shell, so the freeze is enforced here too.\n"
            "Reaching for a shell tool because the file tools are blocked is exactly the path\n"
            "this exists to close.\n"
            "\n"
            "If there is a real bug in the frozen code: STOP and tell the user. Approved changes\n"
            "are logged in CHANGELOG.md with reason and rejected alternatives (CLAUDE.md rule 1).",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
