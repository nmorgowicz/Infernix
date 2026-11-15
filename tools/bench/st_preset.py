"""Build the chat-completion message layout a SillyTavern preset produces, for cache and latency probes.

Reads an exported "OpenAI Settings" preset (JSON) and reproduces what the Prompt Manager sends, without a character card
or world info (their markers are skipped):

* the prompt order of the character id with the most entries, enabled prompts only;
* relative prompts before the Chat History marker become system messages ahead of the history, those after it become
  system messages after the last history message, one message per prompt, empty ones dropped;
* in-chat prompts (injection_position 1) are inserted into the history at their depth, counted from the end
  (depth 0 is after the last message), grouped by depth and ordered by their `injection_order`;
* macros are expanded again for every request, as SillyTavern does: {{char}} and {{user}} take the names given, {{trim}}
  and comments {{// ...}} vanish, {{setvar}} and {{setglobalvar}} store a value that {{getvar}} and {{getglobalvar}}
  return, {{roll:NdM}} and {{random:a,b}} / {{random::a::b}} draw from the request's generator, {{pick}} is stable,
  and any other macro expands to nothing (listed in `Preset.ignored`).

The point is the prompt's shape: how much is static, how much sits after the history, what moves between turns, and
whether a macro changes the prompt in front of the history (which no prefix cache can survive).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

MARKERS = {"worldInfoBefore", "worldInfoAfter", "charDescription", "charPersonality", "personaDescription",
           "scenario", "dialogueExamples"}
ROLE_RANK = {"user": 0, "assistant": 1, "system": 2}
DICE = re.compile(r"^(\d*)d(\d+)([+-]\d+)?$", re.I)


class Macros:
    def __init__(self, char: str, user: str, rng: random.Random):
        self.char, self.user, self.rng = char, user, rng
        self.local: dict[str, str] = {}
        self.glob: dict[str, str] = {}
        self.ignored: set[str] = set()
        self.volatile: set[str] = set()

    def expand(self, text: str) -> str:
        out, i = [], 0
        while i < len(text):
            start = text.find("{{", i)
            if start < 0:
                out.append(text[i:])
                break
            out.append(text[i:start])
            depth, j = 1, start + 2
            while j < len(text) and depth:
                if text.startswith("{{", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("}}", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            body = text[start + 2:j - 2] if depth == 0 else text[start + 2:]
            out.append(self.macro(body))
            i = j
        return "".join(out)

    def macro(self, body: str) -> str:
        head, sep, rest = body.partition("::")
        name = head.strip()
        low = name.lower()
        if low.startswith("//") or low == "trim":
            return ""
        if low == "char":
            return self.char
        if low == "user":
            return self.user
        if low in ("setvar", "setglobalvar"):
            key, _, value = rest.partition("::")
            (self.local if low == "setvar" else self.glob)[key.strip()] = self.expand(value)
            return ""
        if low in ("getvar", "getglobalvar"):
            return (self.local if low == "getvar" else self.glob).get(rest.strip(), "")
        if low.startswith("roll"):
            spec = (name[4:].lstrip(":") if not sep else rest).strip()
            dice = DICE.match(spec)
            if dice:
                count, sides, bonus = int(dice.group(1) or 1), int(dice.group(2)), int(dice.group(3) or 0)
                self.volatile.add("roll")
                return str(sum(self.rng.randint(1, sides) for _ in range(count)) + bonus)
        if low.startswith("random") or low.startswith("pick"):
            options = [o for o in (rest.split("::") if sep else name.partition(":")[2].split(",")) if o.strip()]
            if options:
                if low.startswith("random"):
                    self.volatile.add("random")
                    return self.expand(self.rng.choice(options).strip())
                return self.expand(options[0].strip())
        self.ignored.add(low.split(":")[0])
        return ""


class Preset:
    """A parsed preset; `render(rng)` builds one request's (before, injections, after)."""

    def __init__(self, path: Path, char: str = "Mara", user: str = "Captain"):
        preset = json.loads(Path(path).read_text(encoding="utf-8"))
        prompts = {p["identifier"]: p for p in preset["prompts"]}
        order = max(preset["prompt_order"], key=lambda o: len(o["order"]))["order"]
        self.char, self.user = char, user
        self.entries = []  # (prompt, after_history)
        seen = False
        for entry in order:
            ident = entry["identifier"]
            if ident == "chatHistory":
                seen = True
            elif ident in prompts and entry.get("enabled") and ident not in MARKERS:
                self.entries.append((prompts[ident], seen))
        self.ignored: set[str] = set()
        self.volatile: set[str] = set()

    def render(self, rng: random.Random):
        """Returns (before, injections, after): message lists and [(depth, order, role, content)]. Macro state restarts
        for every call, and prompts are expanded in order, so a getvar before its setvar reads an empty value."""
        macros = Macros(self.char, self.user, rng)
        before, after, injections = [], [], []
        for prompt, behind in self.entries:
            content = macros.expand(prompt.get("content") or "").strip()
            if not content:
                continue
            role = prompt.get("role") or "system"
            if prompt.get("injection_position") == 1:
                injections.append((int(prompt.get("injection_depth") or 0), int(prompt.get("injection_order") or 0),
                                   role, content))
            else:
                (after if behind else before).append({"role": role, "content": content})
        self.ignored |= macros.ignored
        self.volatile |= macros.volatile
        return before, injections, after


def apply(before: list, history: list, injections: list, after: list) -> list:
    """The request's messages: static prefix, history with the in-chat prompts at their depth, trailing prompts."""
    chat = list(history)
    # Deepest first so earlier insertions do not shift the later ones' positions.
    for depth in sorted({d for d, *_ in injections}, reverse=True):
        group = sorted((i for i in injections if i[0] == depth), key=lambda i: (ROLE_RANK.get(i[2], 3), i[1]))
        at = max(0, len(chat) - depth)
        chat[at:at] = [{"role": role, "content": content} for _, _, role, content in group]
    return before + chat + after
