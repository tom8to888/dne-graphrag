"""Conversational agent over the index (brief §13.1: "implement a conversational agent").

graphrag 3.3's public query API answers one question at a time, so follow-ups are handled the usual
conversational-RAG way: the LLM first rewrites the follow-up into a standalone question using the recent
history ("condense question"), then the chosen search mode (basic / local / global) answers it. The same
mechanism works for every mode, which keeps the RAG vs GraphRAG comparison fair in chat as well.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

Message = dict[str, str]
SearchFn = Callable[[str, list[Message]], str]
RewriteFn = Callable[[str, list[Message]], str]

CONDENSE_PROMPT = """You help a tennis coaching assistant. Rewrite the player's last message as ONE standalone
question that can be understood without the conversation. Keep every detail that matters (stroke, problem,
age, level, conditions). If it is already standalone, return it unchanged. Reply with the question only.

Conversation so far:
{history}

Last message: {question}
Standalone question:"""


def format_history(history: list[Message]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(none)"


class ChatSession:
    def __init__(self, search_fn: SearchFn, rewrite_fn: RewriteFn | None = None, max_turns: int = 6):
        self.search_fn = search_fn
        self.rewrite_fn = rewrite_fn
        self.max_turns = max_turns
        self.history: list[Message] = []
        self.standalone: list[str] = []

    def recent(self) -> list[Message]:
        return self.history[-2 * self.max_turns:] if self.max_turns else []

    def ask(self, question: str) -> str:
        recent = self.recent()
        query = self.rewrite_fn(question, recent) if (self.rewrite_fn and recent) else question
        self.standalone.append(query)
        answer = self.search_fn(query, recent)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return answer

    async def aask(self, question: str) -> str:
        """ask() for async search/rewrite functions (graphrag's engines run in one event loop)."""
        recent = self.recent()
        query = await self.rewrite_fn(question, recent) if (self.rewrite_fn and recent) else question
        self.standalone.append(query)
        answer = await self.search_fn(query, recent)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return answer


def load_dialogues(path: Path) -> list[list[str]]:
    """One user turn per line; a line '---' starts the next dialogue; blank lines and # lines are skipped.
    Each dialogue has 2-6 turns."""
    dialogues: list[list[str]] = [[]]
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln == "---":
            dialogues.append([])
        elif ln and not ln.startswith("#"):
            dialogues[-1].append(ln)
    for n, turns in enumerate(dialogues, start=1):
        if not 2 <= len(turns) <= 6:
            raise ValueError(f"{path}: dialogue {n} needs 2-6 turns, found {len(turns)}")
    return dialogues


def transcript_markdown(rows: list[dict]) -> str:
    """Per dialogue, then per system (basic, local, global), turns in order; a rewritten follow-up is shown."""
    order = {"basic": 0, "local": 1, "global": 2}
    out = []
    for d in sorted({r.get("dialogue", 1) for r in rows}):
        out.append(f"## Dialogue {d}\n")
        in_d = [r for r in rows if r.get("dialogue", 1) == d]
        for system in sorted({r["system"] for r in in_d}, key=lambda s: order.get(s, 9)):
            out.append(f"### {system}\n")
            for r in sorted((r for r in in_d if r["system"] == system), key=lambda r: r["turn"]):
                out.append(f"**User ({r['turn']}):** {r['user']}\n")
                if r.get("standalone") and r["standalone"] != r["user"]:
                    out.append(f"*Standalone question:* {r['standalone']}\n")
                ans = r["answer"] if r.get("answer") else f"(error: {r.get('error', 'no answer')})"
                out.append(f"**{system} ({r.get('latency_s', 0):.0f} s):** {ans}\n")
    return "\n".join(out)
