from dne.chat import ChatSession


def _echo():
    seen = []

    def fn(question, history):
        seen.append(list(history))
        return f"answer to {question!r} with {len(history)} prior turns"
    return fn, seen


def test_history_grows():
    fn, _ = _echo()
    chat = ChatSession(fn)
    chat.ask("How do I fix my backhand in long rallies?")
    chat.ask("Which drills?")
    assert [m["role"] for m in chat.history] == ["user", "assistant", "user", "assistant"]


def test_follow_up_receives_previous_turns():
    fn, seen = _echo()
    chat = ChatSession(fn)
    chat.ask("My backhand breaks down in long rallies.")
    chat.ask("Which drills?")
    assert seen[0] == []
    assert seen[1][0] == {"role": "user", "content": "My backhand breaks down in long rallies."}
    assert seen[1][1]["role"] == "assistant"


def test_history_truncated_to_max_turns():
    fn, seen = _echo()
    chat = ChatSession(fn, max_turns=2)
    for i in range(5):
        chat.ask(f"q{i}")
    assert len(seen[-1]) == 4  # 2 turns = 2 user + 2 assistant messages
    assert seen[-1][0]["content"] == "q2"


def test_rewrite_used_for_follow_ups_only():
    fn, _ = _echo()
    calls = []

    def rewrite(q, history):
        calls.append(q)
        return f"standalone: {q}"
    chat = ChatSession(fn, rewrite_fn=rewrite)
    chat.ask("My backhand breaks down in long rallies.")
    chat.ask("Which drills?")
    assert calls == ["Which drills?"]
    assert chat.standalone == ["My backhand breaks down in long rallies.", "standalone: Which drills?"]


def test_aask_awaits_async_search_and_rewrite():
    import asyncio
    rewrites, searched = [], []

    async def search(q, history):
        searched.append(q)
        return f"answer to {q}"

    async def rewrite(q, history):
        rewrites.append((q, len(history)))
        return f"standalone: {q}"

    async def go():
        chat = ChatSession(search, rewrite_fn=rewrite)
        await chat.aask("My juniors lose focus in long matches.")
        await chat.aask("Which drills help?")
        return chat
    chat = asyncio.run(go())
    assert rewrites == [("Which drills help?", 2)]
    assert searched == ["My juniors lose focus in long matches.", "standalone: Which drills help?"]
    assert chat.history[-1] == {"role": "assistant", "content": "answer to standalone: Which drills help?"}


def test_load_dialogues_skips_blanks_and_comments(tmp_path):
    from dne.chat import load_dialogues
    p = tmp_path / "dialogue.txt"
    p.write_text("# demo\nMy juniors lose focus.\n\n  Which drills help?  \n")
    assert load_dialogues(p) == [["My juniors lose focus.", "Which drills help?"]]


def test_load_dialogues_splits_on_dashes(tmp_path):
    from dne.chat import load_dialogues
    p = tmp_path / "dialogue.txt"
    p.write_text("Q1\nQ2\n---\nR1\nR2\nR3\n")
    assert load_dialogues(p) == [["Q1", "Q2"], ["R1", "R2", "R3"]]


def test_load_dialogues_needs_two_to_six_turns_each(tmp_path):
    import pytest
    from dne.chat import load_dialogues
    p = tmp_path / "dialogue.txt"
    p.write_text("Q1\nQ2\n---\nOnly one turn.\n")
    with pytest.raises(ValueError):
        load_dialogues(p)
    p.write_text("\n".join(f"q{i}" for i in range(7)))
    with pytest.raises(ValueError):
        load_dialogues(p)


def test_transcript_markdown_groups_by_dialogue_then_system():
    from dne.chat import transcript_markdown
    rows = [
        {"dialogue": 2, "system": "basic", "turn": 1, "user": "R1", "standalone": "R1", "answer": "C1", "latency_s": 5.0},
        {"dialogue": 1, "system": "local", "turn": 1, "user": "Q1", "standalone": "Q1", "answer": "A1", "latency_s": 10.0},
        {"dialogue": 1, "system": "basic", "turn": 2, "user": "Q2", "standalone": "Q2 full", "answer": "B2", "latency_s": 5.0},
        {"dialogue": 1, "system": "basic", "turn": 1, "user": "Q1", "standalone": "Q1", "answer": "B1", "latency_s": 5.0},
    ]
    md = transcript_markdown(rows)
    assert md.index("Dialogue 1") < md.index("Dialogue 2")
    d1 = md[:md.index("Dialogue 2")]
    assert d1.index("### basic") < d1.index("### local")
    assert d1.index("B1") < d1.index("B2")
    assert "Q2 full" in md            # a rewritten follow-up is shown
    assert md.count("Standalone") == 1  # an unchanged turn is not
