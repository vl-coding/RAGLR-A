# RAGLR-A — A Guide for Researchers and Non-Technical Users

This page explains what this tool does, when it's useful, and how to read its results — without assuming any background in machine learning or software engineering.

---

## What is this, in plain language?

RAGLR-A is a **search assistant for academic papers**. You type in a research question — like "how are graph neural networks used for drug discovery?" — and it searches a large collection of papers from [arXiv](https://arxiv.org) (a free repository of research papers across computer science, physics, math, biology, economics, and more) to find the ones most relevant to your question.

It's different from a normal keyword search (like searching a library catalog for exact words) in a few ways:

- **It understands meaning, not just words.** If you ask about "neural networks that learn from molecular structures," it can find a paper that talks about "graph-based deep learning for chemistry" even if none of your words match exactly.
- **It uses an AI model to "imagine" an ideal answer first.** Before searching, the system asks Claude (an AI assistant) to write a short hypothetical abstract describing what a perfect matching paper might say. It then searches for real papers that look similar to that imagined abstract. This is a common technique called "HyDE" and tends to surface more relevant results than searching with your raw question alone.
- **It combines two different search methods.** One method matches based on exact words and phrases (similar to traditional search), and the other matches based on overall meaning and topic. The results from both are blended together so you get the benefit of each.
- **It explains itself.** For each paper returned, the system also asks Claude to write a short note explaining *why* that paper might be relevant to your question, along with two numeric scores (described below).

Think of it as a research assistant that reads your question, thinks about what a good answer would look like, searches a large library with two different strategies, and then hands you a shortlist with notes on why each item might be worth your time.

---

## What you get back

For each search, you'll see a ranked list of papers (up to a configurable number, usually 10). For each paper:

- **Title, authors, and abstract**
- **A link to the paper on arXiv**
- **A relevance score** (1–10) — how directly this paper addresses your question
- **A specificity score** (1–10) — how narrowly focused the paper is on your specific topic, versus being a broad survey or tangentially related work
- **A short written justification** explaining the AI's reasoning for the scores

You'll also see some behind-the-scenes information (a "trace") showing how the search was narrowed down and how long each step took — useful if you're curious about *how* the result was produced, but not essential for using the results.

---

## When to use this tool

RAGLR-A is well-suited for:

- **Getting oriented in an unfamiliar topic.** If you're starting a new project and want a quick sense of "what's out there," this can produce a useful starting reading list in a couple of minutes.
- **Finding papers when you don't know the right terminology.** Because the search understands meaning, it can surface relevant work even if you don't know the specific jargon researchers in that subfield use.
- **Brainstorming related work or framing sections.** Quickly surfacing papers that touch on a topic from different angles can help you map out how a field talks about a problem.
- **Spot-checking coverage.** Running a few related queries can give you a rough sense of how much (or how little) has been written on a topic on arXiv.

## When NOT to rely on this tool

- **As a substitute for a systematic literature review.** This tool does not guarantee completeness. It returns a *capped* shortlist (e.g., up to 10 results) based on similarity, not an exhaustive list of everything relevant. A systematic review (e.g., PRISMA-style) requires comprehensive, reproducible search strategies across multiple databases — this tool is a single, non-exhaustive source.
- **For citation counts, impact, or "who cites whom."** RAGLR-A does not model citations, authorship networks, or how influential a paper has been. It only judges topical similarity to your question.
- **For very recent events or papers**, unless the underlying data has been refreshed recently. The system searches a snapshot of arXiv collected at some point in time (the snapshot date depends on how the system was set up and how often it's updated).
- **For non-English research, or fields not well-represented on arXiv** (e.g., most social sciences, humanities, medicine outside bioRxiv-adjacent areas). arXiv mainly covers physics, math, computer science, and related quantitative fields, plus a smaller amount of biology, economics, and statistics.
- **For highly mathematical or symbol-heavy queries.** The search tools used here work better with words and phrases than with mathematical notation, so queries dense with equations or symbols may retrieve less reliably.
- **As a final word on "is this relevant."** The AI-generated relevance scores and explanations are a *starting point for your own judgment*, not a substitute for actually reading the paper. They can occasionally be overly generous, slightly off-topic, or (rarely) based on a misunderstanding of the abstract.
- **Older "foundational" papers may rank lower than you'd expect.** A famous, decades-old paper on a topic often uses older terminology that doesn't match how the topic is described today, so it may not appear near the top even if it's the most important paper on the subject. If you're looking for the "classic" paper on a topic, it's worth searching for it by name directly in addition to using this tool.

**Bottom line:** treat RAGLR-A as a fast, AI-assisted starting point for exploring a topic — a way to generate candidates and gain orientation — not as the final or only step in a literature search.

---

## What do the scores actually mean?

### Relevance score and specificity score (1–10, shown for each paper)

These two numbers are generated by an AI model (Claude) that reads your question and the paper's abstract, and rates:

- **Relevance (1–10):** How directly does this paper address your question? A 10 means the paper's central contribution *is* your topic — it's likely the kind of paper you'd cite as a primary reference. Lower scores (e.g., 1) mean the paper is essentially unrelated. Middle scores (e.g., 6–7) mean the paper touches on a related area but its main focus is something else.
- **Specificity (1–10):** How narrowly focused is the paper on your exact question, versus being a broad survey, a tangential application, or a passing mention? A high specificity score suggests the paper is specifically *about* what you asked; a lower score suggests it's more of a general or adjacent treatment.

**Important caveats:**
- These scores are produced by the same AI model for every search, using a fixed scoring guide, so a score of "8" is *intended* to mean roughly the same thing across different searches. However, they're still AI judgments, not measurements — use them to compare papers *within the same search results*, not as precise, universal numbers.
- Because the AI only sees papers that have already been shortlisted by the search (i.e., papers that were judged at least somewhat relevant before this scoring step even happens), scores tend to cluster on the higher end of the scale (most scored papers land in the 6–10 range). A "6" among your results may still be one of the *less* relevant papers returned, even though 6/10 sounds like a middling score in everyday use.
- Occasionally the written explanation may be overly positive, overly verbose, or based on a surface-level reading of the abstract — especially for very technical or specialized subject matter the AI may not deeply understand. Always use your own judgment, especially before citing a paper.

### Precision, Recall, NDCG, MRR (in the project's evaluation reports)

If you look at the project's technical evaluation results, you may see metrics like "Precision@10," "Recall@10," "NDCG@10," and "MRR." These describe how well the *system as a whole* performs on a fixed set of test questions with known "correct" answers — they are not numbers you'll see while using the tool day-to-day, but here's what they mean if you're curious:

- **Precision@10** — Of the 10 results returned, what fraction were actually relevant (according to a hand-curated answer key)? Higher is better.
- **Recall@10** — Of all the known-relevant papers for a test question, what fraction showed up in the top 10 results? Higher is better.
- **NDCG@10** — Similar to recall, but also rewards putting the *most* relevant papers near the *top* of the list, not just somewhere in the top 10. Higher is better.
- **MRR (Mean Reciprocal Rank)** — On average, how close to the #1 spot did the *first* relevant paper appear? An MRR of 1.0 would mean the first relevant paper was always ranked #1; an MRR of 0.5 means it was typically ranked around #2.

These numbers are measured against a small, hand-picked set of test questions (26 of them, at the time of writing), each with a short list of papers known to be good answers. Because that answer key is small and necessarily incomplete (a real corpus almost certainly contains other relevant papers not on the list), these numbers should be read as a rough, conservative signal of quality — useful for comparing different versions of the system to each other, but not as an absolute guarantee of how well any single real-world search will perform.

---

## A note on cost and speed

Each search makes several calls to an AI service (Claude) and, depending on how the system is run, may take anywhere from a few seconds (on a small test collection) to one to two minutes (on the full multi-million-paper collection). If you're running this yourself, be aware that heavy use will incur API costs from the AI provider.

---

For more technical detail — including how the system is built, its known limitations, and the full evaluation methodology — see the [README](README.md) and the `docs/` folder.
