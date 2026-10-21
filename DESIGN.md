# Design notes

**Principle:** code enforces every rule that has a correct answer: identity, eligibility, precedence order, the token budget, permissions, working days, confirmation, routes and masking. The LLM interprets the question, judges which statements describe the same rule, picks tools and writes the answer.

## 1. Tables and the scanned document

- **PDF:** `pdfplumber` extracts tables as rows, rendered as Markdown with the header row. Text inside a table's box is skipped, so values are never duplicated.
- **Headings:**
  - PDF: from font metrics (bold text larger than the body; margin text such as running headers is dropped).
  - DOCX, HTML, Markdown: from styles, `h2`–`h6` tags or `#` lines.
- **Metadata tables** are dropped; the manifest is the source of truth.
- **Scanned pages:** a page with no text layer is rendered at 200 dpi and read with RapidOCR (ONNX, CPU, deterministic). Boxes are grouped into rows, then paragraphs. `ocr.py::repair_spacing` restores dropped spaces with a dictionary-based word splitter, re-joining pieces that the rest of the corpus uses as one word.
- **Holidays:** `holidays.py` reads calendar tables that have a date column into closure dates at ingestion. Tables headed restricted, optional or floating are excluded.

## 2. Chunking and retrieval

- **Chunks:** one per section, up to 350 tokens. Longer sections split on paragraph boundaries. Tables never split mid-row, and a split table repeats its header. Sections here are short and self-contained, so this gives precise locators (the heading, or `p. N`) and keeps values with their context.
- **Hybrid search:** BM25 catches exact policy terms. Local `bge-small-en-v1.5` embeddings catch paraphrases ("doctor's note" vs "medical certificate"). Reciprocal-rank fusion merges the two rankings without score calibration.
- **Packing:**
  - At most 3 chunks per document.
  - Of two near-duplicates (Jaccard ≥ 0.8), the higher-precedence copy is kept. Tied copies are both kept, so a conflict can be reported.
  - When a section is selected, the same-named section of any higher-precedence eligible document is added, so a regional override is never missed.
- **Agentic search:** the agent calls `search_policies` itself, at most twice per message, so it can rephrase or fetch accrual rules after reading a balance.
- **Token budget:**
  - Each search is capped at 1,800 tokens and at whatever remains of 3,000 for the message, counted with tiktoken `cl100k_base`.
  - Later LLM calls in a turn carry all earlier results, so the running total is the per-call count. It is logged on every `llm_call` event.
  - Earlier turns are passed as text only, so their chunks are never resent.

## 3. Section 5 rules

- **Eligibility:** `eligibility.py::is_eligible` checks role versus `access_level`, region (or GLOBAL), and `effective_date <= reference date`. Ineligible chunks are removed before scoring.
- **Precedence:**
  - `precedence.py::precedence_key` = (authority, scope, effective date), compared in that order.
  - Each search result starts with a code-computed precedence order of its documents, with `=` marking ties.
  - The prompt says to take a rule's value from the highest document that states it, and to report a conflict on a tie.
- **Upcoming changes:** future-dated readable documents for the region are ranked with the current ones and labelled `UPCOMING`, with `still_outranked_by` from `blockers_of_upcoming`. A future document always wins on recency, so it can only lose on authority or scope. In the worked example this mentions v3 to India but not to Germany.
- **Reference date:** the model passes a `reference_date` for past-date questions; code clamps it to today at the latest.
- **Missing information:** an answer with no valid citation becomes `no_answer`, with fixed wording.

## 4. Where access control is enforced

| What | Where |
|---|---|
| Identity from header → DB; claims in messages ignored | `main.py::chat`, `web/routes.py::context`, `prompts.py::requester_context` |
| Session ownership (403) | `sessions.py::SessionStore.get_or_create` |
| Restricted or other-region documents never reach the LLM | `search.py::Retriever.search` (filter before ranking) |
| Citations only from chunks retrieved for this requester | `agent.py::ChatService._finalise` |
| Leave data by role | `permissions.py::require_view_leave`, `require_manager`, called inside each tool in `leave_tools.py` |
| Bulk requests (more than one other employee per message) | `agent.py::ChatService._dispatch` |
| Writes only for oneself; another person's request looks like "not found" | `leave_tools.py::check_apply`, `check_cancel` (use `ctx.requester`, never arguments) |

`AccessDenied` ends the turn with a fixed message. The denied data is never returned to the model.

## 5. Personal data

- **Prompts:** the requester block carries ID, role, region, level and join date. Tool results carry no email or phone. Names appear only in `find_employee` and `get_team_leave`, limited to employees the requester may see.
- **Answers:** `pii.py::AnswerScrubber` masks employee emails (`r***@domain`), removes employee phone numbers, and withholds names the requester may not see. It also scrubs `tool_calls` arguments.
- **Logs:** JSON lines with IDs, counts, routes, tool statuses, latency and corpus tokens. Message text, model output and exception messages are never logged. `JsonFormatter` also masks anything shaped like an email or phone number.

## 6. Failure modes found

**Fixed, during development:**
1. OCR dropped spaces, split the company name and misordered the two-column header. Fixed with spacing repair, corpus-vocabulary merging and row grouping.
2. Restricted holidays would have counted as closures. They are excluded, and the calendar is chosen per date under Section 5. All 26 seeded requests' working days are re-derived in tests.
3. The model could assume a leave type. A keyword guard sends it back to ask.
4. Route, citations and tool_calls could disagree. Code now derives the route.
5. A third search could break the budget. Searches are hard-capped.
6. A missing API key gave an HTTP 500. It now returns a clean `no_answer`.

**Fixed, found in the live run (Groq) and covered by tests:**

7. De-duplication kept v1's carry-forward section over v2's near-identical one, so the model gave a superseded deadline. The higher-precedence copy now wins.
8. A broad question missed the India addendum's overriding section. Same-named sections are now pulled in.
9. "Documents I can access" hinted that restricted documents exist. `no_answer` now uses fixed wording.
10. Chunk refs leaked into answers; they are now stripped.
11. The model asked "please confirm" instead of calling `apply_leave`. After a prompt fix, the rules check runs at once.
12. The log scrubber masked dates as phone numbers; it now matches only phone shapes.

**Not fixed:**
- "Same rule" detection depends on the model.
- Text before a document's first heading is located by page or title.

## 7. At 10,000 documents

**What breaks first: startup ingestion.** Ingestion runs fully at every start and would exceed the 10-minute `/health` limit (OCR takes seconds per page, and every chunk is re-embedded). Fixes:
- a persistent index keyed by file hash;
- parallel parse and OCR workers;
- a vector store with metadata filters (Qdrant or pgvector) on access, region and date, so filtering still happens before ranking.

**What breaks next: retrieval quality.** Many versions and near-duplicates crowd the top results. Fixes:
- group by document family and keep the precedence winner plus the next tier;
- add a cross-encoder reranker;
- extract per-rule facts at ingestion, so precedence is decided in code.

**Also:** sessions move to Redis.
