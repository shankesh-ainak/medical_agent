# dscribe — Agentic Discharge-Summary Drafting

An agentic AI that reads a patient's messy source-note PDFs (typed **and**
handwritten) and produces a **structured discharge-summary draft for clinician
review**. It plans, reads the chart, reconciles medications, surfaces conflicts
and pending results, checks drug interactions, escalates safety concerns — and,
above all, **never invents a clinical fact**.

Demo video - https://drive.google.com/file/d/10j4nUKcrIkTq68Nt-r1HdwW_sj-AKRjX/view?usp=sharing

> The output is always a *draft for review*, never a finalized clinical
> document. Gaps and disagreements are surfaced, not filled.

---

## Table of contents

- [What it does](#what-it-does)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [The agent loop (LangGraph)](#the-agent-loop-langgraph)
- [The tools the agent can call](#the-tools-the-agent-can-call)
- [Data model & the no-fabrication contract](#data-model--the-no-fabrication-contract)
- [How each hard requirement is met](#how-each-hard-requirement-is-met)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Test fixtures](#test-fixtures)
- [Observability](#observability)
- [Project layout](#project-layout)
- [Security & privacy](#security--privacy)
- [Limitations & caveats](#limitations--caveats)

---

## What it does

Given **one PDF bundle** for a single patient (a mix of digital text pages and
scanned/handwritten pages — admission notes, progress notes, lab reports,
medication charts, nursing notes, discharge notes), `dscribe`:

1. **Ingests** every page with a hybrid extractor (digital text layer where it
   exists, vision OCR where it doesn't) into one searchable, metadata-tagged
   index.
2. **Runs an agent** that plans what to read, reads it, and fills a fixed set of
   discharge-summary sections — each value anchored to the source page(s) it
   came from.
3. **Reconciles medications** (admission vs discharge: added / stopped / changed
   / continued) and flags any change with no documented reason.
4. **Detects conflicts** (e.g. two different principal diagnoses or lab values
   across pages) and records *both*, never silently picking a winner.
5. **Handles pending and missing data** explicitly (`PENDING` / `MISSING`)
   instead of fabricating a value.
6. **Checks drug interactions** via a (mocked) external service and **escalates**
   anything unsafe for clinician review.
7. **Emits a full trace** of every step (reasoning → tool → inputs → result) and
   renders a Markdown draft.

There is also a **Streamlit app**: Tab 1 uploads a PDF and shows the draft +
trace; Tab 2 is a grounded chat that answers questions about the same patient
with page citations.

## Design principles

- **No fabrication is a code invariant, not a prompt wish.** A clinical value is
  only writable if it cites at least one real source page. The guarded writer
  `record_field` *rejects* uncited values, so the agent's only honest
  alternatives are `mark_missing`, `mark_pending`, or `record_conflict`.
  Fabrication is structurally impossible, not merely discouraged.
- **Safety-critical reads are exhaustive, not top-k.** Exploration uses semantic
  (vector) search; medication reconciliation and conflict detection use exact,
  exhaustive page fetches, because a top-k retriever can silently drop a page and
  defeat the no-fabrication guarantee.
- **Clinical logic is deterministic.** Reconciliation and conflict detection are
  plain, auditable Python — no LLM judgement in the part that decides whether two
  facts disagree.
- **The agent always terminates with a complete draft.** A hard step cap plus a
  `force_finalize` path guarantee the loop can't run forever and never leaves a
  section silently blank.
- **Errors are observations, never crashes.** Every tool call is wrapped so an
  exception becomes a structured `{"status": "error", ...}` envelope the agent
  can react to.

## Architecture

```
              ┌──────────────────────── Ingestion ─────────────────────────┐
  patient.pdf │  pdfplumber (digital text layer) ─┐                         │
   (1 bundle) │                                    ├─► PageDoc[] ─► LlamaIndex
              │  vision OCR (handwritten/scanned) ─┘   (+page metadata)  +  │
              │  routed per page by a char-count threshold        Chroma idx │
              └─────────────────────────────────────────────────┬───────────┘
                                                                  │
        ┌──────────────── LangGraph agent loop ──────────────────▼──────────┐
        │  START ─► agent ─►(route)─► tools ─► agent ─► …                    │
        │                       └────► force_finalize ─► END                 │
        │  • step cap   • per-step trace   • guarded draft writers           │
        └─────────────────────────────┬─────────────────────────────────────┘
                                       ▼
                  Structured DischargeSummary  ─►  Markdown draft
                  (value+source | MISSING | PENDING | CONFLICT)
                                       │
                                       ▼
                  Streamlit: draft + trace (Tab 1) · grounded chat (Tab 2)
```

**Stack:** Python ≥3.13 · OpenAI (agent model, vision OCR, embeddings) ·
LangGraph + LangChain (agent loop) · LlamaIndex + Chroma (index/retrieval) ·
pdfplumber + PyMuPDF (PDF text & rendering) · Pydantic (schema) ·
tenacity (retry) · Streamlit (UI).

## The pipeline, stage by stage

`pipeline.run(pdf_path)` (`dscribe/pipeline.py`) orchestrates everything the CLI
and the Streamlit app share:

1. **Load** — `ingestion/pdf_loader.py`. For each page, pull the digital text
   layer with `pdfplumber` and render a PNG with PyMuPDF (`fitz`) for possible
   OCR. No LLM calls here.
2. **Extract (hybrid routing)** — `ingestion/extractor.py`. If a page's text
   layer has ≥ `text_layer_min_chars` (default 40) characters it's used directly
   (`source="text-layer"`) and classified by header keywords. Otherwise the page
   is treated as scanned/handwritten and sent to **vision OCR**
   (`source="vision"`). A vision failure on one page degrades gracefully to a
   low-confidence `PageDoc` with an error note rather than aborting the ingest.
   Results are cached on disk by file hash (`storage/cache/`) so repeated runs
   don't re-OCR.
3. **Vision OCR** — `ingestion/vision_ocr.py`. A vision model transcribes the
   page image verbatim (writing `[illegible]` rather than guessing) and returns
   `{text, doc_type, confidence}` as strict JSON. Retried on transient failure.
4. **Index** — `ingestion/index_store.py`. Pages are chunked
   (`SentenceSplitter`, 512/64), embedded (`text-embedding-3-small`), and stored
   in a **fresh per-run Chroma collection** (each patient's chart is isolated).
   Page-level metadata (`page_no`, `doc_type`, `source`, `confidence`) rides on
   every node, so provenance survives retrieval.
5. **Agent** — `agent/loop.py`. The LangGraph loop drives the draft to
   completion (next section).
6. **Render** — `render/markdown.py`. The structured `DischargeSummary` becomes
   a Markdown draft with statuses, per-value source tags `[p3]`, a medication
   table, and a Clinician Review section.

## The agent loop (LangGraph)

A small explicit state machine (`dscribe/agent/loop.py`):

```
START ─► agent ─►(route)─► tools ─► agent ─► …
                    └────► force_finalize ─► END
```

- **`agent` node** — calls the tool-bound chat model (`temperature=0`). Its
  natural-language rationale is stashed in `ctx.last_reasoning` so the *next*
  tool call's trace entry reads reasoning → action → result.
- **`route`** — the control gate (requirement #9). It sends flow to
  `force_finalize` if (a) the draft is already finalized, (b) the step budget
  (`max_steps`, default 14) is exhausted, or (c) the model stopped requesting
  tools. Otherwise it goes to `tools`.
- **`tools` node** — LangGraph's `ToolNode` executes the requested tool(s) and
  returns observations to the agent.
- **`force_finalize`** — closes the draft if the agent didn't, auto-marking any
  untouched section `MISSING`. This is the guarantee that output is always
  structurally complete.
- A LangGraph `recursion_limit` (`max_steps*2 + 5`) is a belt-and-braces backstop
  beneath the explicit step cap.

The system prompt (`agent/prompts.py`) encodes the clinical contract in prose,
but the hard guarantees live in code (guarded tools + step cap), so the agent
behaves safely even if it ignores the prompt.

## The tools the agent can call

All 13 are assembled in `dscribe/tools/registry.py`. Each is wrapped by a
`traced` decorator that (a) converts any exception into a structured error
envelope and (b) logs a trace step on every call.

**Document access (read)**
- `list_source_documents()` — list every page with `doc_type`, source, legibility
  confidence, and a preview. The recommended first call.
- `read_document(page_no)` — full text of one page (read a med table end-to-end
  without retrieval truncation).
- `search_documents(query, doc_type?, top_k=5)` — semantic search for *where* a
  fact might be; for complete comparisons, prefer reading whole pages.

**Deterministic clinical logic**
- `reconcile_medications(admission, discharge)` — diff the two lists →
  added/stopped/changed/continued; flags changes lacking a documented reason
  (`tools/reconciliation.py`).
- `detect_conflicts(field, observations)` — groups observed values by light
  normalisation; >1 group ⇒ conflict. Never picks a winner (`tools/conflicts.py`).
- `check_drug_interactions(medications)` — **mocked** external lookup
  (`tools/drug_interactions.py`). The first call for a given med set deliberately
  fails (transient) to exercise the retry path.

**Escalation**
- `flag_for_clinician_review(topic, reason, severity=info|warning|critical)`.

**Guarded draft writers (write)** — `tools/draft.py`
- `record_field(section, value, sources)` — **rejects** empty/invalid sources.
- `mark_missing(section, note?)`
- `mark_pending(section, note, sources?)`
- `record_conflict(section, options, note?)` — requires ≥2 competing options.
- `set_discharge_medications(medications)` — each item carries change_type and
  sources.
- `finalize_draft()` — call once; auto-marks untouched sections `MISSING`.

## Data model & the no-fabrication contract

`dscribe/schema.py` defines the output. Every clinical fact is a `Field` with a
`FieldStatus`:

| Status | Meaning |
|---|---|
| `PRESENT` | a value sourced from the documents (carries ≥1 `SourceRef`) |
| `MISSING` | not found anywhere in the source notes |
| `PENDING` | ordered/awaited, no result yet (e.g. culture sent) |
| `CONFLICT` | sources disagree; both kept, neither chosen |

A `SourceRef` is `{page_no, doc_type, quote?}`. The eleven required text sections
(`TEXT_SECTIONS`) are: patient demographics, admission date, discharge date,
principal diagnosis, secondary diagnoses, hospital course, procedures, allergies,
follow-up instructions, pending results, discharge condition — plus an explicit
discharge **medication list** (`MedicationChange[]`) and **review flags**
(`ReviewFlag[]`).

The contract is enforced in `DraftStore` (`tools/draft.py`): `_validate_sources`
raises `DraftRejected` if a value has no sources or cites a page that isn't in
the bundle. So "a value with no citation" is unrepresentable.

## How each hard requirement is met

| # | Requirement | Where |
|---|---|---|
| 1 | Real agent loop (plan / re-plan) | `agent/loop.py` — LangGraph `agent ↔ tools` state machine |
| 2 | In-house PDF ingestion | `ingestion/` — pdfplumber text layer + PyMuPDF render + vision OCR |
| 3 | No fabrication | `tools/draft.py` rejects uncited values; `schema.py` field model |
| 4 | Pending / missing data | `mark_pending` / `mark_missing`; `finalize` auto-marks gaps |
| 5 | Medication reconciliation + flag | `tools/reconciliation.py` (deterministic diff) |
| 6 | Conflicting info (flag, never pick) | `tools/conflicts.py` + `record_conflict` (keeps both) |
| 7 | Use tools, agent decides when | `tools/registry.py`; mocked `check_drug_interactions`, `flag_for_clinician_review` |
| 8 | Robust failure handling | `robustness.py` (tenacity retry + structured error envelopes); flaky mock |
| 9 | Control (hard step cap) | `config.max_steps` + `route`/`force_finalize`; LangGraph `recursion_limit` |
| 10 | Observability trace | `observability/trace.py` — reasoning → action → inputs → result (console + JSONL) |

## Setup

Requires **Python ≥ 3.13** and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env        # then edit .env and set OPENAI_API_KEY
```

The only external calls are to the OpenAI API (vision OCR, embeddings, agent &
chat models). Everything else — PDF parsing, chunking, the vector store — runs
locally.

## Usage

**CLI**
```bash
dscribe run sample_patient_data.pdf              # draft + live trace, saved to storage/
dscribe run sample_patient_data.pdf --no-cache   # force re-extraction / re-OCR
dscribe run sample_patient_data.pdf --out draft.md
```
Writes `<name>.md` and `<name>.json` (the structured `DischargeSummary`) and a
JSONL trace under `storage/trace/`.

**Streamlit**
```bash
streamlit run streamlit_app.py
```
- *Tab 1 — Discharge Summary*: upload a PDF, generate the draft, view headline
  counts (flags / conflicts / pending), the Markdown draft, a download button,
  and the full agent trace.
- *Tab 2 — Ask the Chart*: grounded Q&A over the same patient's indexed notes
  (`dscribe/chat.py`), answering only from retrieved excerpts and citing pages,
  or saying "I can't find that in the records."

**As a library**
```python
from dscribe.pipeline import run
result = run("sample_patient_data.pdf")
print(result.markdown)              # the rendered draft
result.summary                      # the structured DischargeSummary (Pydantic)
result.trace                        # list of trace steps (dicts)
```

## Configuration

All settings live in `dscribe/config.py` and can be overridden by environment
variables (e.g. in `.env`):

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| `agent_model` | `DSCRIBE_AGENT_MODEL` | `gpt-5.2` | agent + chat model |
| `vision_model` | `DSCRIBE_VISION_MODEL` | `gpt-5.2` | page-image OCR / classification |
| `embed_model` | `DSCRIBE_EMBED_MODEL` | `text-embedding-3-small` | embeddings for the index |
| `max_steps` | `DSCRIBE_MAX_STEPS` | `14` | hard agent step cap (req #9) |
| `text_layer_min_chars` | `DSCRIBE_TEXT_MIN_CHARS` | `40` | below this ⇒ route page to vision OCR |
| `page_render_dpi` | `DSCRIBE_RENDER_DPI` | `180` | DPI for page→PNG render |
| `tool_max_retries` | `DSCRIBE_TOOL_RETRIES` | `3` | tenacity retry attempts |
| `llm_timeout_s` | `DSCRIBE_LLM_TIMEOUT` | `60` | per-call timeout (seconds) |
| `storage_dir` | — | `<repo>/storage` | cache, chroma, traces, uploads |

## Test fixtures

Two synthetic, self-contained PDFs are checked in, each with a generator script
so you can regenerate or tweak them (the generators need `PyMuPDF`/`fitz`, which
is already a project dependency). **All data is fictional.**

- **`test_patient_record.pdf`** (5 pages) — built by `make_test_pdf.py`. Compact
  smoke test: gastroenteritis-vs-DKA diagnosis conflict, sodium conflict, pending
  cultures, a metformin dose change with no reason, an ondansetron+tramadol
  interaction, no documented procedures (forces MISSING), and one handwritten
  nursing note (vision-OCR page).
- **`test_patient_record_extended.pdf`** (12 pages, 4 handwritten) — built by
  `make_test_pdf_extended.py`. Harder bundle with cross-source conflicts that
  span typed *and* handwritten pages:
  - principal diagnosis CAP vs PE; potassium 5.8 vs 4.2;
  - **allergy** "NKDA" (typed) vs "SULFA" (handwritten);
  - **warfarin dose** 5 mg (typed discharge) vs 3 mg (handwritten order);
  - pending blood/sputum cultures + pleural-fluid cytology;
  - reconciliation across added/stopped/changed/continued meds;
  - two interacting pairs (meropenem+valproate major, ondansetron+tramadol
    moderate);
  - documented procedures (CTPA, thoracentesis); and **no follow-up
    instructions anywhere** (forces MISSING).

The handwritten-only facts (warfarin 3 mg, sulfa allergy) are invisible to text
extraction — they exist only in page images — so catching the resulting conflicts
also proves the vision-OCR path genuinely read the handwriting.

Run either through the CLI or the Streamlit uploader:
```bash
dscribe run test_patient_record_extended.pdf
```

## Observability

`observability/trace.py` records one `TraceStep` per action —
`reasoning → action → inputs → result (→ next_decision)` — and emits it three
ways: an in-memory list (shown in the Streamlit trace expander), a JSONL file
(`storage/trace/<name>.jsonl`), and a human-readable console line (CLI). This
lets a reviewer replay exactly why the agent did what it did.

## Project layout

```
dscribe/
  __init__.py        config.py     schema.py     robustness.py
  pipeline.py        chat.py       cli.py
  ingestion/
    pdf_loader.py        # pdfplumber text + PyMuPDF render  (no LLM)
    vision_ocr.py        # vision-model transcription + doc_type
    extractor.py         # hybrid routing + on-disk cache
    index_store.py       # LlamaIndex + Chroma; semantic + exhaustive reads
  agent/
    prompts.py           # system prompt (the clinical contract in prose)
    loop.py              # LangGraph state machine + step cap
  tools/
    context.py           # ToolContext (index/draft/trace shared state)
    draft.py             # guarded writers — the no-fabrication enforcement
    reconciliation.py    # deterministic medication diff
    conflicts.py         # deterministic conflict detection
    drug_interactions.py # MOCKED flaky external service
    registry.py          # wraps the 13 tools (tracing + error envelopes)
  observability/
    trace.py             # per-step trace (console + JSONL)
  render/
    markdown.py          # DischargeSummary -> Markdown draft
streamlit_app.py         # Tab 1: upload+draft+trace · Tab 2: grounded chat
pyproject.toml           # package metadata; `dscribe` console script
```

## Security & privacy

All sample data is **synthetic**. Extraction stays local (pdfplumber, PyMuPDF,
local Chroma); no patient data is sent to any third-party retention service. The
only network calls are to the OpenAI API for vision OCR, embeddings, and the
agent/chat models. Do not feed real patient data through a hosted model without
the appropriate agreements in place.

## Limitations & caveats

- **OCR cost/latency.** Vision OCR makes one call per non-text page, so the first
  run on a large or heavily-handwritten bundle is slow; results are cached by file
  hash under `storage/cache/` (use `--no-cache` to force a refresh).
- **The drug-interaction service is mocked** (`tools/drug_interactions.py`) — a
  small illustrative DB and a deliberately flaky first call. Swap in a real
  service behind the same interface for production.
- **Conflict detection errs toward flagging.** Light normalisation only; when
  values still differ it raises a conflict for a human to resolve. A
  false-positive flag is the safe failure mode; silently merging a real
  disagreement is not.
- **One patient per run.** The Chroma collection is recreated each run to keep
  charts isolated; this is not a multi-tenant store.
- **It produces a draft, not a clinical document.** A clinician must review every
  field, every flag, and every conflict before any use.
