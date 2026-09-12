# RAG vs Agentic: A Hallucination Benchmark on Real Legal Text

**Does adding retrieval actually stop an LLM from making things up? Does adding a verification agent on top of that catch what retrieval alone misses?**

This project answers both questions empirically — by running the same 15 questions through three progressively more capable pipelines (plain LLM → RAG → RAG + Agentic Verification) against India's Digital Personal Data Protection Act & Rules, and measuring exactly where each one fails.

## The three conditions, head to head

| # | Condition | What it does |
|---|---|---|
| 1 | **LLM-alone** | Question straight to the model. No context, no retrieval — just what it learned in training. |
| 2 | **RAG** | Question → retrieve top-3 relevant chunks from a vector database → generate an answer grounded in those chunks. |
| 3 | **RAG + Agentic Verification** | Same as #2, plus a second LLM call that checks whether the answer is actually supported by its source. On a negative verdict, the system **automatically retries with broader retrieval and re-verifies** — a real decision-and-action loop, not just a flagged warning for a human to read. |

## Why this domain

Legal/regulatory text is unforgiving — a wrong number or misattributed clause has real consequences, which makes it a much better hallucination stress-test than a trivia benchmark. The corpus used here (India's DPDP Act, 2023 and DPDP Rules, 2025) has a built-in advantage for this purpose: the Rules were notified **13 November 2025**, well after the generation model's training cutoff — so the LLM-alone condition is guaranteed to be working from incomplete knowledge, not just imperfect recall.

## Corpus

| Document | Source | Pages |
|---|---|---|
| Digital Personal Data Protection Act, 2023 | Official Gazette (meity.gov.in) | 21 |
| Digital Personal Data Protection Rules, 2025 | Official Gazette notification, 13 Nov 2025 | 18 |

Two official government PDFs only — no secondary/aggregator sources, to keep grounding-correctness measurements uncontaminated.

## Pipeline

**Ingestion:** `pdfplumber`, plain text extraction per page. Table extraction was tested and deliberately dropped — for this corpus it duplicated surrounding text, misread two-column legal layouts, and picked up bilingual (Hindi/English) gazette noise. Text-only extraction was more reliable for prose-based legal provisions.

**Chunking:** `RecursiveCharacterTextSplitter` (LangChain), chunk_size=500, overlap=100 → 325 chunks. Chunk size was reduced from an initial 1000 after testing showed larger chunks diluted short but structurally important text (like chapter headings) inside unrelated surrounding content.

**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (local, free, 384-dim).

**Vector store:** ChromaDB (persistent, local), each chunk tagged with `source` and `page` metadata for grounding checks.

**Generation model:** `openai/gpt-oss-120b` via Groq API.
- Released by OpenAI: **August 5, 2025** (Apache 2.0, open-weight, 117B params / 5.1B active, MoE)
- Knowledge cutoff: **June 2024**
- `temperature=0` for reproducibility

The cutoff gap (June 2024 model vs. a Nov 2025 Rules document) is exactly why LLM-alone fails so consistently below — this isn't a hypothetical knowledge gap, it's a measured one.

*Initial runs used `llama-3.3-70b-versatile`, deprecated by Groq mid-project (decommissioned 16 Aug 2026); migrated to `openai/gpt-oss-120b`.*

## Benchmark

15 hand-written questions across common / specialized / obscure / unanswerable-by-omission categories, each with a manually verified ground-truth answer and exact source page.

## Results

| Condition | Outcome |
|---|---|
| **LLM-alone** | Failed on nearly every question — confidently. Fabricated law names, wrong jurisdictions (cited UK/Canadian/Australian law for Indian-law questions), invented procedures (video-KYC, blockchain consent logs) not in the source, wrong dates, wrong penalty figures. Never hedged; fabrication was always presented as fact. |
| **RAG** | 10/15 fully correct and grounded. 3/15 correctly abstained ("not found in context") on retrieval misses. 2/15 genuine errors: understated a penalty (₹200 crore → "Hundred crore") and applied the wrong retention-period rule (3 years → answered 1 year). |
| **RAG + Agentic Verification** | Correctly caught the retention-period error. **Missed the penalty-amount error on two independent runs** — verified the same numerically wrong answer as fully grounded both times, including once after triggering the agentic retry path. Also over-trusted an incomplete answer (partial safeguards list marked as fully correct). |

### Retrieval quality (measured separately from final answers)

- ~53% (8/15) clean top-3 retrieval accuracy on manual inspection.
- Strong on procedural clauses with distinctive phrasing ("48 hours," "local level committee").
- Weak on foundational/definitional text — lost to denser, more repetitive competing chunks even after reducing chunk size.
- Weak when an answer spans a reference clause and a separately-located schedule (e.g. salary tables) — the reference retrieves, the actual data often doesn't make top-3.

## The core finding

RAG sharply reduces hallucination versus LLM-alone — expected, and confirmed here. The more interesting result is that **agentic verification is not a reliable safety net by itself**. It reliably catches a wrong-rule substitution but consistently misses a single-word numeric truncation ("hundred crore" vs "two hundred crore") — reproduced on a second independent run, including through the retry path. This is a precise, demonstrable gap: the verifier judges *general support*, not *exact numeric match* — a distinction that matters enormously in legal/financial domains where two "plausible" numbers can differ by 2x.

## Limitations

- Agentic retry triggers only on an explicit negative verdict and retries once with broader retrieval — it doesn't reformulate the query, and a false-positive "yes" bypasses the retry entirely (as seen above).
- 15-question benchmark is small; error counts are illustrative, not statistically robust.
- No latency/cost metrics captured in this run.
- Text-only ingestion; a table-heavy corpus (e.g. financial filings) would need different extraction.
- Small local embedding model trades precision for speed — a likely contributor to definitional-text retrieval misses.

## Next steps

- Make the verifier extract and compare numeric values explicitly, not just judge general support.
- Have the retry step reformulate the query, not just widen top-k.
- Add hybrid (semantic + keyword) retrieval to recover sparse-but-important text like headings and definitions.
- Chunk by document structure (sections/chapters) instead of fixed character counts.
- Try a larger embedding model and measure the retrieval-accuracy delta.
- Add latency/cost tracking per condition.

## Tech stack

Python · pdfplumber · pandas · LangChain (`RecursiveCharacterTextSplitter`) · sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · Groq API (`openai/gpt-oss-120b`)
