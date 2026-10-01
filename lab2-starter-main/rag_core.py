"""Core ingestion, retrieval, and report generation for the M&A review tool.

The application prefers ChromaDB when it is installed.  A dependency-free local
backend is deliberately retained so that reviewers can still inspect the corpus
and validate the workflow before installing optional packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DATA_FOLDERS = ("board_papers", "contracts", "emails")
MAJOR_CLIENTS = ("PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]+", re.I)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    category: str
    title: str
    chunk_index: int
    line_start: int
    line_end: int
    clients: str = ""
    date: str = ""


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float
    lexical_score: float

    @property
    def citation(self) -> str:
        return f"{self.chunk.source}:L{self.chunk.line_start}-L{self.chunk.line_end}"


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def embed_text(text: str, dimensions: int = 512) -> list[float]:
    """Create a deterministic, local feature-hashing embedding.

    Word unigrams and bigrams make exact legal language highly retrievable, while
    character trigrams add tolerance for variants such as PayWise/Paywise.
    """
    words = _tokens(text)
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    features.extend(f"c:{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dimensions
        vector[index] += 1.0 if (raw >> 9) & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _title_for(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = re.sub(r"^[#*\s]+|[*\s]+$", "", line).strip()
        if clean.lower().startswith(("subject:", "email subject:")):
            return clean.split(":", 1)[1].strip() or fallback
        if 4 <= len(clean) <= 140 and clean not in {"---", "___"}:
            return clean
    return fallback


def _date_for(text: str) -> str:
    patterns = (
        r"\b(?:Date|Effective Date):\s*\**([^\n*]{4,50})",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}",
        r"\b20\d{2}-\d{2}-\d{2}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return (match.group(1) if match.lastindex else match.group(0)).strip()
    return ""


def discover_documents(data_root: Path) -> list[Path]:
    paths: list[Path] = []
    for folder in DATA_FOLDERS:
        paths.extend(sorted((data_root / folder).glob("*.txt")))
    return paths


def chunk_document(path: Path, data_root: Path, max_chars: int = 1500, overlap_lines: int = 3) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    title = _title_for(text, path.stem.replace("_", " "))
    clients = ", ".join(client for client in MAJOR_CLIENTS if re.search(re.escape(client), text, re.I))
    date = _date_for(text)
    source = str(path.relative_to(data_root))
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines):
            addition = len(lines[end]) + 1
            if end > start and size + addition > max_chars:
                break
            size += addition
            end += 1
        body = "\n".join(lines[start:end]).strip()
        if body:
            digest = hashlib.sha1(f"{source}:{start}:{body}".encode("utf-8")).hexdigest()[:20]
            chunks.append(Chunk(digest, body, source, path.parent.name, title, len(chunks), start + 1, end, clients, date))
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def load_chunks(data_root: Path) -> list[Chunk]:
    return [chunk for path in discover_documents(data_root) for chunk in chunk_document(path, data_root)]


def corpus_signature(data_root: Path) -> str:
    digest = hashlib.sha256()
    for path in discover_documents(data_root):
        stat = path.stat()
        digest.update(str(path.relative_to(data_root)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:20]


class EvidenceStore:
    """Persistent Chroma store with a transparent in-memory fallback."""

    def __init__(self, data_root: str | Path, db_path: str | Path | None = None):
        self.data_root = Path(data_root).resolve()
        self.db_path = Path(db_path or self.data_root / ".chroma").resolve()
        self.chunks: list[Chunk] = []
        self._collection = None
        self.backend = "local"
        try:
            import chromadb  # type: ignore

            self.db_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.db_path))
            self._collection = client.get_or_create_collection(
                "canvassian_due_diligence",
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "ChromaDB"
        except Exception:
            self._collection = None

    @property
    def manifest_path(self) -> Path:
        return self.db_path / "manifest.json"

    def _write_manifest(self, signature: str) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        payload = {"signature": signature, "chunks": [asdict(chunk) for chunk in self.chunks]}
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _read_manifest(self) -> dict | None:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def ensure_index(self, force: bool = False) -> dict[str, int | str]:
        signature = corpus_signature(self.data_root)
        manifest = self._read_manifest()
        valid = bool(manifest and manifest.get("signature") == signature)
        if valid and not force:
            self.chunks = [Chunk(**item) for item in manifest["chunks"]]
            if self._collection is None or self._collection.count() == len(self.chunks):
                return self.stats()

        self.chunks = load_chunks(self.data_root)
        if self._collection is not None:
            existing = self._collection.get(include=[]).get("ids", [])
            if existing:
                for start in range(0, len(existing), 500):
                    self._collection.delete(ids=existing[start:start + 500])
            for start in range(0, len(self.chunks), 100):
                batch = self.chunks[start:start + 100]
                self._collection.add(
                    ids=[item.id for item in batch],
                    documents=[item.text for item in batch],
                    embeddings=[embed_text(item.text) for item in batch],
                    metadatas=[{
                        "source": item.source, "category": item.category, "title": item.title,
                        "chunk_index": item.chunk_index, "line_start": item.line_start,
                        "line_end": item.line_end, "clients": item.clients, "date": item.date,
                    } for item in batch],
                )
        self._write_manifest(signature)
        return self.stats()

    def stats(self) -> dict[str, int | str]:
        docs = discover_documents(self.data_root)
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "board_papers": sum(path.parent.name == "board_papers" for path in docs),
            "contracts": sum(path.parent.name == "contracts" for path in docs),
            "emails": sum(path.parent.name == "emails" for path in docs),
            "backend": self.backend,
        }

    def _local_candidates(self, query: str, limit: int, category: str | None) -> list[tuple[Chunk, float]]:
        query_vector = embed_text(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks:
            if category and chunk.category != category:
                continue
            vector = embed_text(chunk.text)
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append((chunk, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def search(self, query: str, top_k: int = 8, category: str | None = None) -> list[SearchResult]:
        if not self.chunks:
            self.ensure_index()
        candidate_count = min(max(top_k * 8, 40), len(self.chunks))
        candidates: list[tuple[Chunk, float]] = []
        if self._collection is not None and self._collection.count():
            where = {"category": category} if category else None
            result = self._collection.query(
                query_embeddings=[embed_text(query)], n_results=candidate_count,
                where=where, include=["documents", "metadatas", "distances"],
            )
            for doc, meta, distance, item_id in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], result["ids"][0]
            ):
                candidates.append((Chunk(item_id, doc, **meta), 1.0 - float(distance)))
        else:
            candidates = self._local_candidates(query, candidate_count, category)

        query_tokens = _tokens(query)
        query_counts = Counter(query_tokens)
        phrase = " ".join(query_tokens)
        reranked: list[SearchResult] = []
        for chunk, vector_score in candidates:
            haystack = f"{chunk.title} {chunk.source} {chunk.text}".lower()
            counts = Counter(_tokens(haystack))
            overlap = sum(min(counts[token], count) for token, count in query_counts.items())
            lexical = overlap / max(1, sum(query_counts.values()))
            if phrase and phrase in " ".join(_tokens(haystack)):
                lexical += 1.5
            score = 0.58 * vector_score + 0.42 * lexical
            reranked.append(SearchResult(chunk, score, vector_score, lexical))
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


def _line_excerpt(path: Path, pattern: str, context: int = 2) -> tuple[str, int, int] | None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    regex = re.compile(pattern, re.I)
    for index, line in enumerate(lines):
        if regex.search(line):
            start, end = max(0, index - context), min(len(lines), index + context + 1)
            excerpt = " ".join(item.strip(" *#-\t") for item in lines[start:end] if item.strip(" *#-\t"))
            return excerpt, start + 1, end
    return None


def change_control_review(data_root: str | Path) -> list[dict[str, str | int | list[str]]]:
    root = Path(data_root)
    rows: list[dict[str, str | int | list[str]]] = []
    contract_paths = sorted((root / "contracts").glob("*.txt"))
    for client in MAJOR_CLIENTS:
        relevant = [path for path in contract_paths if re.search(client, path.read_text(encoding="utf-8", errors="replace"), re.I)]
        control: list[str] = []
        convenience = 0
        excerpts: list[str] = []
        for path in relevant:
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"change (?:in|of) control", text, re.I):
                control.append(path.name)
                hit = _line_excerpt(path, r"change (?:in|of) control", 3)
                if hit:
                    excerpt, start, end = hit
                    excerpts.append(f"{excerpt} [{path.relative_to(root)}:L{start}-L{end}]")
            if re.search(r"terminat\w* (?:this agreement )?(?:for convenience|for any reason)", text, re.I):
                convenience += 1
        if control:
            status = "ADVERSE"
        elif convenience:
            status = "REVIEW"
        else:
            status = "NOT FOUND"
        rows.append({
            "client": client, "contracts": len(relevant), "control_clauses": len(control),
            "convenience_termination": convenience, "status": status,
            "sources": control, "evidence": excerpts,
        })
    return rows


TOPICS = (
    {
        "heading": "1. Founder / key-person risk — HIGH",
        "query": "Jane Wu CEO potential departure stepping down within six months succession leadership transition",
        "evidence_queries": (
            "Jane Wu potential departure stepping down within six months",
            "Leadership Update Announcement Jane Wu potential departure remains committed",
        ),
        "finding": (
            "The evidence supports a material key-person risk. A board paper records Jane Wu's possible departure "
            "within six months for personal reasons, identifies client-confidence and operational impacts, and discusses "
            "successors. An internal email says she remains committed, but does not resolve the board-level departure risk."
        ),
        "action": "Make signing/closing conditional on an agreed retention package, minimum service period, succession plan and orderly handover; interview Jane directly.",
    },
    {
        "heading": "2. PayWise concentration and credit risk — HIGH",
        "query": "PayWise financial difficulty unpaid invoices cash flow insolvency major client revenue dependency",
        "evidence_queries": (
            "PayWise financial difficulty unpaid invoices cash flow",
            "PayWise delayed payments layoffs financial distress revenue percentage",
        ),
        "finding": (
            "The rumour is corroborated by internal correspondence stating that PayWise is experiencing financial "
            "difficulty and calling for review of unpaid invoices, cash-flow impact and project dependencies. Given the "
            "client-supplied 20% revenue concentration, downside could be material. The corpus does not establish the "
            "amount or ageing of receivables."
        ),
        "action": "Obtain an aged receivables ledger, payment history, current credit information, project profitability and direct customer confirmation; model loss/default scenarios and seek a price adjustment or specific indemnity.",
    },
    {
        "heading": "3. Intellectual-property litigation — HIGH",
        "query": "patent infringement lawsuit injunction damages AUD 10 million CanvasPro TechWave TechInnovators",
        "evidence_queries": (
            "TechWave patent infringement AUD 10 million injunction CanvasPro",
            "TechInnovators formal letter patent infringement CanvassPro",
            "Alphabear patent infringement litigation CanvasSoft",
        ),
        "finding": (
            "Multiple records describe patent claims concerning core software, including a formal notice, possible "
            "injunctive relief and a stated AUD 10 million damages claim. Some records name different claimants/products, "
            "so the matters may be separate or the dataset may contain inconsistencies requiring reconciliation."
        ),
        "action": "Require a complete disputes schedule, claim correspondence and counsel's merits/quantum assessment; verify IP provenance, insurance and freedom-to-operate; use escrow, indemnity and closing conditions.",
    },
    {
        "heading": "4. Privacy, security and regulatory exposure — MEDIUM/HIGH",
        "query": "regulatory scrutiny data privacy investigation data breach compliance fines sanctions cybersecurity",
        "evidence_queries": (
            "regulatory scrutiny software data privacy policies investigation fines sanctions",
            "Charlemont data breaches unauthorized access compliance risks",
            "risk management gaps compliance breaches cyber threats regulatory penalties",
        ),
        "finding": (
            "The documents flag regulatory scrutiny of data-privacy policies, cyber/compliance control gaps and prior "
            "minor unauthorised-access incidents at a major client. These statements require substantiation but are "
            "material for a cybersecurity-software target."
        ),
        "action": "Commission privacy/cyber diligence; obtain incident registers, regulator correspondence, penetration-test reports, certifications, cyber insurance and contractual notification records.",
    },
)


def _evidence_markdown(results: Sequence[SearchResult], limit: int = 3) -> str:
    bullets = []
    seen: set[str] = set()
    for result in results:
        if result.chunk.source in seen:
            continue
        seen.add(result.chunk.source)
        clean = re.sub(r"\s+", " ", result.chunk.text).strip()
        excerpt = clean[:420].rstrip()
        if len(clean) > 420:
            excerpt += "…"
        bullets.append(f"- **Evidence:** {excerpt}  \n  *Source: `{result.citation}`* ")
        if len(bullets) >= limit:
            break
    return "\n".join(bullets)


def build_due_diligence_report(store: EvidenceStore) -> str:
    store.ensure_index()
    stats = store.stats()
    matrix = change_control_review(store.data_root)
    adverse = [str(row["client"]) for row in matrix if row["status"] == "ADVERSE"]
    lines = [
        "# Canvassian Pty Ltd — Buyer-side preliminary due diligence report",
        "",
        f"**Generated:** {datetime.now().astimezone().strftime('%d %B %Y, %H:%M %Z')}  ",
        f"**Corpus reviewed:** {stats['documents']} documents / {stats['chunks']} chunks using {stats['backend']}  ",
        "**Purpose:** Decision-support for the purchaser's Board; not a substitute for legal advice or full due diligence.",
        "",
        "## Executive conclusion",
        "",
        "**Recommendation: do not approve an unconditional acquisition on the current record.** Proceed only subject to focused confirmatory diligence, protective conditions and valuation/indemnity protection. The most material issues are Jane Wu's possible departure, corroborated PayWise financial distress, termination rights triggered by a sale of Canvassian, and patent/privacy exposure.",
        "",
        "## Priority findings",
        "",
    ]
    for topic in TOPICS:
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for evidence_query in topic.get("evidence_queries", (topic["query"],)):
            for result in store.search(str(evidence_query), top_k=4):
                if result.chunk.id not in seen_ids:
                    results.append(result)
                    seen_ids.add(result.chunk.id)
                    break
        lines.extend([
            f"### {topic['heading']}", "", str(topic["finding"]), "",
            _evidence_markdown(results), "", f"**Recommended response:** {topic['action']}", "",
        ])

    lines.extend([
        "### 5. Major-client change-in-control review — HIGH",
        "",
        f"Express change-in-control language was located for **{', '.join(adverse) or 'no major clients'}**. An absence of matching language is not a clean legal conclusion: several contracts contain broad termination-for-convenience rights, and the corpus may contain drafts, duplicates or incomplete contract sets.",
        "",
        "| Client | Contracts found | Express control clauses | Convenience termination | Triage |",
        "|---|---:|---:|---:|---|",
    ])
    for row in matrix:
        lines.append(f"| {row['client']} | {row['contracts']} | {row['control_clauses']} | {row['convenience_termination']} | **{row['status']}** |")
    lines.extend(["", "**Clause evidence:**", ""])
    for row in matrix:
        if row["evidence"]:
            lines.append(f"- **{row['client']}:** {row['evidence'][0]}")
    lines.extend([
        "",
        "**Recommended response:** Establish the executed/current agreement for each major client, reconcile amendments and schedules, obtain required notices/consents before signing or closing, and seek customer confirmations. Do not treat a keyword absence as confirmation that no restriction exists.",
        "",
        "## Immediate Board action list",
        "",
        "1. Pause unconditional sign-off and open focused confirmatory diligence on the four high-priority areas above.",
        "2. Secure Jane Wu retention and succession arrangements as transaction conditions.",
        "3. Quantify PayWise exposure and run valuation/covenant scenarios for delayed payment, default and customer loss.",
        "4. Obtain Bravocat, Charlemont and Echona consents or waivers where required; confirm the executed contract set for all six major clients.",
        "5. Ring-fence IP, privacy and litigation exposures through specific indemnities, escrow/holdback, warranty coverage and disclosure requirements.",
        "",
        "## Methodology and limitations",
        "",
        f"The supplied folders contain **{stats['documents']}**, not the 1,000 documents stated in the brief ({stats['board_papers']} board papers, {stats['contracts']} contracts and {stats['emails']} emails). The tool chunks and hybrid-searches all supplied text, then applies clause-focused checks. Documents appear synthetic and may include inconsistent names, duplicate/draft contracts and conflicting allegations. Findings must therefore be validated against originals, executed agreements, financial records, management interviews and external counsel advice.",
    ])
    return "\n".join(lines)
