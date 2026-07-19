# Phase 0:Eval 语料构建与 Seed 闭环 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 110 条评测用例构建 104 篇对齐的合成企业语料,提供校验/seed/冒烟三个脚本,使评测用例首次"有题有卷"、可被真实检索命中。

**Architecture:** 校验逻辑放在 `app/evals/corpus.py`(纯函数,可单测),CLI 薄封装在 `scripts/eval/` 下;canonical doc_id 经 `DocumentCreate.metadata_json` → ingestion pipeline(需一处小改)→ ChromaDB chunk metadata 透传,检索结果即可携带 canonical_doc_id 用于命中判定。

**Tech Stack:** Python 3.11+ / uv / pytest / SQLAlchemy / ChromaDB(已装 `ml` extra)/ 现有 `IngestionPipeline` 与 `RetrievalService`。

**对应 spec:** `docs/superpowers/specs/2026-07-19-eval-driven-rag-loop-design.md` 的 Phase 0。后续 Phase 1-4 各有独立计划,在本 PR 合入后再写。

---

## 背景知识(零上下文必读)

- **仓库约定(CLAUDE.md)**:禁止直接推 main;分支名 `feat/<name>`;commit 用语义化前缀;推送前必须跑完整本地验证(见 Task 11)。所有 Python 命令用 `uv run` 前缀,在仓库根目录执行。
- **数据集**:`data/eval/datasets/*.jsonl` 共 110 条用例(3 个文件),格式见 `app/evals/datasets.py:8` 的 `EvalCase`(字段:`id/question/expected_keywords/expected_doc_ids/should_answer`)。其中 6 条负例(`should_answer: false`,expected 字段为空),104 条正例,每条正例恰好引用 1 个唯一 doc_id → **需撰写 104 篇文档**。
- **语料目录**:`data/eval/corpus/<doc_id>.md`,一篇文档对应一个 doc_id。
- **摄入链路**:先 `DocumentRepository.create_document()` 建记录(`storage_path` 直接指向语料文件绝对路径即可——`LocalDocumentStorage.get_local_path()` 原样返回,见 `app/documents/storage.py:80`),置状态 QUEUED,再 `await IngestionPipeline().run(tenant_id, document_id)`。不需要 Celery worker。
- **embedding 离线行为**:无 API key 时 `OpenAICompatibleEmbeddingProvider` 回退到确定性假向量(`app/embeddings/providers_openai.py:55`),摄入不会崩溃,但检索质量无意义。**冒烟验收必须在配置了真实 embedding API key 的环境跑**。
- **tenant**:默认 `"public"`(`app/config.py:95` 的 `tenant_default_id`)。

## 文件结构

| 动作 | 路径 | 职责 |
|---|---|---|
| Create | `app/evals/corpus.py` | 语料↔用例对齐校验(纯函数) |
| Create | `tests/test_eval_corpus.py` | 校验逻辑单测 |
| Create | `scripts/eval/validate_corpus.py` | 校验 CLI |
| Create | `data/eval/corpus/*.md` × 104 | 合成语料(Task 4-7 分批) |
| Modify | `app/ingestion/pipeline.py:128` | 把文档级 metadata_json 并入 chunk metadata |
| Modify | `tests/test_ingestion_pipeline.py` | 新增 metadata 透传测试 |
| Create | `scripts/eval/seed_corpus.py` | 语料摄入脚本(幂等) |
| Create | `scripts/eval/smoke_retrieval.py` | 检索冒烟脚本 |

---

### Task 1: 创建分支

- [ ] **Step 1: 从最新 main 拉实现分支**

```bash
git checkout main && git pull && git checkout -b feat/eval-corpus
```

Expected: `Switched to a new branch 'feat/eval-corpus'`

---

### Task 2: 校验模块 `app/evals/corpus.py`(TDD)

**Files:**
- Create: `app/evals/corpus.py`
- Test: `tests/test_eval_corpus.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_eval_corpus.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.evals.corpus import collect_expected_doc_ids, validate_corpus
from app.evals.datasets import load_cases


def _write_dataset(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _case(
    case_id: str,
    doc_id: str | None,
    keywords: list[str],
    should_answer: bool = True,
) -> dict:
    return {
        "id": case_id,
        "question": f"question for {case_id}?",
        "expected_keywords": keywords,
        "expected_doc_ids": [doc_id] if doc_id else [],
        "should_answer": should_answer,
    }


def test_validate_corpus_passes_when_docs_and_keywords_align(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text(
        "# Leave Policy\nAnnual Leave is 25 days.", encoding="utf-8"
    )

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok
    assert report.issues == []
    assert report.orphan_files == []
    assert report.checked_docs == 1


def test_validate_corpus_flags_missing_doc(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    report = validate_corpus(load_cases(dataset), corpus)

    assert not report.ok
    assert [issue.kind for issue in report.issues] == ["missing_doc"]
    assert report.issues[0].doc_id == "doc_hr_leave"


def test_validate_corpus_flags_missing_keyword(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave", "carry-over"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("Annual Leave is 25 days.", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert [issue.kind for issue in report.issues] == ["missing_keyword"]
    assert "carry-over" in report.issues[0].detail


def test_validate_corpus_keyword_match_is_case_insensitive(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["ANNUAL LEAVE"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("annual leave: 25 days", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok


def test_validate_corpus_flags_negative_case_with_refs(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset, [_case("c1", "doc_hr_leave", ["annual leave"], should_answer=False)]
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    report = validate_corpus(load_cases(dataset), corpus)

    assert [issue.kind for issue in report.issues] == ["negative_case_has_refs"]


def test_validate_corpus_lists_orphan_files(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("annual leave", encoding="utf-8")
    (corpus / "doc_unused.md").write_text("nobody references me", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok  # orphan 只是警告,不算 issue
    assert report.orphan_files == ["doc_unused.md"]


def test_collect_expected_doc_ids_dedupes(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset,
        [
            _case("c1", "doc_hr_leave", ["annual leave"]),
            _case("c2", "doc_hr_leave", ["entitlement"]),
            _case("c3", None, [], should_answer=False),
        ],
    )

    ids = collect_expected_doc_ids(load_cases(dataset))

    assert ids == {"doc_hr_leave"}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_eval_corpus.py -q
```

Expected: FAIL,`ModuleNotFoundError: No module named 'app.evals.corpus'`(或 ImportError)

- [ ] **Step 3: 实现 `app/evals/corpus.py`**

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.evals.datasets import EvalCase


@dataclass(slots=True, frozen=True)
class CorpusIssue:
    case_id: str
    doc_id: str
    kind: str  # "missing_doc" | "missing_keyword" | "negative_case_has_refs"
    detail: str


@dataclass(slots=True)
class CorpusReport:
    total_cases: int
    checked_docs: int = 0
    issues: list[CorpusIssue] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def collect_expected_doc_ids(cases: Sequence[EvalCase]) -> set[str]:
    ids: set[str] = set()
    for case in cases:
        ids.update(case.expected_doc_ids)
    return ids


def validate_corpus(cases: Sequence[EvalCase], corpus_dir: Path) -> CorpusReport:
    """Check that every positive case has an aligned corpus document.

    Rules:
      * each expected_doc_id must exist as <corpus_dir>/<doc_id>.md
      * each expected keyword must appear (case-insensitively) in the
        combined text of the case's documents
      * negative cases must not reference docs or keywords
    Orphan corpus files (not referenced by any case) are reported as
    warnings, not issues.
    """
    report = CorpusReport(total_cases=len(cases))
    cache: dict[str, str | None] = {}

    def _load(doc_id: str) -> str | None:
        if doc_id not in cache:
            path = corpus_dir / f"{doc_id}.md"
            cache[doc_id] = (
                path.read_text(encoding="utf-8").lower() if path.is_file() else None
            )
        return cache[doc_id]

    for case in cases:
        if not case.should_answer:
            if case.expected_doc_ids or case.expected_keywords:
                report.issues.append(
                    CorpusIssue(
                        case_id=case.id,
                        doc_id=",".join(case.expected_doc_ids) or "-",
                        kind="negative_case_has_refs",
                        detail="negative case must have empty expected_doc_ids/expected_keywords",
                    )
                )
            continue
        loaded: list[str] = []
        for doc_id in case.expected_doc_ids:
            text = _load(doc_id)
            if text is None:
                report.issues.append(
                    CorpusIssue(
                        case_id=case.id,
                        doc_id=doc_id,
                        kind="missing_doc",
                        detail=f"{doc_id}.md not found in {corpus_dir}",
                    )
                )
            else:
                loaded.append(text)
        if not loaded:
            continue
        combined = "\n".join(loaded)
        for keyword in case.expected_keywords:
            if keyword.lower() not in combined:
                report.issues.append(
                    CorpusIssue(
                        case_id=case.id,
                        doc_id=",".join(case.expected_doc_ids),
                        kind="missing_keyword",
                        detail=f"keyword '{keyword}' not found",
                    )
                )
    report.checked_docs = sum(1 for text in cache.values() if text is not None)
    referenced = collect_expected_doc_ids(cases)
    if corpus_dir.is_dir():
        report.orphan_files = sorted(
            path.name for path in corpus_dir.glob("*.md") if path.stem not in referenced
        )
    return report
```

- [ ] **Step 4: 跑测试确认通过**

```bash
uv run pytest tests/test_eval_corpus.py -q
```

Expected: `7 passed`

- [ ] **Step 5: 静态检查**

```bash
uv run ruff check app/evals/corpus.py tests/test_eval_corpus.py && uv run mypy app/evals/corpus.py
```

Expected: 无报错

- [ ] **Step 6: Commit**

```bash
git add app/evals/corpus.py tests/test_eval_corpus.py
git commit -m "feat: eval 语料对齐校验模块"
```

---

### Task 3: 校验 CLI `scripts/eval/validate_corpus.py`

**Files:**
- Create: `scripts/eval/validate_corpus.py`

- [ ] **Step 1: 写脚本**

```python
"""Validate that the synthetic eval corpus aligns with the eval datasets.

For every positive eval case, checks that data/eval/corpus/<doc_id>.md
exists and contains all expected keywords (case-insensitively).

Usage:
    uv run python scripts/eval/validate_corpus.py
    uv run python scripts/eval/validate_corpus.py --list-missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evals.corpus import validate_corpus
from app.evals.datasets import EvalCase, load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate eval corpus/dataset alignment")
    parser.add_argument("--datasets-dir", default="data/eval/datasets")
    parser.add_argument("--corpus-dir", default="data/eval/corpus")
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="only print doc_ids that still have no corpus file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases: list[EvalCase] = []
    for path in sorted(Path(args.datasets_dir).glob("*.jsonl")):
        cases.extend(load_cases(path))
    report = validate_corpus(cases, Path(args.corpus_dir))
    if args.list_missing:
        missing = sorted({issue.doc_id for issue in report.issues if issue.kind == "missing_doc"})
        print("\n".join(missing))
        return 0
    for issue in report.issues:
        print(f"[{issue.kind}] case={issue.case_id} doc={issue.doc_id}: {issue.detail}")
    for orphan in report.orphan_files:
        print(f"[orphan] {orphan}")
    print(
        f"cases={report.total_cases} docs_checked={report.checked_docs} "
        f"issues={len(report.issues)}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行验证(语料目录尚空)**

```bash
mkdir -p data/eval/corpus
uv run python scripts/eval/validate_corpus.py; echo "exit=$?"
```

Expected: 104 行 `[missing_doc] ...`,最后 `cases=110 docs_checked=0 issues=104` 和 `exit=1`

- [ ] **Step 3: Commit**

```bash
git add scripts/eval/validate_corpus.py
git commit -m "feat: eval 语料校验 CLI"
```

---

## 语料撰写约定(Task 4-7 通用)

**虚构公司:Meridian Analytics**(数据分析 SaaS 公司)。每篇文档:

1. 文件名 `data/eval/corpus/<doc_id>.md`,一篇对应一个 doc_id
2. 英文,300-600 词,markdown 标题结构;HR/财务/合规写成公司政策文档,tech/api 写成内部工程手册(design doc / runbook)风格
3. **必含该 doc_id 对应用例的所有 `expected_keywords`**(短语原样出现,大小写不限;校验脚本会逐条检查)
4. **必须能明确回答对应 `question`**——包含具体数字/事实(如 "25 working days"),不能含糊
5. **不得回答任何负例问题**。6 条负例问题用此命令打印,写作时避开这些主题:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('data/eval/datasets/*.jsonl')):
    for line in open(f):
        c = json.loads(line)
        if not c['should_answer']:
            print(c['id'], '-', c['question'])
"
```

**每批的工作清单**(doc_id | 问题 | 关键词)用此命令生成,把 `doc_hr_` 换成对应批次前缀:

```bash
python3 -c "
import json, glob, sys
prefixes = sys.argv[1:]
for f in sorted(glob.glob('data/eval/datasets/*.jsonl')):
    for line in open(f):
        c = json.loads(line)
        for d in c['expected_doc_ids']:
            if any(d.startswith(p) for p in prefixes):
                print(d, '|', c['question'], '|', ', '.join(c['expected_keywords']))
" doc_hr_
```

**完整示例**(`data/eval/corpus/doc_hr_leave.md`,对应 rag-001:问题 "What is the maximum annual leave entitlement for employees with more than 5 years of service?",关键词 `annual leave / entitlement / years of service`):

```markdown
# Meridian Analytics — Annual Leave Policy

**Document ID:** doc_hr_leave
**Owner:** People Operations
**Last updated:** 2026-01-15

## Purpose

This policy defines the annual leave entitlement for all full-time
employees of Meridian Analytics.

## Entitlement

| Years of service | Annual leave entitlement |
|---|---|
| 0–2 years | 15 working days |
| 3–5 years | 20 working days |
| More than 5 years | 25 working days (maximum) |

The maximum annual leave entitlement is **25 working days** per calendar
year, granted to employees with more than 5 years of service.

## Accrual and carry-over

Leave accrues monthly at 1/12 of the annual entitlement. Up to 5 unused
days may be carried over into the first quarter of the following year;
carried-over days expire on 31 March.

## How to request leave

Submit requests in the HR portal at least 10 working days in advance.
Requests longer than 10 consecutive days require director approval.
Public holidays observed by the employee's registered office do not
count against the entitlement.
```

---

### Task 4: HR 批语料(11 篇)

**Files:**
- Create: `data/eval/corpus/doc_hr_{benefits,bonus,compensation,leave,notice,parental,pto,remote,retirement,sick,tuition}.md`

- [ ] **Step 1: 生成 HR 批工作清单**(用上方命令,前缀 `doc_hr_`)
- [ ] **Step 2: 按约定撰写 11 篇文档**(`doc_hr_leave.md` 直接用上方示例)
- [ ] **Step 3: 校验**

```bash
uv run python scripts/eval/validate_corpus.py | grep -E "doc_hr_|issues="
```

Expected: 无任何 `missing_keyword`/`missing_doc` 行含 `doc_hr_`;末行 `issues=93`(104−11)。若有 missing_keyword,修文档重跑至消失。

- [ ] **Step 4: Commit**

```bash
git add data/eval/corpus && git commit -m "feat: eval 合成语料 HR 批(11 篇)"
```

---

### Task 5: IT/合规/API/产品/组织批语料(20 篇)

**Files:**
- Create: `doc_it_*`(3)、`doc_compliance_*`(7)、`doc_api_*`(5)、`doc_product_*`(4)、`doc_org_structure`(1),均在 `data/eval/corpus/`

- [ ] **Step 1: 生成工作清单**(前缀 `doc_it_ doc_compliance_ doc_api_ doc_product_ doc_org_`)
- [ ] **Step 2: 按约定撰写 20 篇**(IT/合规为政策文档;API 为 NeuralFlow 风格的平台 API 手册;产品/组织为内部说明文档)
- [ ] **Step 3: 校验**:同 Task 4 方式,Expected 末行 `issues=73`
- [ ] **Step 4: Commit**:`git add data/eval/corpus && git commit -m "feat: eval 合成语料 IT/合规/API/产品批(20 篇)"`

---

### Task 6: 财务批语料(33 篇)

**Files:**
- Create: `doc_fin_*`(29)、`doc_finance_*`(4),均在 `data/eval/corpus/`

- [ ] **Step 1: 生成工作清单**(前缀 `doc_fin_ doc_finance_`)
- [ ] **Step 2: 按约定撰写 33 篇**(US GAAP/税务/SEC 主题写成 Meridian 财务部内部会计备忘录,注意负例 fin-027 的问题不得被任何文档回答)
- [ ] **Step 3: 校验**:Expected 末行 `issues=40`
- [ ] **Step 4: Commit**:`git add data/eval/corpus && git commit -m "feat: eval 合成语料财务批(33 篇)"`

---

### Task 7: 技术批语料(40 篇)+ 全量校验

**Files:**
- Create: `doc_tech_*`(40),在 `data/eval/corpus/`

- [ ] **Step 1: 生成工作清单**(前缀 `doc_tech_`)
- [ ] **Step 2: 按约定撰写 40 篇**(工程手册风格,注意负例 tech-027 不得被回答)
- [ ] **Step 3: 全量校验**

```bash
uv run python scripts/eval/validate_corpus.py; echo "exit=$?"
```

Expected: `cases=110 docs_checked=104 issues=0`、无 `[orphan]` 行、`exit=0`

- [ ] **Step 4: Commit**:`git add data/eval/corpus && git commit -m "feat: eval 合成语料技术批(40 篇),110 用例全对齐"`

---

### Task 8: Pipeline 透传文档 metadata(TDD)

**Files:**
- Modify: `app/ingestion/pipeline.py:128`
- Test: `tests/test_ingestion_pipeline.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_ingestion_pipeline.py` 末尾追加(文件顶部已有所需 import 与 Stub 类):

```python
@pytest.mark.asyncio
async def test_ingestion_pipeline_merges_document_metadata(monkeypatch, tmp_path: Path) -> None:
    from app.db.session import SessionLocal, init_db
    from app.documents.enums import DocumentStatus
    from app.documents.repository import DocumentRepository
    from app.documents.schemas import DocumentCreate

    init_db()
    db = SessionLocal()
    repo = DocumentRepository(db)
    file_path = tmp_path / "leave.md"
    file_path.write_text("# Annual Leave\n员工请假需要提前申请。", encoding="utf-8")

    document_id = f"doc_test_meta_{uuid4().hex[:8]}"
    repo.create_document(
        DocumentCreate(
            tenant_id="public",
            owner_user_id="tester",
            title="Leave Policy",
            filename="leave.md",
            original_filename="leave.md",
            file_type="md",
            mime_type="text/markdown",
            size_bytes=file_path.stat().st_size,
            storage_path=str(file_path),
            checksum_sha256="abc123",
            metadata_json={"canonical_doc_id": "doc_hr_leave", "eval_corpus": True},
            source_info_json={},
        ),
        document_id=document_id,
    )
    repo.update_status("public", document_id, DocumentStatus.QUEUED)
    db.close()

    stub_store = StubStore()
    monkeypatch.setattr("app.ingestion.pipeline.ParserFactory.create", lambda path: StubParser())
    monkeypatch.setattr(
        "app.ingestion.pipeline.ChromaDocumentStore", lambda *args, **kwargs: DummyStore()
    )
    pipeline = IngestionPipeline()
    pipeline.embedding_service = StubEmbeddingService()
    pipeline.store = stub_store

    await pipeline.run(
        tenant_id="public", document_id=document_id, embedding_model="test-embedding"
    )

    metadata = stub_store.upserts[0][0]["metadata"]
    assert metadata["canonical_doc_id"] == "doc_hr_leave"
    assert metadata["document_id"] == document_id  # 系统字段不被文档 metadata 覆盖
```

- [ ] **Step 2: 跑测试确认失败**

```bash
uv run pytest tests/test_ingestion_pipeline.py::test_ingestion_pipeline_merges_document_metadata -q
```

Expected: FAIL,`KeyError: 'canonical_doc_id'`

- [ ] **Step 3: 修改 `app/ingestion/pipeline.py`**

在 `pipeline.py:128` 的 `chunk.metadata.update({...系统字段...})` 之前插入一行,即把:

```python
            for chunk, vector in zip(chunks, vectors, strict=False):
                chunk.embedding = vector
                chunk.metadata.update(
                    {
                        "tenant_id": tenant_id,
```

改为:

```python
            for chunk, vector in zip(chunks, vectors, strict=False):
                chunk.embedding = vector
                # carry document-level metadata (e.g. canonical_doc_id) into chunks;
                # system keys below take precedence
                chunk.metadata.update(record.metadata_json or {})
                chunk.metadata.update(
                    {
                        "tenant_id": tenant_id,
```

- [ ] **Step 4: 跑测试确认通过 + 回归**

```bash
uv run pytest tests/test_ingestion_pipeline.py -q
```

Expected: 全部 passed(含原有 2 个测试)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/pipeline.py tests/test_ingestion_pipeline.py
git commit -m "feat: ingestion 透传文档级 metadata 至 chunk(canonical_doc_id)"
```

---

### Task 9: Seed 脚本 `scripts/eval/seed_corpus.py`

**Files:**
- Create: `scripts/eval/seed_corpus.py`

- [ ] **Step 1: 写脚本**

```python
"""Seed the synthetic eval corpus through the ingestion pipeline.

Each data/eval/corpus/<doc_id>.md becomes one document whose canonical
doc_id (= file stem) is carried into chunk metadata for citation
matching during evals. Idempotent: already-seeded docs are skipped
unless --force is given.

Usage:
    uv run python scripts/eval/seed_corpus.py
    uv run python scripts/eval/seed_corpus.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal, init_db
from app.documents.enums import DocumentFileType, DocumentStatus
from app.documents.repository import DocumentRepository
from app.documents.schemas import DocumentCreate
from app.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed eval corpus via ingestion pipeline")
    parser.add_argument("--corpus-dir", default="data/eval/corpus")
    parser.add_argument("--tenant", default=None, help="defaults to settings.tenant_default_id")
    parser.add_argument("--force", action="store_true", help="re-ingest existing documents")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    settings = get_settings()
    tenant_id = args.tenant or settings.tenant_default_id
    corpus_dir = Path(args.corpus_dir).resolve()
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        print(f"no corpus files found in {corpus_dir}")
        return 1

    init_db()
    pipeline = IngestionPipeline()
    seeded = skipped = failed = 0
    for path in files:
        canonical_id = path.stem
        document_id = f"eval_{canonical_id}"
        db = SessionLocal()
        try:
            repo = DocumentRepository(db)
            existing = repo.get_document(tenant_id=tenant_id, document_id=document_id)
            if existing is not None and not args.force:
                skipped += 1
                continue
            if existing is None:
                content = path.read_bytes()
                repo.create_document(
                    DocumentCreate(
                        tenant_id=tenant_id,
                        owner_user_id="eval-seeder",
                        title=canonical_id,
                        filename=path.name,
                        original_filename=path.name,
                        file_type=DocumentFileType.MARKDOWN.value,
                        mime_type="text/markdown",
                        size_bytes=len(content),
                        storage_path=str(path),
                        checksum_sha256=hashlib.sha256(content).hexdigest(),
                        metadata_json={"canonical_doc_id": canonical_id, "eval_corpus": True},
                        source_info_json={"seeded_by": "seed_corpus.py"},
                    ),
                    document_id=document_id,
                )
            repo.update_status(tenant_id, document_id, DocumentStatus.QUEUED)
        finally:
            db.close()
        try:
            result = await pipeline.run(
                tenant_id=tenant_id,
                document_id=document_id,
                embedding_model=settings.embedding_model,
            )
            seeded += 1
            print(f"seeded {canonical_id}: {result['chunk_count']} chunks")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAILED {canonical_id}: {exc}")
    print(f"done: seeded={seeded} skipped={skipped} failed={failed} total={len(files)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
```

- [ ] **Step 2: 运行 seed**(确保环境已配置真实 embedding API key,否则会得到假向量)

```bash
uv run python scripts/eval/seed_corpus.py
```

Expected: 104 行 `seeded doc_...: N chunks`,末行 `done: seeded=104 skipped=0 failed=0 total=104`

- [ ] **Step 3: 幂等性验证(再跑一次)**

```bash
uv run python scripts/eval/seed_corpus.py
```

Expected: `done: seeded=0 skipped=104 failed=0 total=104`

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/seed_corpus.py
git commit -m "feat: eval 语料 seed 脚本(幂等,canonical_doc_id 入 chunk metadata)"
```

---

### Task 10: 冒烟脚本 `scripts/eval/smoke_retrieval.py`

**Files:**
- Create: `scripts/eval/smoke_retrieval.py`

- [ ] **Step 1: 写脚本**

```python
"""Retrieval smoke test for the seeded eval corpus.

For every positive eval case, runs top-k retrieval and checks whether
any result chunk carries the expected canonical_doc_id. Exits non-zero
below 90% hit rate.

Usage:
    uv run python scripts/eval/smoke_retrieval.py [--top-k 5]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal, init_db
from app.documents.repository import DocumentRepository
from app.evals.datasets import EvalCase, load_cases
from app.retrieval.schemas import RetrievalRequest
from app.retrieval.service import RetrievalService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval smoke test for eval corpus")
    parser.add_argument("--datasets-dir", default="data/eval/datasets")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tenant", default=None, help="defaults to settings.tenant_default_id")
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    settings = get_settings()
    tenant_id = args.tenant or settings.tenant_default_id
    cases: list[EvalCase] = []
    for path in sorted(Path(args.datasets_dir).glob("*.jsonl")):
        cases.extend(load_cases(path))
    positives = [case for case in cases if case.should_answer]

    init_db()
    db = SessionLocal()
    hits = 0
    misses: list[str] = []
    try:
        service = RetrievalService(document_repo=DocumentRepository(db))
        for case in positives:
            response = await service.search(
                tenant_id, RetrievalRequest(query=case.question, top_k=args.top_k)
            )
            found = {result.metadata.get("canonical_doc_id") for result in response.results}
            if found & set(case.expected_doc_ids):
                hits += 1
            else:
                got = sorted(str(item) for item in found if item)
                misses.append(f"{case.id} expected={list(case.expected_doc_ids)} got={got}")
    finally:
        db.close()

    for line in misses:
        print("MISS", line)
    rate = hits / len(positives) if positives else 0.0
    print(f"hit@{args.top_k}: {hits}/{len(positives)} = {rate:.1%}")
    return 0 if rate >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
```

- [ ] **Step 2: 运行冒烟**

```bash
uv run python scripts/eval/smoke_retrieval.py; echo "exit=$?"
```

Expected: `hit@5: ≥94/104 = ≥90%`、`exit=0`。首次运行 CrossEncoderReranker 可能下载模型,耐心等待。

- [ ] **Step 3: 若命中率 < 90%**:逐条看 MISS 输出,常见原因是文档太泛(与问题词汇重叠不足)——把该 doc 的用词向问题贴近、补充具体事实后,`seed_corpus.py --force` 重灌该批,再跑冒烟。循环至 ≥90%。

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/smoke_retrieval.py
git commit -m "feat: eval 语料检索冒烟脚本(hit@5 门槛 90%)"
```

---

### Task 11: 全量本地验证 + PR

- [ ] **Step 1: CLAUDE.md 完整本地验证**

```bash
uv run pre-commit run --all-files
uv run ruff check .
uv run mypy app tests worker.py
uv run pytest -q
cd frontend && npm run lint && npm run typecheck && npm test && cd ..
```

Expected: 全部通过(pre-commit 若改写了文件,把改动 `git add` 后重跑至干净)

- [ ] **Step 2: 推送并建 PR**

```bash
git push -u origin feat/eval-corpus
gh pr create --title "feat: eval 合成语料与 seed 闭环(Phase 0)" --body "$(cat <<'EOF'
## Summary
- 104 篇合成企业语料(Meridian Analytics),与 110 条评测用例全量对齐
- 语料↔用例对齐校验模块 + CLI(missing_doc / missing_keyword / 负例约束)
- ingestion pipeline 透传文档级 metadata(canonical_doc_id)至 chunk
- seed 脚本(幂等)+ 检索冒烟脚本(hit@5 ≥90%)

## 冒烟结果
<粘贴 smoke_retrieval.py 输出>

## 对应设计
docs/superpowers/specs/2026-07-19-eval-driven-rag-loop-design.md 的 Phase 0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: 等 CI 全绿后 squash merge**(遵循分支保护规则,不 bypass)

---

## 计划自检记录

- **Spec 覆盖**:Phase 0 四项(盘点 → Task 3 `--list-missing`;撰写 → Task 4-7;seed → Task 8-9;冒烟 → Task 10)全覆盖;负例反向约束落在撰写约定第 5 条 + 校验器 `negative_case_has_refs` 规则。
- **数量修正**:spec 预估 30-40 篇,实际盘点为 104 篇(每条正例一个唯一 doc_id),spec 已同步更新。
- **关键签名已核实**:`LocalDocumentStorage.get_local_path` 原样返回本地路径(storage.py:80);`DocumentFileType.MARKDOWN.value == "md"`;`RetrievalService` 结果 metadata 直通 ChromaDB;chromadb 1.5.7 已随 `ml` extra 安装(跨进程持久化成立)。
