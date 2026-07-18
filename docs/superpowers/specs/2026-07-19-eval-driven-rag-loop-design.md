# Eval 驱动的 RAG 质量闭环 — 设计文档

- **日期**: 2026-07-19
- **状态**: 已与用户对齐,待实施
- **定位**: NeuralFlow 当前阶段为求职作品集(目标岗位:AI/LLM 应用工程师)。本设计是两阶段规划的第一阶段("A → B":先建评测地基,再做深度研究 Agent),预期投入 3-4 周。

## 1. 背景与问题

NeuralFlow 的 eval 基础设施"有框架、无闭环":

| 现状 | 证据 |
|---|---|
| eval API 从不调用真实 RAG pipeline | `app/api/eval.py:57-63` 中 `retrieve_fn` 返回空列表,`answer_fn` 硬编码返回 `"eval answer stub"` |
| CLI 只能跑 mock | `app/evals/cli.py` 使用 `_make_mock_retrieve()` / `_make_mock_answer()` |
| 评测数据集"有题无卷" | `data/eval/datasets/` 下 110 条用例引用语义化 doc_id(如 `doc_hr_leave`),但仓库中不存在对应语料文档 |
| LLM-as-Judge 已实现但未接入 eval 循环 | `app/rag/answer_evaluator.py` 完整可用,但 API 端点不传 `answer_eval_fn` |
| CI eval-gate 是回归门而非质量门 | `eval-gate.yml` 只跑 mock 数据的 `test_eval_regression.py`,无分数门槛 |
| 无成本追踪 | litellm 响应中的 token usage 未被提取记录 |

结果:项目无法回答"这个 RAG 系统好不好、每项技术带来多少提升"——而这正是 AI/LLM 应用工程师面试中最有分量的叙事。

## 2. 目标与成功标准

**目标**:把 eval 做成真闭环,产出可量化的 RAG 迭代叙事(baseline → 逐项改进 → 数据证明)。

**成功标准**:

1. eval API/CLI 跑真实 pipeline(真检索 + 真生成 + 真 LLM-as-Judge),不再有 stub
2. 一套与 110 条用例精确对齐的合成企业语料,一条命令 seed 入库
3. ≥5 组消融实验有 before/after 对比报告(markdown + 前端展示)
4. 每次 eval run 记录实验配置快照与 token 成本,结果可复现、可对比

## 3. 关键决策(已与用户确认)

| 决策点 | 选择 | 理由 |
|---|---|---|
| 评测语料来源 | **合成企业语料**(LLM 撰写虚构公司的 HR/IT/财务/合规/技术文档) | 自包含、可控、与现有 110 条用例精确对齐;企业内部语料本不可公开,合成是行业常态 |
| LLM 资源 | **云端 API 为主**(gpt-4o-mini,已有预算) | 单轮 110 条 × 生成+Judge 成本约几十美分,可接受 |
| CI 中的真实 eval | **不进常规 CI**,只做手动/nightly workflow | 控制成本;常规 CI 保持零 LLM 调用 |

## 4. 闭环架构(5 层)

```
语料层 → 执行层 → 实验层 → 报告层 → 门槛层
```

| 层 | 内容 | 依托现有 |
|---|---|---|
| 语料层 | `data/eval/corpus/` 合成文档(经盘点为 104 篇:每条正例用例一个唯一 doc_id),覆盖 110 条用例的 `expected_doc_ids`;seed 脚本走现有 ingestion 管线,metadata 携带 canonical doc_id 供 citation 匹配 | `app/ingestion/` 全链路 |
| 执行层 | 替换 `app/api/eval.py` stub:注入真实 `retrieve_fn`(HybridRetrievalService)、`answer_fn`(与线上 chat 相同的 RAG 生成路径,advanced 特性由实验 overlay 控制)、`answer_eval_fn`(`answer_evaluator.py`);CLI 加 `--live/--mock` 开关;EvalRun 落库记录配置快照与成本 | `app/evals/runner.py` 骨架 |
| 实验层 | 实验 overlay 机制(一个 JSON/YAML 定义一组 settings 覆盖 = 一个实验组);消融矩阵:baseline纯向量 → +hybrid → +reranker → +multi_query → +HyDE → chunking 对比;`compare` 命令跑真实 A/B | `comparison.py`、`config.py` 的 10+ 开关 |
| 报告层 | 对比报告落盘 `docs/eval-reports/`;前端 eval 页新增 A/B 对比视图;README 量化叙事 | 前端 eval 组件、雷达图 |
| 门槛层 | 新增 `eval-live.yml`(手动 + nightly,API key 走 secret),显式分数门槛;现有 `eval-gate.yml` mock 回归门不动 | `.github/workflows/` |

**两个横切亮点**:

- **Judge 结果缓存**:按 `(question, answer, judge_model)` 哈希缓存打分,复用现有 Redis 缓存设施,重复实验不重复花钱
- **成本追踪**:从 litellm 响应提取 token usage 记入 EvalRun("这轮评测花了 $0.43"本身即工程成熟度展示)

## 5. 分阶段工作分解

每阶段一个 PR(分支 → PR → CI 全绿 → squash merge),每个 PR 合入后系统保持可用。

### Phase 0 · 语料构建与 seed(~3-5 天)

- 盘点 110 条用例的 `expected_doc_ids`,去重得出待写文档清单
- LLM 撰写合成企业文档(markdown),对齐约束:
  - positive 用例:`expected_keywords` 必须出现在对应文档中,问题可被该文档回答
  - negative 用例(`should_answer: false`):语料中**不得**含有其答案
- `scripts/eval/seed_corpus.py`:走 ingestion 管线摄入,metadata 带 canonical doc_id
- **验收**:冒烟脚本确认每条 positive 用例的目标文档已入库且可被检索命中

### Phase 1 · 接通真实闭环(~1 周)

- 替换 `app/api/eval.py:60` stub(检索/生成/Judge 三个注入点)
- `cli.py` 加 `--live / --mock` 开关(mock 保留给回归测试)
- `EvalRunORM` 增加字段:RAG 配置快照(JSON)、token 用量/成本
- Judge 结果 Redis 缓存
- **验收**:一条命令跑通 110 条用例,产出第一份真实 baseline 报告

### Phase 2 · 消融实验矩阵(~1 周)

- 实验 overlay 机制(settings 覆盖集)
- 消融序列:`baseline(纯向量 top5)` → `+hybrid` → `+reranker` → `+multi_query` → `+HyDE` → `chunking 策略对比`
- `compare` 命令改为真实 A/B,输出各指标 delta
- **验收**:≥5 组实验各有一份 markdown 对比报告

### Phase 3 · 报告与前端呈现(~1 周)

- 对比报告落盘 `docs/eval-reports/`
- 前端 eval 页新增 A/B 对比视图(两次 run 并排、指标 delta 高亮,复用 metric-card/雷达图组件风格)
- README 更新为量化叙事(改进曲线表格)
- **验收**:前端可选任意两次 run 对比;新组件补 Vitest 测试

### Phase 4 · CI 质量门槛(~2-3 天)

- `eval-live.yml`:手动触发 + nightly,API key 走 GitHub secret
- 显式门槛:hit rate、faithfulness 低于基线阈值即 fail
- **验收**:手动触发一次全绿;故意调低阈值验证会红

## 6. 测试与错误处理策略

- 新逻辑单元测试全部用 mock LLM(沿用 `tests/test_eval_regression.py` 模式),常规 CI 零 LLM 成本
- 单条用例 LLM 调用失败:标记该 case 为 error,不中断整轮 eval;生成侧沿用现有 offline fallback 链(`app/core/llm.py`)
- mypy / ruff / pre-commit 全程保持通过;推送前按 CLAUDE.md 跑完整本地验证

## 7. 非目标(YAGNI)

- Vue 3 前端迁移(`docs/superpowers/plans/2026-05-24-vue-frontend-migration.md` 搁置)
- ACL/细粒度权限、Alembic 迁移框架
- 中文评测用例扩展(现有 110 条英文用例已够讲故事;后续可作为增量)
- 公开学术基准集(HotpotQA 等)对接
- Multimodal/OCR 管线启用(候选方向 C,另行规划)

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 合成语料与用例对齐有遗漏(关键词缺失/负例泄漏答案) | Phase 0 写自动校验脚本:逐条用例检查 keywords 在目标文档中出现;负例做反向检索抽查 |
| 消融实验结果不显著甚至倒退 | 这本身就是有效叙事("HyDE 在本语料上无收益,已数据驱动地关闭");报告如实记录 |
| Judge 打分不稳定(同输入不同分) | Judge 缓存保证同一 (q, a) 只打一次分;必要时 temperature=0 |
| nightly workflow 消耗预算 | 门槛 workflow 默认只跑 50 条核心子集,全量留给手动触发 |

## 9. 后续

本设计获批后,进入 writing-plans 流程产出逐步实施计划。第二阶段(深度研究 Agent 场景)另行 brainstorm,其每项 Agent 能力改进将复用本阶段建成的评测体系来证明价值。
