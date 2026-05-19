# NeuralFlow 项目约定

## Git 工作流

- **禁止直接推送 main 分支**。所有修改必须走 feature/fix/chore 分支 → PR → CI 全绿 → squash merge。
- 分支命名：`feat/<name>` / `fix/<name>` / `chore/<name>`
- 合入方式：squash merge，保持 main 线性历史。

## 提交前本地验证

必须全部通过才能推送：

```bash
# 1. Pre-commit 格式化（必须先跑，否则 CI 必报 ruff 格式错误）
uv run pre-commit run --all-files

# 2. 后端
uv run ruff check .
uv run mypy app tests worker.py
uv run pytest -q

# 3. 前端
cd frontend && npm run lint && npm run typecheck && npm test
```

## GitHub 分支保护

main 分支已启用 Ruleset：
- 必须通过 PR 合并
- 必须通过 CI status checks（quality + frontend）
- 分支必须保持 up to date
- 禁止 force push
- 禁止 bypass

## Commit 格式

语义化前缀：`feat:` `fix:` `chore:` `test:` `docs:` `refactor:`
