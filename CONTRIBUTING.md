# 贡献指南

## 分支规范

- `main` — 保护分支，禁止直接推送
- `feat/<name>` — 新功能
- `fix/<name>` — 修复
- `chore/<name>` — 构建、CI、重构、文档

## 工作流程

```bash
# 1. 从 main 创建分支
git checkout main
git pull
git checkout -b feat/my-feature

# 2. 开发和提交
git add <files>
git commit -m "feat: what changed"

# 3. 推送并创建 PR
git push -u origin feat/my-feature
```

在 GitHub 上创建 Pull Request → CI 自动运行 → 所有检查通过后 squash merge 到 `main`。

## 本地验证

提交前运行以下命令确保全部通过：

```bash
# 后端
uv run ruff check .
uv run mypy app tests worker.py
uv run pytest -q

# 前端
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

## Commit 格式

使用语义化前缀：

- `feat:` 新功能
- `fix:` 修复
- `chore:` 杂项（CI、构建等）
- `test:` 测试
- `docs:` 文档
- `refactor:` 重构

## GitHub 保护设置

main 分支已启用以下规则：

- 必须通过 PR 合并
- 必须通过 CI 检查
- 分支必须保持最新
- 禁止管理员绕过
- 线性历史（squash merge）
