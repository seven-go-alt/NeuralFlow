from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_copies_lockfile_before_uv_sync() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv.lock" in dockerfile
    assert "uv sync --frozen" in dockerfile



def test_env_example_exposes_openai_api_key_and_mcp_base_url_as_separate_entries() -> None:
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    active_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    assert any(line.startswith("OPENAI_API_KEY=") for line in active_lines)
    assert any(line.startswith("MCP_BASE_URL=") for line in active_lines)
    assert not any("OPENAI_API_KEY=" in line and "MCP_BASE_URL=" in line for line in active_lines)



def test_docker_compose_omits_obsolete_top_level_version_key() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    first_non_empty = next(line.strip() for line in compose.splitlines() if line.strip())
    assert not first_non_empty.startswith("version:")



def test_docker_compose_uses_example_env_for_api_and_worker() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "worker:" in compose
    assert compose.count("- .env.example") >= 2
