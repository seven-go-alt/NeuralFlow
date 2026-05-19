from __future__ import annotations

from app.config import Settings


class TestMultimodalSettingsDefaults:
    def setup_method(self) -> None:
        self.settings = Settings()

    def test_multimodal_defaults(self) -> None:
        assert self.settings.multimodal_enabled is False
        assert self.settings.vision_model == "gpt-4o"
        assert "Describe this image" in self.settings.vision_prompt_template
        assert self.settings.multimodal_max_images == 20
        assert self.settings.multimodal_max_image_size_mb == 5
        assert self.settings.multimodal_max_tables == 50

    def test_multimodal_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("MULTIMODAL_ENABLED", "true")
        monkeypatch.setenv("VISION_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("MULTIMODAL_MAX_IMAGES", "10")
        monkeypatch.setenv("MULTIMODAL_MAX_IMAGE_SIZE_MB", "2")
        monkeypatch.setenv("MULTIMODAL_MAX_TABLES", "25")

        s = Settings()
        assert s.multimodal_enabled is True
        assert s.vision_model == "gpt-4o-mini"
        assert s.multimodal_max_images == 10
        assert s.multimodal_max_image_size_mb == 2
        assert s.multimodal_max_tables == 25
