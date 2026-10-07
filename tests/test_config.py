"""Kiểm tra cấu hình bí mật: không có mặc định và không rò giá trị trong lỗi."""
import os
import types
import unittest
from unittest.mock import MagicMock, patch

from backend.core import config


class ConfigTests(unittest.TestCase):
    def test_missing_database_password_fails_without_default(self):
        with patch.dict(os.environ, {"POSTGRES_DB": "test", "POSTGRES_USER": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "POSTGRES_PASSWORD"):
                config.database_config()

    def test_missing_redis_password_fails_without_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "REDIS_PASSWORD"):
                config.redis_config()

    def test_loader_targets_project_root_and_preserves_exported_values(self):
        loader = MagicMock()
        with patch.dict("sys.modules", {"dotenv": types.SimpleNamespace(load_dotenv=loader)}):
            config.load_environment()
        loader.assert_called_once_with(config.PROJECT_ROOT / ".env", override=False)


if __name__ == "__main__":
    unittest.main()
