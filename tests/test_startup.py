import os
import subprocess
import sys
from pathlib import Path


def test_import_app_without_api_key_and_without_startup_network_calls():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("GROQ_API_KEY", None)

    cmd = [
        sys.executable,
        "-c",
        "import os; os.environ.pop('GROQ_API_KEY', None); import app; print('startup-ok')",
    ]
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "startup-ok" in result.stdout
