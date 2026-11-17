"""A hostile company name must not break or inject code into the downloadable bundle."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from crucible.build_bundle import clean_company_name
from crucible.server.app import create_app

HOSTILE = 'Evil"""\nimport os; os.system("id") #{x}\\ <img src=x onerror=alert(1)>'


def test_clean_company_name_strips_injection_characters() -> None:
    cleaned = clean_company_name(HOSTILE)
    assert not set(cleaned) & set('"<>{}\\;#\n')
    assert clean_company_name("   ") == "Apex Dynamics"
    assert clean_company_name(None) == "Apex Dynamics"
    assert clean_company_name("Meridian Labs & Co.") == "Meridian Labs & Co."
    assert clean_company_name("O'Neil (R&D)") == "O'Neil (R&D)"
    assert len(clean_company_name("x" * 500)) == 60


def test_bundle_with_hostile_company_name_stays_valid_python(tmp_path: Path) -> None:
    client = TestClient(create_app(run_dir=tmp_path / "ui_state"))
    resp = client.get("/api/build/bundle.zip", params={"company_name": HOSTILE})
    assert resp.status_code == 200
    assert '"' not in resp.headers["content-disposition"].split("filename=")[-1].strip('"')
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in ("scripts/run_company_app.py", "scripts/client_query_llm.py"):
            code = zf.read(name).decode()
            compile(code, name, "exec")
            assert 'os.system("id")' not in code and "<img" not in code
