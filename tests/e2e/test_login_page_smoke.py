import os

import pytest

playwright = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright.sync_playwright


def test_login_page_smoke() -> None:
    base_url = os.getenv("E2E_BASE_URL", "http://127.0.0.1:8501")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(2500)
        content = page.content().lower()
        browser.close()

    assert "login" in content
    assert "register" in content or "stock anomaly detector" in content
