#!/usr/bin/env python3
"""
scripts/capture_studio_screenshots.py — Automated Cross-Browser Visual Screenshot
Capture for the WebGL2 Single-Cell Portable Studio.
Captures high-resolution 1920x1080 screenshots in Google Chrome and Mozilla Firefox.
"""

from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
STUDIO_HTML = ROOT / "studio" / "cellsnail_portable_studio.html"
SCREENSHOTS_DIR = ROOT / "studio" / "screenshots"

def find_chrome() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("google-chrome") or "",
        shutil.which("chrome") or "",
        shutil.which("chromium") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    raise FileNotFoundError("Could not find Google Chrome executable.")

def find_firefox() -> str:
    candidates = [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        shutil.which("firefox") or "",
    ]
    for f in candidates:
        if f and Path(f).exists():
            return str(Path(f).resolve())
    raise FileNotFoundError("Could not find Mozilla Firefox executable.")

def capture_chrome(chrome_exe: str, html_url: str, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome_exe,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1920,1080",
        "--virtual-time-budget=3000",
        f"--screenshot={output_png.resolve()}",
        html_url,
    ]
    print(f"[*] Running Chrome command: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if not output_png.exists() or output_png.stat().st_size == 0:
        raise RuntimeError(f"Chrome failed to capture screenshot. Stderr: {res.stderr.decode(errors='replace')}")
    print(f"[+] Chrome screenshot captured: {output_png} ({output_png.stat().st_size / 1024:.1f} KB)")

def capture_firefox(firefox_exe: str, html_url: str, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_profile:
        # Firefox headless screenshot command:
        # firefox -no-remote -profile <dir> -headless -screenshot <path> --window-size 1920,1080 <url>
        cmd = [
            firefox_exe,
            "--no-remote",
            "-profile",
            temp_profile,
            "-headless",
            "-screenshot",
            str(output_png.resolve()),
            "--window-size",
            "1920,1080",
            html_url,
        ]
        print(f"[*] Running Firefox command: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35)
        # Give Firefox a moment to flush file if needed
        time.sleep(1.0)
        
        # If output_png does not exist directly, check if firefox wrote screenshot.png in cwd
        if not output_png.exists() or output_png.stat().st_size == 0:
            local_ss = Path("screenshot.png")
            if local_ss.exists():
                shutil.move(str(local_ss), str(output_png))
        
        if not output_png.exists() or output_png.stat().st_size == 0:
            raise RuntimeError(f"Firefox failed to capture screenshot. Stderr: {res.stderr.decode(errors='replace')}")
        print(f"[+] Firefox screenshot captured: {output_png} ({output_png.stat().st_size / 1024:.1f} KB)")

def main() -> int:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not STUDIO_HTML.exists():
        raise FileNotFoundError(f"Studio HTML not found at {STUDIO_HTML}")

    html_uri = STUDIO_HTML.resolve().as_uri()
    print(f"[*] Studio HTML URL: {html_uri}")

    chrome_exe = find_chrome()
    firefox_exe = find_firefox()
    print(f"[*] Found Chrome: {chrome_exe}")
    print(f"[*] Found Firefox: {firefox_exe}")

    chrome_out = SCREENSHOTS_DIR / "chrome_studio.png"
    firefox_out = SCREENSHOTS_DIR / "firefox_studio.png"

    # 1. Capture in Chrome
    capture_chrome(chrome_exe, html_uri, chrome_out)

    # 2. Capture in Firefox
    capture_firefox(firefox_exe, html_uri, firefox_out)

    print("\n[+] All cross-browser studio screenshots successfully captured!")
    print(f"    - Chrome:  {chrome_out} ({chrome_out.stat().st_size / 1024:.1f} KB)")
    print(f"    - Firefox: {firefox_out} ({firefox_out.stat().st_size / 1024:.1f} KB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
