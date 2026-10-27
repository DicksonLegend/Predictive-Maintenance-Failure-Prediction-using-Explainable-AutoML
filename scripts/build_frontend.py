"""Frontend Build Automation Script.

Attempts to compile the React Single Page Application into app/frontend/dist.
Falls back safely to app/static_fallback/index.html if node/npm is absent or errors.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
STATIC_FALLBACK = PROJECT_ROOT / "app" / "static_fallback" / "index.html"


def build_frontend() -> str:
    """Installs dependencies and compiles React app with Vite.

    Returns:
        str: 'react' if dist build succeeded, else 'static_fallback'.
    """
    print("=" * 60)
    print("FRONTEND BUILD PIPELINE")
    print("=" * 60)

    # 1. Check for node and npm
    node_bin = shutil.which("node")
    npm_bin = shutil.which("npm") or shutil.which("npm.cmd")

    if not node_bin or not npm_bin:
        print("[WARNING] Node.js or npm not detected in system PATH.")
        print(f"[INFO] Using static fallback frontend at: {STATIC_FALLBACK}")
        return "static_fallback"

    print(f"[INFO] Node detected: {node_bin}")
    print(f"[INFO] npm detected: {npm_bin}")

    try:
        # 2. Run npm install
        print(f"[INFO] Running 'npm install' in {FRONTEND_DIR}...")
        install_cmd = [npm_bin, "install", "--prefer-offline", "--no-audit", "--no-fund"]
        ret = subprocess.run(install_cmd, cwd=str(FRONTEND_DIR), capture_output=True, text=True, timeout=180)
        if ret.returncode != 0:
            print(f"[WARNING] npm install failed (exit {ret.returncode}):\n{ret.stderr[:400]}")
            print(f"[INFO] Falling back to static frontend at: {STATIC_FALLBACK}")
            return "static_fallback"

        # 3. Run npm run build
        print(f"[INFO] Running 'npm run build'...")
        build_cmd = [npm_bin, "run", "build"]
        ret_b = subprocess.run(build_cmd, cwd=str(FRONTEND_DIR), capture_output=True, text=True, timeout=120)
        if ret_b.returncode != 0:
            print(f"[WARNING] npm run build failed (exit {ret_b.returncode}):\n{ret_b.stderr[:400]}")
            print(f"[INFO] Falling back to static frontend at: {STATIC_FALLBACK}")
            return "static_fallback"

        # 4. Check for dist/index.html
        if (DIST_DIR / "index.html").exists():
            print("[SUCCESS] React Single Page Application successfully compiled to dist!")
            return "react"
        else:
            print("[WARNING] dist/index.html not generated after build.")
            return "static_fallback"

    except Exception as e:
        print(f"[WARNING] Exception during frontend compilation: {e}")
        return "static_fallback"


if __name__ == "__main__":
    mode = build_frontend()
    print(f"[RESULT] Selected Frontend Mode: {mode}")
    if mode == "react":
        sys.exit(0)
    else:
        # Fallback is acceptable according to spec, but exit 0 so pipeline continues
        sys.exit(0)
