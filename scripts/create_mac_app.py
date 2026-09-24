"""Create a Finder launcher with the same icon as the Qwen application."""
from __future__ import annotations

from pathlib import Path
import plistlib
import shlex
import stat
import subprocess
import sys

from PIL import Image


def create() -> Path:
    root = Path(__file__).resolve().parents[1]
    bundle = root / "Qwen.app"
    contents = bundle / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    launcher = macos / "launch"
    command = root / "quickStart.command"
    launcher.write_text(
        "#!/bin/bash\nexec open -a Terminal " + shlex.quote(str(command)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump({
            "CFBundleDevelopmentRegion": "en",
            "CFBundleDisplayName": "Qwen",
            "CFBundleExecutable": "launch",
            "CFBundleIconFile": "Qwen.icns",
            "CFBundleIdentifier": "com.ido108.qwen",
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundleName": "Qwen",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0",
        }, handle)
    iconset = resources / "Qwen.iconset"
    iconset.mkdir(exist_ok=True)
    source = Image.open(root / "public" / "branding" / "icon-source.png").convert("RGBA")
    for size in (16, 32, 128, 256, 512):
        source.resize((size, size), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        source.resize((size * 2, size * 2), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
    try:
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(resources / "Qwen.icns")], check=True)
    finally:
        for item in iconset.iterdir():
            item.unlink()
        iconset.rmdir()
    return bundle


if __name__ == "__main__":
    if sys.platform != "darwin":
        raise SystemExit("This launcher is created on macOS.")
    print(f"Finder app ready: {create()}")
