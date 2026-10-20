"""Build the lbvelox extension against a Velox build tree configured with
VELOX_BUILD_SHARED=ON and VELOX_BUILD_PYTHON_PACKAGE=ON.

Reuses the exact compile and link commands of PyVelox's plan_builder module (same
flags, same libvelox.so and test-utils archives), swapping in lbvelox.cpp. Also stages
PyVelox and writes a .pth into the active venv so both import without installation.

  uv run python velox_ext/build.py --velox-build ~/source/velox/_build/release-py
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = "velox/python/plan_builder/plan_builder.cpp"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd)[:300], "...", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--velox-build", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "build")
    args = ap.parse_args()
    vb = args.velox_build.expanduser().resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")

    entry = next(e for e in json.loads((vb / "compile_commands.json").read_text())
                 if e["file"].endswith(TEMPLATE))
    cc = shlex.split(entry["command"])
    obj = out / "lbvelox.o"
    cc = [a for a in cc if a != "-Dplan_builder_EXPORTS"]
    cc[cc.index("-o") + 1] = str(obj)
    cc[cc.index("-c") + 1] = str(HERE / "lbvelox.cpp")
    cc.insert(1, f"-I{HERE}")
    run(cc, Path(entry["directory"]))

    target = f"velox/python/plan_builder{suffix}"
    link = subprocess.run(["ninja", "-t", "commands", target], cwd=vb, check=True,
                          capture_output=True, text=True).stdout.strip().splitlines()[-1]
    link = shlex.split(next(seg for seg in link.split("&&") if " -o " in seg))
    so = out / f"lbvelox{suffix}"
    new = []
    for a in link:
        if a.endswith("plan_builder.cpp.o"):
            new.append(str(obj))
        elif a == target:
            new.append(str(so))
        elif a.startswith("-Wl,-soname,"):
            new.append(f"-Wl,-soname,lbvelox{suffix}")
        elif "velox/python/libvelox_py_" in a:
            continue
        elif a.endswith("/libgflags.a"):  # system gflags archive is not PIC
            new.append(a[:-2] + ".so")
        elif a.endswith("/libunwind.so"):  # its _Unwind_* would replace libgcc's: exceptions abort
            continue
        else:
            new.append(a)
    run(new, vb)

    # Stage PyVelox (modules + libvelox.so + python sources) next to the build tree.
    stage = vb / "pyvelox-stage"
    run(["cmake", "--install", ".", "--component", "pyvelox_libraries", "--prefix", str(stage)], vb)
    src_pkg = vb.parent.parent / "python" / "pyvelox"
    shutil.copytree(src_pkg, stage / "pyvelox", dirs_exist_ok=True)

    pth = Path(sysconfig.get_paths()["purelib"]) / "loopbench_velox.pth"
    pth.write_text(f"{out}\n{stage}\n")
    print(f"built {so}\nwrote {pth}")


if __name__ == "__main__":
    sys.exit(main())
