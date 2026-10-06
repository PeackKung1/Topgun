"""แตก $ROAST_DATA_DIR/zips/<source>.zip → raw/<source>/ แล้วสำรวจโครงสร้าง (ml-spec ข้อ 5)

กฎ (CLAUDE.local.md):
  - แตกเฉพาะเมื่อ raw/<source>/ ยังไม่มี — ไม่เขียนทับ ไม่ลบ ไม่แก้ของใน raw/ และไม่ลบ zip
  - รับเฉพาะ source ใน SOURCES · ชื่ออื่นเตือนแล้วข้าม
  - member ที่ path หลุดออกนอกปลายทาง (zip-slip) → ปฏิเสธทั้งไฟล์
  - zip เสีย → รายงานแล้วข้าม
  - แตกลง cache/extract_tmp/ ก่อน แล้ว rename เป็น raw/<source>/ ครั้งเดียว (ไม่ค้างครึ่งๆ ใน raw/)

ใช้: python tools/extract_zips.py [--data-dir PATH] [--only agtron rf_hendi ...] [--no-survey]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from roastml.paths import CACHE, RAW, ZIPS, DataDirError, data_dir, ensure_layout

# rf_color ตัดออก 6 ต.ค. (มี augment + adaptive equalization → ขัด ml-spec ข้อ 5)
SOURCES = ("agtron", "ontoum224", "rf_robusta", "rf_boos", "rf_hendi", "rf_devlong")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
TMP_PARENT = "extract_tmp"  # ใต้ cache/


@dataclass
class ExtractResult:
    source: str
    status: str  # extracted | exists | unknown_source | unsafe | bad_zip | error
    message: str = ""
    n_members: int = 0
    warnings: list[str] = field(default_factory=list)


class UnsafeZipError(Exception):
    """zip มี member ที่ path ไม่ปลอดภัย"""


# ------------------------------------------------------------------ zip-slip
def _unsafe_reason(name: str) -> str | None:
    """คืนเหตุผลถ้าชื่อ member ไม่ปลอดภัย (None = ปลอดภัย) — ตรวจทั้งแบบ POSIX และ Windows"""
    if not name or "\x00" in name:
        return "ชื่อว่างหรือมี NUL"
    norm = name.replace("\\", "/")
    win = PureWindowsPath(name)
    if norm.startswith("/") or win.drive or win.root:
        return "absolute path"
    if ".." in PurePosixPath(norm).parts:
        return "มี '..'"
    if ":" in norm:  # drive-relative (C:foo) หรือ NTFS alternate data stream
        return "มี ':'"
    return None


def validate_members(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """ตรวจทุก member ก่อนแตก → raise UnsafeZipError ถ้ามีตัวไหนหลุดออกนอก dest · คืน warnings"""
    dest_resolved = dest.resolve()
    seen: Counter[str] = Counter()
    for info in zf.infolist():
        reason = _unsafe_reason(info.filename)
        if reason is None:
            target = (dest_resolved / info.filename.replace("\\", "/")).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                reason = "resolve แล้วอยู่นอกปลายทาง"
        if reason:
            raise UnsafeZipError(f"member {info.filename!r}: {reason}")
        if not info.is_dir():
            seen[info.filename.replace("\\", "/").casefold()] += 1
    # Windows ไม่แยกตัวพิมพ์เล็ก/ใหญ่ → ไฟล์ชื่อซ้ำจะทับกัน
    dups = [n for n, c in seen.items() if c > 1]
    return [f"ชื่อไฟล์ซ้ำ (ไม่สนตัวพิมพ์) {len(dups)} ชื่อ เช่น {dups[0]!r} — จะทับกันบน Windows"] if dups else []


# ------------------------------------------------------------------ extract
def extract_one(zip_path: Path, root: Path) -> ExtractResult:
    """แตก zip หนึ่งไฟล์ตามกฎในหัวไฟล์ · ไม่ raise (คืนสถานะใน ExtractResult)"""
    source = zip_path.stem
    if source not in SOURCES:
        return ExtractResult(source, "unknown_source", f"ไม่ใช่ source ที่รู้จัก {SOURCES} → ข้าม")

    dest = root / RAW / source
    if os.path.lexists(dest):
        return ExtractResult(source, "exists", f"{dest} มีอยู่แล้ว → ไม่แตะ")

    tmp_parent = root / CACHE / TMP_PARENT
    tmp = tmp_parent / f"{source}_{uuid.uuid4().hex[:8]}"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            warnings = validate_members(zf, tmp)
            n = len(zf.infolist())
            tmp.mkdir(parents=True)
            zf.extractall(tmp)  # CRC เสียจะ raise BadZipFile ระหว่างนี้
        # rename ครั้งเดียว · Windows: ล้มถ้า dest เกิดขึ้นระหว่างทาง (ไม่ทับ)
        os.rename(tmp, dest)
        return ExtractResult(source, "extracted", f"→ {dest}", n_members=n, warnings=warnings)
    except UnsafeZipError as e:
        return ExtractResult(source, "unsafe", f"ปฏิเสธทั้งไฟล์ (zip-slip): {e}")
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, NotImplementedError) as e:
        return ExtractResult(source, "bad_zip", f"zip เสีย/อ่านไม่ได้: {type(e).__name__}: {e}")
    except OSError as e:
        return ExtractResult(source, "error", f"{type(e).__name__}: {e}")
    finally:
        # ลบเฉพาะโฟลเดอร์ชั่วคราวของรอบนี้ (อยู่ใน cache/ ไม่ใช่ raw/)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def extract_all(root: Path, only: list[str] | None = None) -> list[ExtractResult]:
    zips = sorted((root / ZIPS).glob("*.zip"), key=lambda p: p.name.lower())
    if only:
        zips = [z for z in zips if z.stem in only]
    return [extract_one(z, root) for z in zips]


# ------------------------------------------------------------------ survey
@dataclass
class Survey:
    source: str
    n_files: int
    n_images: int
    total_bytes: int
    tree: list[str]  # 2 ชั้นแรก (โฟลเดอร์ + จำนวนไฟล์ภาพข้างใต้)
    image_dirs: dict[str, int]  # โฟลเดอร์ที่มีภาพโดยตรง (path สัมพัทธ์) → จำนวนภาพ
    ext_counts: dict[str, int]


def survey_source(src_dir: Path) -> Survey:
    n_files = n_images = total = 0
    image_dirs: Counter[str] = Counter()
    exts: Counter[str] = Counter()
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        rel = Path(dirpath).relative_to(src_dir).as_posix()
        for fn in filenames:
            n_files += 1
            try:
                total += (Path(dirpath) / fn).stat().st_size
            except OSError:
                pass
            ext = Path(fn).suffix.lower()
            exts[ext or "(none)"] += 1
            if ext in IMAGE_EXTS:
                n_images += 1
                image_dirs[rel] += 1

    def images_under(prefix: str) -> int:
        return sum(c for d, c in image_dirs.items() if d == prefix or d.startswith(prefix + "/"))

    tree = []
    for d1 in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        tree.append(f"{d1.name}/  ({images_under(d1.name)} ภาพ)")
        for d2 in sorted(p for p in d1.iterdir() if p.is_dir()):
            key = f"{d1.name}/{d2.name}"
            tree.append(f"  {d2.name}/  ({images_under(key)} ภาพ)")
    top_files = [p.name for p in src_dir.iterdir() if p.is_file()]
    if top_files:
        tree.append(f"(ไฟล์ระดับบนสุด {len(top_files)}: {', '.join(sorted(top_files)[:8])})")
    return Survey(src_dir.name, n_files, n_images, total, tree,
                  dict(sorted(image_dirs.items())), dict(exts.most_common()))


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def print_survey(s: Survey) -> None:
    print(f"\n=== {s.source}: {s.n_images} ภาพ / {s.n_files} ไฟล์ · {_fmt_size(s.total_bytes)}")
    print("  นามสกุล:", ", ".join(f"{k}={v}" for k, v in s.ext_counts.items()))
    print("  โครง 2 ชั้นแรก:")
    for line in s.tree or ["(ไม่มีโฟลเดอร์ย่อย)"]:
        print("   ", line)
    print("  โฟลเดอร์ที่มีภาพ (ชั้นสุดท้าย = ชื่อคลาสที่น่าจะเป็น):")
    for d, c in s.image_dirs.items():
        print(f"    {d or '.'}: {c}")


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="แทนค่าจาก env/local_paths.json")
    ap.add_argument("--only", nargs="+", metavar="SOURCE", help="แตกเฉพาะ source เหล่านี้")
    ap.add_argument("--no-survey", action="store_true", help="ไม่สำรวจ raw/ หลังแตก")
    args = ap.parse_args(argv)

    # Windows: stdout ที่ถูก pipe เป็น cp1252 → print ภาษาไทยแล้วล้ม
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        root = args.data_dir.resolve() if args.data_dir else data_dir()
        if not root.is_dir():
            raise DataDirError(f"data dir ไม่มีอยู่: {root}")
        ensure_layout(root)
    except DataDirError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"data dir: {root}")

    results = extract_all(root, args.only)
    if not results:
        print(f"ไม่มีไฟล์ .zip ใน {root / ZIPS}")
    for r in results:
        tag = "WARN" if r.status == "unknown_source" else r.status.upper()
        print(f"[{tag}] {r.source}: {r.message}" + (f" ({r.n_members} members)" if r.n_members else ""))
        for w in r.warnings:
            print(f"    warning: {w}")

    if not args.no_survey:
        raw = root / RAW
        dirs = sorted(p for p in raw.iterdir() if p.is_dir() and (not args.only or p.name in args.only))
        if not dirs:
            print(f"\n{raw} ยังว่าง — ไม่มีอะไรให้สำรวจ")
        for d in dirs:
            print_survey(survey_source(d))

    failed = [r for r in results if r.status in ("unsafe", "bad_zip", "error")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
