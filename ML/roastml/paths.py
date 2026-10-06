"""หาโฟลเดอร์ข้อมูล ($ROAST_DATA_DIR) ที่อยู่นอก repo (ml-spec ข้อ 5)

ลำดับ (ห้าม fallback เงียบๆ — ถ้าที่ระบุไว้ใช้ไม่ได้ให้ error ทันที ไม่ข้ามไปลำดับถัดไป):
  1. env `ROAST_DATA_DIR` (ถ้าตั้งไว้และไม่ว่าง)
  2. `ML/local_paths.json` key `data_dir` (gitignored; path สัมพัทธ์ = เทียบกับโฟลเดอร์ของไฟล์ json)
  3. DataDirError พร้อมวิธีแก้

ใช้แค่ stdlib — import ได้ทั้งบน notebook และ Pi
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "ROAST_DATA_DIR"
# ML/local_paths.json (roastml/ อยู่ใต้ ML/)
LOCAL_PATHS_FILE = Path(__file__).resolve().parent.parent / "local_paths.json"
JSON_KEY = "data_dir"

# โฟลเดอร์ย่อยมาตรฐานใน data dir
ZIPS, RAW, CONTACT_SHEETS, CACHE = "zips", "raw", "contact_sheets", "cache"
SUBDIRS = (ZIPS, RAW, CONTACT_SHEETS, CACHE)

_HOW_TO_FIX = (
    "วิธีแก้ (เลือกอย่างใดอย่างหนึ่ง):\n"
    f"  - ตั้ง env: {ENV_VAR}=C:/Users/Public/Documents/TOPGUN_CONTEST/data\n"
    f'  - หรือสร้าง {LOCAL_PATHS_FILE.name} ใน ML/: {{"{JSON_KEY}": "C:/.../data"}}'
)


class DataDirError(RuntimeError):
    """หา data dir ไม่ได้ หรือค่าที่ตั้งไว้ใช้ไม่ได้"""


def _check_dir(path: Path, origin: str, must_exist: bool) -> Path:
    if must_exist and not path.is_dir():
        raise DataDirError(f"data dir จาก {origin} ไม่มีอยู่หรือไม่ใช่โฟลเดอร์: {path}\n{_HOW_TO_FIX}")
    return path


def data_dir(
    *,
    environ: dict[str, str] | None = None,
    local_paths: Path | None = None,
    must_exist: bool = True,
) -> Path:
    """คืน path ของ data dir (absolute) ตามลำดับในหัวไฟล์

    environ / local_paths มีไว้ให้ test แทนค่าได้ (ค่าเริ่มต้น = os.environ / ML/local_paths.json)
    """
    env = os.environ if environ is None else environ
    json_path = LOCAL_PATHS_FILE if local_paths is None else Path(local_paths)

    # 1) env — ค่าว่าง/ช่องว่างล้วนถือว่าไม่ได้ตั้ง
    value = env.get(ENV_VAR, "").strip()
    if value:
        p = Path(value).expanduser().resolve()
        return _check_dir(p, f"env {ENV_VAR}", must_exist)

    # 2) local_paths.json — ถ้ามีไฟล์แต่อ่านไม่ได้/ค่าผิด ให้ error (ไม่ข้ามเงียบๆ)
    if json_path.is_file():
        origin = str(json_path)
        try:
            cfg = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise DataDirError(f"อ่าน {origin} ไม่ได้: {e}\n{_HOW_TO_FIX}") from e
        raw = cfg.get(JSON_KEY) if isinstance(cfg, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            raise DataDirError(f'{origin} ต้องเป็น object ที่มี "{JSON_KEY}" เป็น string ไม่ว่าง\n{_HOW_TO_FIX}')
        p = Path(raw.strip()).expanduser()
        if not p.is_absolute():
            p = json_path.parent / p
        return _check_dir(p.resolve(), origin, must_exist)

    # 3) ไม่มีทั้งสองอย่าง
    raise DataDirError(f"ไม่ได้ตั้ง data dir (ไม่มี env {ENV_VAR} และไม่มี {json_path})\n{_HOW_TO_FIX}")


def ensure_layout(root: Path) -> list[Path]:
    """สร้างโฟลเดอร์ย่อยมาตรฐานที่ยังไม่มี (ไม่แตะของที่มีอยู่แล้ว) · คืนรายการที่สร้างใหม่"""
    root = Path(root)
    if not root.is_dir():
        raise DataDirError(f"data dir ไม่มีอยู่: {root}")
    created = []
    for name in SUBDIRS:
        d = root / name
        if not d.exists():
            d.mkdir()
            created.append(d)
        elif not d.is_dir():
            raise DataDirError(f"{d} มีอยู่แล้วแต่ไม่ใช่โฟลเดอร์")
    return created
