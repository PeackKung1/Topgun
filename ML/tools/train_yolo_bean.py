"""เทรน detector เมล็ดกาแฟคลาสเดียว (Ultralytics YOLO11n, AGPL-3.0) แล้ว export ONNX สำหรับ roastml.detector

    python -m tools.train_yolo_bean --exp holdout_boos --out results/yolo_bean_20261010/holdout_boos

- ประกาศ config ก่อนเทรน (declaration.json) · epoch คงที่ ใช้ last.pt — ไม่เลือก epoch/threshold ด้วยข้อมูลใด
- ผลเทรนของ Ultralytics (มีภาพจากข้อมูล) อยู่นอก git ที่ $ROAST_DATA_DIR/cache/yolo_runs/
- น้ำหนักที่ใช้ต่อ: ML/models/<run>/<exp>/bean.pt + bean.onnx (models/ ถูก gitignore)
- ใช้บน notebook เท่านั้น · Pi ใช้แค่ onnxruntime (ไม่ต้องลง ultralytics/torch)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from roastml.paths import data_dir
from tools.perbean_protocol import GATES, SEED, DevData, declare, dump


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=("holdout_boos", "final"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="yolo_synth_v1")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # ห้าม ultralytics ลง package เอง (เช่น opencv-python ทับ headless)
    data = DevData(data_dir())
    ds = data.root / "cache" / args.dataset
    yaml_path = ds / f"{args.exp}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"ไม่พบ {yaml_path} — รัน python -m tools.make_synth_beans ก่อน")
    ds_summary = json.loads((ds / "summary.json").read_text(encoding="utf-8"))
    train_cfg = {"model": args.weights, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": SEED,
                 "single_cls": True, "amp": False, "deterministic": True, "plots": False, "augment": "ultralytics defaults",
                 "checkpoint_used": "last.pt (fixed epochs; val only monitors loss)"}
    declare(args.out, f"yolo-bean-{args.exp}", {
        "selection": "none: one predeclared config, fixed epochs, last.pt",
        "inference": "conf 0.25, NMS IoU 0.7, max_det 1000 (ultralytics defaults except max_det); not tuned",
        "evaluation": "holdout_boos (synthetic from ontoum sprites only) is tested on real rf_boos; final on Empty report-half/agtron; synthetic val is never gate evidence",
        "gates": {k: GATES[k] for k in ("flat_mae", "boos_mae", "empty_zero_rate", "fold_f1_drop", "cross_extreme_rate", "pi_p95_ms")},
        "dataset": ds_summary["experiments"][args.exp], "sprites": ds_summary["sprites"], "license": "Ultralytics AGPL-3.0; data CC BY 4.0"}, train_cfg, data)

    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # ไม่ส่ง analytics
    pre = Path(__file__).resolve().parent.parent / "models" / "pretrained"
    pre.mkdir(parents=True, exist_ok=True)
    runs = data.root / "cache" / "yolo_runs"
    name = f"{args.out.parent.name}_{args.out.name}"
    t0 = time.perf_counter()
    model = YOLO(str(pre / args.weights))
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, seed=SEED, deterministic=True, single_cls=True, amp=False, plots=False,
                project=str(runs), name=name, exist_ok=False, verbose=False, patience=args.epochs)
    run_dir = Path(model.trainer.save_dir)
    metrics = {k: float(v) for k, v in (model.trainer.metrics or {}).items()}
    model_dir = Path(__file__).resolve().parent.parent / "models" / args.out.parent.name / args.out.name
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "weights" / "last.pt", model_dir / "bean.pt")
    onnx_path = YOLO(str(model_dir / "bean.pt")).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=False, dynamic=False)
    shutil.move(str(onnx_path), model_dir / "bean.onnx")
    dump(args.out / "train_summary.json", {
        "exp": args.exp, "seconds": time.perf_counter() - t0, "ultralytics": ultralytics.__version__, "run_dir": str(run_dir),
        "val_metrics_last_epoch(monitor only)": metrics, "model_dir": str(model_dir),
        "onnx_bytes": (model_dir / "bean.onnx").stat().st_size, "config": train_cfg})
    print(json.dumps({"done": args.exp, "model_dir": str(model_dir), "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
