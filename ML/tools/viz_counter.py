"""Safe debug contact sheet from an already declared counter run (tune half only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from roastml.paths import data_dir
from tools.perbean_protocol import DevData
from tools.tune_counter import load_rgb

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--name", default="debug_counter_dev_20261010.jpg")
    args = p.parse_args()
    data = DevData(data_dir())
    recs = json.loads((args.run / "per_image.json").read_text(encoding="utf-8"))
    tune, _ = data.empty_split()
    tune_paths = {r["path"] for r in tune}
    picks = [r for r in recs if r["source"] == "ontoum224" and r["n"] > 20][:6]
    picks += [r for r in recs if r["source"] == "rf_boos" and r["n"] > r["gt"]][:6]
    picks += [r for r in recs if r["path"] in tune_paths][:6]
    sheet = Image.new("RGB", (4*240, ((len(picks)+3)//4)*264), "white")
    for i, r in enumerate(picks):
        img = Image.fromarray(load_rgb(data, data.by_path[r["path"]]))
        img.thumbnail((236, 236))
        x,y = i%4*240,i//4*264
        sheet.paste(img, (x,y+24))
        ImageDraw.Draw(sheet).text((x,y), f"{i} {r['source']} GT {r['gt']} n {r['n']} ref {r['ref_area']:.0f}", fill="black")
    out = data.root / "contact_sheets" / args.name
    sheet.save(out)
    print(str(out))

if __name__ == "__main__":
    main()
