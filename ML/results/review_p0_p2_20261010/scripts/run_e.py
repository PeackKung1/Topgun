import sys, os, json, io, logging, collections, random, importlib.util
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from roastml.api import load
from roastml.contract import RESULT_KEYS, LABELS, STATUSES, WARNINGS, TIMING_KEYS
from tools.perbean_protocol import DevData
from tools.bench_perbean import synthetic_pile_300, synthetic_300, jpeg

OUT = sys.argv[1]
MAIN = "C:/Users/Public/Documents/TOPGUN_CONTEST/Topgun/ML/models/"
SW = "../../roast-classification-seed-count-91d3d9/"
MODELS = {"current(main default)": MAIN + "current", "bean_candidate(main)": MAIN + "bean_candidate",
          "v3_preview(worktree only)": SW + "ML/models/perbean_v3_preview_20261010"}


class Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, r):
        self.msgs.append((r.levelname, r.getMessage()[:120], (str(r.exc_info[1])[:160] if r.exc_info else "")))


cap = Cap()
logging.getLogger("roastml").addHandler(cap)
logging.getLogger("roastml").setLevel(logging.INFO)


def modload(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


svc_old = modload("svc_old", "../Firmware/service.py")           # origin/main FW (unpatched)
svc_new = modload("svc_new", SW + "Firmware/service.py")         # patched FW (uncommitted)


def mqtt_result(result):  # copy of patched app.mqtt_result (app.py itself imports flask)
    event = {k: v for k, v in result.items() if k != "beans"}
    if result.get("schema_version") != 3:
        event["beans"] = []
    return event


def fw_check(svc, res):
    try:
        svc.validate_event({"msg_id": "12345678-1234-5678-1234-567812345678",
                            "created_at": "2026-10-10T00:00:00.000+00:00", "result": res})
        return "ok"
    except Exception as e:
        return "REJECT: " + str(e)


data = DevData(os.environ["ROAST_DATA_DIR"])   # safe rows only: trainval minus count-test frozen groups
rng = random.Random(20261010)
safe = sorted(data.rows, key=lambda r: r["path"])


def pick(src, label=None):
    return rng.choice([r for r in safe if r["source"] == src and (label is None or r["label"] == label)])


five = [("ontoum", pick("ontoum224", "light")), ("boos", pick("rf_boos", "mixed")),
        ("agtron(full frame)", pick("agtron", "medium")), ("empty(hendi)", pick("rf_hendi", "empty"))]
pile, vis_gt, _ = synthetic_pile_300()
five_bytes = [(n, r["path"], r["label"], (data.root / r["path"]).read_bytes()) for n, r in five]
five_bytes.append(("synthetic pile300", "(synthetic)", "visible gt=%d" % vis_gt, jpeg(pile)))
hundred = rng.sample(safe, 100)


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


valid_j = jpeg(np.full((300, 400, 3), 200, np.uint8))
extra = [("white", png(np.full((600, 800, 3), 255, np.uint8))), ("black", png(np.zeros((600, 800, 3), np.uint8))),
         ("rgb noise", png(np.random.default_rng(1).integers(0, 255, (600, 800, 3), dtype=np.uint8))),
         ("hendi empty 2", (data.root / pick("rf_hendi", "empty")["path"]).read_bytes()),
         ("hendi empty 3", (data.root / pick("rf_hendi", "empty")["path"]).read_bytes()),
         ("synthetic 300 flat", jpeg(synthetic_300()))]
bad = [("empty bytes", b""), ("random bytes", bytes(np.random.default_rng(2).integers(0, 255, 5000, dtype=np.uint8))),
       ("truncated jpeg", valid_j[:len(valid_j) // 3]), ("text", b"hello world" * 50),
       ("4x4 png", png(np.zeros((4, 4, 3), np.uint8))), ("None", None), ("str path", "a.jpg"),
       ("pdf header", b"%PDF-1.4 abc")]


def iou_pairs(beans, t=0.5):
    if len(beans) < 2:
        return 0
    b = np.array([x["bbox"] for x in beans], float)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0), 0, None)
    inter = iw * ih
    a = b[:, 2] * b[:, 3]
    return int(np.triu(inter / np.maximum(a[:, None] + a - inter, 1e-9) > t, 1).sum())


def check(res, expect_bad=False):
    v = []
    if tuple(res.keys()) != tuple(RESULT_KEYS):
        v.append("keys")
    try:
        json.dumps(res, allow_nan=False)
    except Exception:
        v.append("json")
    if res["status"] not in STATUSES:
        v.append("status")
    if set(res["timing_ms"]) != set(TIMING_KEYS):
        v.append("timing_keys")
    if any(w not in WARNINGS for w in res["warnings"]):
        v.append("warning_unknown")
    if res["status"] in ("bad_image", "error"):
        if any(res[k] is not None for k in ("label", "probs", "confidence", "n_beans", "counts", "count_method",
                                            "proportions")) or res["beans"]:
            v.append("failure_not_null")
        return v
    if expect_bad:
        v.append("bad_input_got_label")
    p = res["probs"]
    if abs(sum(p.values()) - 1) > 2e-4:
        v.append("probs_sum")
    if res["label"] != max(LABELS, key=lambda k: p[k]):
        v.append("label!=argmax(probs)")
    n, c, b = res["n_beans"], res["counts"], res["beans"]
    if n is None:
        if c is not None or res["count_method"] is not None or b or res["proportions"] is not None:
            v.append("null_count_inconsistent")
        return v
    if not (n == len(b) == sum(c.values())):
        v.append("n!=len(beans)!=sum(counts)")
    if c != {k: sum(x["label"] == k for x in b) for k in LABELS}:
        v.append("counts!=tally")
    if n and c[res["label"]] != max(c.values()):
        v.append("label!=count_majority")
    if n:
        pr = res["proportions"]
        if pr is None or abs(sum(pr.values()) - 1) > 1e-9:
            v.append("proportions_sum")
        elif any(abs(pr[k] - c[k] / n) > 6e-5 for k in LABELS):
            v.append("proportions!=counts/n")
    elif res["proportions"] is not None:
        v.append("proportions_not_null_at_0")
    W, H = res["image_size"]
    if any(x < 0 or y < 0 or w < 1 or h < 1 or x + w > W or y + h > H for x, y, w, h in (q["bbox"] for q in b)):
        v.append("bbox_oob")
    ws = set(res["warnings"])
    if ("mixed_roast" in ws) != (sum(x > 0 for x in c.values()) > 1):
        v.append("mixed_roast_flag")
    if ("no_beans_detected" in ws) != (n == 0):
        v.append("no_beans_flag")
    if ("bean_count_estimated" in ws) != (res["count_method"] == "estimated"):
        v.append("estimated_flag")
    return v


def brief(res):
    return {"status": res["status"], "label": res["label"], "confidence": res["confidence"], "probs": res["probs"],
            "warnings": res["warnings"], "n_beans": res["n_beans"], "counts": res["counts"],
            "count_method": res["count_method"], "proportions": res["proportions"], "len(beans)": len(res["beans"]),
            "beans[:2]": res["beans"][:2], "image_size": res["image_size"],
            "timing_ms": {k: res["timing_ms"][k] for k in ("decode", "ml", "total")}, "model": res["model"],
            "schema_version": res["schema_version"], "message_th": res["message_th"]}


report = {"env": {"ROAST_BEANS": os.environ.get("ROAST_BEANS"), "ROASTML_MODEL": os.environ.get("ROASTML_MODEL")},
          "seed": 20261010, "models": {}}
for mname, mpath in MODELS.items():
    cap.msgs.clear()
    pred = load(mpath)
    info = pred.info()
    R = {"path": mpath, "backend": info["backend"].get("backend"), "name": info["model"],
         "low_conf_threshold": info["low_conf_threshold"], "beans_enabled": info["backend"].get("beans_enabled"),
         "count_config_subset": {k: (info["backend"].get("count_config") or {}).get(k)
                                 for k in ("ws_separation", "metal_chroma_min", "max_beans")},
         "group_config": info["backend"].get("group_config"), "five": []}
    for n, path, label, raw in five_bytes:
        res = pred.predict_bytes(raw)
        R["five"].append({"image": n, "path": path, "true": label, "result": brief(res), "violations": check(res),
                          "dup_iou>0.5": iou_pairs(res["beans"]), "fw_origin_main": fw_check(svc_old, res),
                          "fw_patched(mqtt_result)": fw_check(svc_new, mqtt_result(res)),
                          "fw_patched(full)": fw_check(svc_new, res)})
    viol, status, method, srcs, agree = (collections.Counter() for _ in range(5))
    ties = rej_old = rej_new = dup = dup_imgs = 0
    nb, conf1, ms, rows100 = [], [], [], []
    for r in hundred:
        res = pred.predict_bytes((data.root / r["path"]).read_bytes())
        vv = check(res)
        viol.update(vv)
        status[res["status"]] += 1
        ms.append(res["timing_ms"]["total"])
        srcs[r["source"]] += 1
        rej_old += fw_check(svc_old, res) != "ok"
        rej_new += fw_check(svc_new, mqtt_result(res)) != "ok"
        if res["n_beans"] is not None:
            nb.append(res["n_beans"])
            method[res["count_method"]] += 1
            cv = sorted(res["counts"].values())
            if res["n_beans"] and cv[-1] == cv[-2]:
                ties += 1
            if res["n_beans"] == 1:
                conf1.append(res["confidence"])
            d = iou_pairs(res["beans"])
            dup += d
            dup_imgs += d > 0
        if r["label"] in LABELS and res["label"]:
            agree[(r["source"], res["label"] == r["label"])] += 1
        rows100.append({"path": r["path"], "source": r["source"], "true": r["label"], "status": res["status"],
                        "label": res["label"], "conf": res["confidence"], "n": res["n_beans"],
                        "counts": res["counts"], "warnings": res["warnings"], "ms": res["timing_ms"]["total"],
                        "violations": vv})
    R["hundred"] = {
        "n": len(hundred), "sources": dict(srcs), "status": dict(status), "violations": dict(viol),
        "count_ties": ties, "n_beans_reported": len(nb),
        "n_beans_min_med_max": [int(min(nb)), float(np.median(nb)), int(max(nb))] if nb else None,
        "count_method": dict(method), "single_bean_images": len(conf1),
        "single_bean_conf_min_max": [min(conf1), max(conf1)] if conf1 else None,
        "dup_pairs_iou>0.5": dup, "images_with_dup": dup_imgs,
        "ms_p50_p95_max": [float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), float(max(ms))],
        "fw_origin_main_rejects": rej_old, "fw_patched_rejects": rej_new,
        "label_agreement_single_roast(in-sample; not a metric)":
            {f"{k[0]}:{'right' if k[1] else 'wrong'}": v for k, v in sorted(agree.items())}}
    R["hundred_rows"] = rows100
    R["extra"] = []
    for n, raw in extra:
        res = pred.predict_bytes(raw)
        R["extra"].append({"image": n, "result": brief(res), "violations": check(res),
                           "fw_origin_main": fw_check(svc_old, res)})
    R["bad"] = []
    for n, raw in bad:
        res = pred.predict_bytes(raw)
        R["bad"].append({"input": n, "status": res["status"], "label": res["label"], "message_th": res["message_th"],
                         "image_size": res["image_size"], "violations": check(res, expect_bad=True),
                         "fw_origin_main": fw_check(svc_old, res),
                         "fw_patched": fw_check(svc_new, mqtt_result(res))})
    lm = collections.Counter((l, m, e) for l, m, e in cap.msgs if l in ("ERROR", "WARNING"))
    R["log_errors_warnings"] = [{"level": k[0], "msg": k[1], "exc": k[2], "count": v} for k, v in lm.most_common(12)]
    R["majority_valueerror_count"] = sum("bean-count majority" in e for _, _, e in cap.msgs)
    R["bean_path_fallback_count"] = sum("bean path failed" in m for _, m, _ in cap.msgs)
    report["models"][mname] = R
    print("done", mname, flush=True)
json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
