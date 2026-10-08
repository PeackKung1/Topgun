"""Local web app for taking coffee-bean roast photos from a phone."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from mqtt_pub import enqueue_prediction, start_publisher

LOG = logging.getLogger("topgun.firmware")
ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.getenv("TOPGUN_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))


def create_app(predictor: Any | None = None, *, start_background: bool = True) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    model_load_error: str | None = None
    if predictor is None:
        try:
            # ML package is installed from ../ML; only load once while the service starts.
            from roastml.api import load

            predictor = load(os.getenv("ROASTML_MODEL", "/opt/topgun/ML/models/current"))
        except Exception as exc:
            model_load_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("ML model failed to load; keeping the web service available without predictions")
    app.extensions["roastml_predictor"] = predictor
    app.extensions["roastml_load_error"] = model_load_error

    if start_background and os.getenv("TOPGUN_MQTT_ENABLED", "1").lower() not in {"0", "false", "no"}:
        start_publisher()

    @app.get("/")
    def index():
        from service import read_market_snapshot

        snapshot = read_market_snapshot(os.getenv("TOPGUN_DB_PATH", str(ROOT / "data" / "predictions.sqlite3")))
        return render_template("index.html", snapshot=snapshot)

    @app.post("/api/predict")
    def predict():
        if app.extensions.get("roastml_load_error"):
            return jsonify({"error": "โมเดล ML ยังโหลดไม่สำเร็จ ระบบยังไม่พร้อมวิเคราะห์รูป"}), 503

        uploaded = request.files.get("image") or request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "กรุณาเลือกไฟล์รูปก่อนส่ง"}), 400

        raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if not raw:
            return jsonify({"error": "ไฟล์รูปว่าง กรุณาเลือกรูปใหม่"}), 400
        if len(raw) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "ไฟล์ใหญ่เกินกำหนด กรุณาเลือกรูปใหม่"}), 413

        result = predictor.predict_bytes(raw)
        event = {**result}
        try:
            enqueue_prediction(event)
        except Exception:
            # Logging is best-effort and must never hold up a prediction response.
            LOG.exception("could not queue prediction event")

        return jsonify(result)

    @app.get("/health")
    def health():
        if app.extensions.get("roastml_load_error"):
            return jsonify({
                "status": "error",
                "error": "ML model failed to load",
                "ml": {"model": "unavailable"},
            }), 503
        try:
            info = predictor.info()
            if info.get("model") == "stub":
                return jsonify({
                    "status": "error",
                    "error": "stub model is not allowed for a healthy service",
                    "ml": info,
                }), 503
            return jsonify({"status": "ok", "ml": info}), 200
        except Exception:
            LOG.exception("health check failed")
            return jsonify({"status": "error"}), 503

    @app.get("/market")
    def market():
        from service import read_market_snapshot

        snapshot = read_market_snapshot(os.getenv("TOPGUN_DB_PATH", str(ROOT / "data" / "predictions.sqlite3")))
        return render_template("market.html", snapshot=snapshot)

    @app.get("/api/market")
    def market_snapshot():
        from service import read_market_snapshot

        snapshot = read_market_snapshot(os.getenv("TOPGUN_DB_PATH", str(ROOT / "data" / "predictions.sqlite3")))
        return jsonify(snapshot)

    @app.errorhandler(413)
    def too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "ไฟล์ใหญ่เกินกำหนด กรุณาเลือกรูปใหม่"}), 413
        return "ไฟล์ใหญ่เกินกำหนด", 413

    return app


app = create_app()


if __name__ == "__main__":
    # For local development only. Deployment uses Waitress via systemd.
    app.run(host="0.0.0.0", port=int(os.getenv("TOPGUN_PORT", "8080")), threaded=True)
