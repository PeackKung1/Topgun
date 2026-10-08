"""Run the firmware Flask app under Waitress for systemd deployments."""

from __future__ import annotations

import os

from waitress import serve

from app import app


if __name__ == "__main__":
    port = int(os.getenv("TOPGUN_PORT", "8080"))
    serve(app, host="0.0.0.0", port=port, threads=4)
