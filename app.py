"""
VOLTAI - Stock Market Volatility Prediction
Main Flask Application Entry Point
"""

import os
import json
import logging
import traceback
from flask import Flask, render_template, request, jsonify
from train_models import run_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)


@app.route("/")
def index():
    """Landing page with ticker input."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives ticker from form, runs full ML pipeline, returns results page.
    All training happens here — no manual steps required.
    """
    ticker = request.form.get("ticker", "").strip().upper()
    if not ticker:
        return render_template("index.html", error="Please enter a valid ticker symbol.")

    logger.info("=" * 60)
    logger.info(f"  NEW REQUEST: ticker={ticker}")
    logger.info("=" * 60)

    try:
        results = run_pipeline(ticker)
        return render_template("results.html", results=results, ticker=ticker)

    except ValueError as e:
        logger.error(f"Data error for {ticker}: {e}")
        return render_template("index.html", error=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return render_template("index.html", error=f"An unexpected error occurred: {e}")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint — same pipeline, JSON response."""
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    try:
        results = run_pipeline(ticker)
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting VOLTAI Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
