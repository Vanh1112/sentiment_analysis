from app import app

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", threaded=False)
    except Exception:
        raise
