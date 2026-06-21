# infrastructure/logging.py
import datetime


def append_log(message, file_path, feedback=None):
    """
    Appends a multi-line or single-line message to a text log file,
    and optionally pipes it to QGIS feedback if provided.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    if feedback is not None:
        if message.strip().startswith("!") or "ERROR" in message or "Failed" in message:
            feedback.pushWarning(message)
        else:
            feedback.pushInfo(message)


def initialize_log(file_path, feedback=None):
    """Creates a new log file with a timestamp header."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Bathymetrix-AI LOG - {datetime.datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
    if feedback:
        feedback.pushInfo(">>> Log Initialized.")
