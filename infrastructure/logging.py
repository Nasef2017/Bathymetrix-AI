# infrastructure/logging.py
import datetime


def append_log(message, file_path, feedback=None):
    """
    Appends a multi-line or single-line message to a text log file,
    and formats it beautifully for QGIS feedback.
    """
    # 1. Clean Text for File (No HTML)
    txt_msg = message.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
    import re
    txt_msg = re.sub(r'<font color=[^>]+>', '', txt_msg).replace('</font>', '')
    txt_msg = txt_msg.replace('&nbsp;', ' ')
    
    if file_path:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(txt_msg + "\n")

    # 2. Rich Unicode for QGIS Panel (No HTML tags, since QGIS prints them literally)
    if feedback is not None:
        console_msg = message
        
        # Add icons based on content if they don't already exist
        if "---" in console_msg or "===" in console_msg:
            # Already formatted as headers
            pass
        elif "✅" in console_msg or "Finished" in console_msg or "Success" in console_msg:
            if "✅" not in console_msg:
                console_msg = "✅ " + console_msg
        elif "[Warning]" in console_msg or "⚠️" in console_msg:
            if "⚠️" not in console_msg:
                console_msg = "⚠️ " + console_msg
        elif "Failed" in console_msg or "ERROR" in console_msg or "❌" in console_msg:
            if "❌" not in console_msg:
                console_msg = "❌ " + console_msg
        elif "📊" in console_msg or "[Analytics]" in console_msg or "Model" in console_msg:
            if "📊" not in console_msg and "Model" in console_msg and "Skipping" not in console_msg:
                console_msg = "📊 " + console_msg
            
        # Preserve indentation for readability
        if console_msg.startswith("   "):
            console_msg = "    🔹 " + console_msg.lstrip()

        is_error = message.strip().startswith("!") or "ERROR" in message or "Failed" in message
        
        if getattr(feedback, "is_logging_feedback", False) and feedback.original:
            if is_error:
                feedback.original.pushWarning(console_msg)
            else:
                feedback.original.pushInfo(console_msg)
        else:
            if is_error:
                feedback.pushWarning(console_msg)
            else:
                feedback.pushInfo(console_msg)


def initialize_log(file_path, feedback=None):
    """Creates a new log file with a timestamp header."""
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Bathymetrix-AI LOG - {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
    if feedback:
        if getattr(feedback, "is_logging_feedback", False):
            if feedback.original:
                feedback.original.pushInfo(">>> Log Initialized.")
        else:
            feedback.pushInfo(">>> Log Initialized.")
