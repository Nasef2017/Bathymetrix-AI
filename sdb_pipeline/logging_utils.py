def append_log(msg, log_path, feedback):
    feedback.pushInfo(msg)
    if log_path:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
