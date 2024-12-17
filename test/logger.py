_log_file = None

def init_logger(filepath):
    global _log_file
    _log_file = open(filepath, 'w')

def log(content):
    if _log_file is not None:
        _log_file.write(content + '\n')
    else:
        raise ValueError("Logger is not initialized.")

def close_logger():
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None
