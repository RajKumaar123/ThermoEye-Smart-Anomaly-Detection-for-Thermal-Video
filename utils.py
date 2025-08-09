# utils.py

def extract_timestamp_from_fps(frame_index, fps):
    seconds = frame_index / fps
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02}:{secs:02}"
