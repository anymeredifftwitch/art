from scripts.process_video import trim_video_for_short

def process_chatting_clip(input_path, output_path, max_duration_seconds, clip_data):
    return trim_video_for_short(input_path, output_path, max_duration_seconds, clip_data, enable_webcam_crop=True)
