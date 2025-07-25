import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips
)
from moviepy.video.fx.all import fadeout


def smooth_trippy_effect(frame, t, duration):
    """
    Apply smoothly transitioning color effects using sinusoidal blending.
    Transitions gradually between:
    - Inversion
    - JET
    - HSV
    - OCEAN
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)

    # Normalize time to smooth cyclical factor [0, 1]
    phase = (t % duration) / duration
    angle = 2 * np.pi * phase

    # Generate blend weights from sine waves
    w_invert = (np.sin(angle) + 1) / 2
    w_jet = (np.sin(angle + np.pi/2) + 1) / 2
    w_hsv = (np.sin(angle + np.pi) + 1) / 2
    w_ocean = (np.sin(angle + 3*np.pi/2) + 1) / 2

    # Normalize weights so they sum to 1
    total = w_invert + w_jet + w_hsv + w_ocean
    w_invert /= total
    w_jet /= total
    w_hsv /= total
    w_ocean /= total

    # Compute effects
    inverted = 255 - frame
    jet = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    hsv = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_HSV).astype(np.float32)
    ocean = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_OCEAN).astype(np.float32)

    # Weighted blend of all effects
    blended = (
        w_invert * inverted +
        w_jet * jet +
        w_hsv * hsv +
        w_ocean * ocean
    )

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def apply_trippy_effect(clip, cycle_duration=10):
    return clip.fl(lambda gf, t: smooth_trippy_effect(gf(t), t, cycle_duration))


# File paths
video_path = "chilax.mp4"
audio_path = "chilax.mp3"
output_path = "chilax_video.mp4"

# Processing
print("Loading video and audio...")
video = VideoFileClip(video_path).without_audio()
audio = AudioFileClip(audio_path)

# Resize video to 720p for smaller file (or change to 480 if you want even smaller)
video = video.resize(height=720)  # maintain aspect ratio

# Determine clip length
total_duration = audio.duration + 5  # 5-second fade at end
video_duration = min(video.duration, total_duration)

video = video.subclip(0, video_duration)

print("Applying visual effects...")
processed_video = apply_trippy_effect(video, cycle_duration=12)

# Split for audio/fade handling
fade_start_time = min(audio.duration, processed_video.duration - 5)
first_part = processed_video.subclip(0, fade_start_time).set_audio(audio)
fade_part = processed_video.subclip(fade_start_time, processed_video.duration)
fade_part = fadeout(fade_part, duration=5)

# Combine and export
final_clip = concatenate_videoclips([first_part, fade_part])

print("Exporting final video...")
final_clip.write_videofile(
    output_path,
    codec="libx264",
    audio_codec="aac",
    audio_bitrate="320k",  # high quality audio
    bitrate="2500k",       # reduce video bitrate for smaller file
    fps=video.fps,
    threads=4
)

print("Done! Video saved as:", output_path)

