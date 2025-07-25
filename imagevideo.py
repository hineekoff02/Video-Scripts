import librosa
import numpy as np
from moviepy.editor import AudioFileClip
from moviepy.video.VideoClip import VideoClip
from PIL import Image, ImageEnhance, ImageDraw, ImageOps
import os
import random
import math

# Load files and configure
AUDIO_FILE = 'audio.mp3'
IMAGE_FOLDER = 'images'
OUTPUT_FILE = 'output_video.mp4'
FPS = 24
RESOLUTION = (640, 640)
FADE_DURATION = 2
.0
SEED = 42

# Load audio for analysis
y, sr = librosa.load(AUDIO_FILE)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
volumes = librosa.util.normalize(onset_env)
tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
print(f"Detected BPM: {tempo_val:.2f}")

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# --- LOAD IMAGES ---
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))]
images = [Image.open(f).convert("RGB").resize(RESOLUTION) for f in image_files]
if not images:
    raise ValueError("No images found in IMAGE_FOLDER.")

# Directional fade-in mask
def create_directional_mask(size, direction, progress):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if direction == "left":
        draw.rectangle([0, 0, int(w * progress), h], fill=255)
    elif direction == "right":
        draw.rectangle([int(w * (1 - progress)), 0, w, h], fill=255)
    elif direction == "top":
        draw.rectangle([0, 0, w, int(h * progress)], fill=255)
    elif direction == "bottom":
        draw.rectangle([0, int(h * (1 - progress)), w, h], fill=255)

    return mask

# Zoom function
def apply_zoom(img, t, zoom_schedule, base_size):
    """Apply zoom effect based on current time and zoom schedule"""
    for (start, end, direction, strength) in zoom_schedule:
        if start <= t <= end:
            progress = (t - start) / (end - start)
            if direction == "in":
                scale = 1 + progress * strength
            else:  # zoom out
                scale = 1 + (1 - progress) * strength

            # Calculate new size and crop/pad
            new_w = int(base_size[0] * scale)
            new_h = int(base_size[1] * scale)
            zoomed = img.resize((new_w, new_h), resample=Image.LANCZOS)
            left = (new_w - base_size[0]) // 2
            top = (new_h - base_size[1]) // 2
            img = zoomed.crop((left, top, left + base_size[0], top + base_size[1]))
            break

    return img

# Function for applying effects (including zoom)
def apply_effects(img, beat_strength, fade_factor, color_overlay, rotation_angle, fade_direction, t, zoom_schedule):
    img = img.rotate(rotation_angle, expand=False)

    # Random enhancements
    random_filters = {
        "contrast": random.uniform(1.5, 2.5),
        "brightness": random.uniform(0.9, 1.1),
        "sharpness": random.uniform(1.5, 3),
        "color": random.uniform(1, 1.8),
    }
    img = ImageEnhance.Contrast(img).enhance(random_filters["contrast"])
    img = ImageEnhance.Brightness(img).enhance(random_filters["brightness"])
    img = ImageEnhance.Sharpness(img).enhance(random_filters["sharpness"])
    img = ImageEnhance.Color(img).enhance(random_filters["color"])

    # Random invert
    if random.random() < 0.4:
        img = ImageOps.invert(img)

    # Color overlay
    overlay = Image.new('RGB', img.size, color_overlay)
    img = Image.blend(img, overlay, alpha=0.5)

    # Apply zoom effect
    img = apply_zoom(img, t, zoom_schedule, RESOLUTION)

    # Directional fade-in
    if fade_factor < 1.0:
        mask = create_directional_mask(img.size, fade_direction, fade_factor)
        black = Image.new("RGB", img.size, (0, 0, 0))
        img = Image.composite(img, black, mask)

    return np.array(img)

# Random zoom schedule function
def generate_zoom_schedule(duration, seed, fps):
    random.seed(seed)
    zoom_schedule = []
    t = 0
    while t < duration:
        segment_len = random.uniform(0.5, 2.0)
        direction = random.choice(["in", "out"])
        strength = random.uniform(0.05, 0.15)
        zoom_schedule.append((t, t + segment_len, direction, strength))
        t += segment_len
    return zoom_schedule

# Function for making frames
zoom_schedule = generate_zoom_schedule(duration=librosa.get_duration(y=y, sr=sr), seed=SEED, fps=FPS)

def make_frame(t):
    beat_strength = 0.1
    for i, bt in enumerate(beat_times):
        if t < bt:
            beat_strength = volumes[max(0, i - 1)]
            break

    random.seed(int(t * 1000) + SEED)
    img = random.choice(images).copy()
    rotation_angle = random.choice([0, 90, 180, 270])
    color_overlay = random_color()
    fade_direction = random.choice(['left', 'right', 'top', 'bottom'])
    fade_factor = min(1.0, t / FADE_DURATION)

    return apply_effects(img, beat_strength, fade_factor, color_overlay, rotation_angle, fade_direction, t, zoom_schedule)

# Create the video
audio_clip = AudioFileClip(AUDIO_FILE)
duration = audio_clip.duration

video_clip = VideoClip(make_frame, duration=duration).set_fps(FPS)
final_clip = video_clip.set_audio(audio_clip)

final_clip.write_videofile(OUTPUT_FILE, codec='libx264', audio_codec='aac', audio_bitrate='320k')




