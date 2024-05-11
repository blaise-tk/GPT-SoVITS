import platform, os, traceback
import ffmpeg
import numpy as np


def load_audio(file, sr):
    try:
        file = clean_path(file)
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as error:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {error}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return (
        path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ").strip("\u202a")
    )
