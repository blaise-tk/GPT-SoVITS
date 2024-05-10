import os
import requests
from tqdm import tqdm

# Folder to store the downloaded files
base_destination_folder = "GPT_SoVITS/pretrained_models"
os.makedirs(base_destination_folder, exist_ok=True)


# Function to download a file with progress bar
def download_with_progress(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(destination),
            leave=False,
        )
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
    else:
        print(
            f"Failed to download file {os.path.basename(destination)}. Response status: {response.status_code}"
        )


# List of files to download
files_to_download = [
    {
        "filename": "s2G488k.pth",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s2G488k.pth",
    },
    {
        "filename": "s2D488k.pth",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s2D488k.pth",
    },
    {
        "filename": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    },
    {
        "filename": "chinese-roberta-wwm-ext-large/config.json",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/config.json",
    },
    {
        "filename": "chinese-roberta-wwm-ext-large/pytorch_model.bin",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    },
    {
        "filename": "chinese-roberta-wwm-ext-large/tokenizer.json",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json",
    },
    {
        "filename": "chinese-hubert-base/config.json",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/config.json",
    },
    {
        "filename": "chinese-hubert-base/preprocessor_config.json",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/preprocessor_config.json",
    },
    {
        "filename": "chinese-hubert-base/pytorch_model.bin",
        "url": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/pytorch_model.bin",
    },
]

# Download files with progress bar
for file in files_to_download:
    destination_file_path = os.path.join(base_destination_folder, file["filename"])
    if not os.path.exists(destination_file_path):
        download_with_progress(file["url"], destination_file_path)

# Function to download FFmpeg and FFprobe for Windows
if os.name == "nt":  # Windows
    base_directory = os.path.abspath(".")
    ffmpeg_url = "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/ffmpeg.exe"
    ffprobe_url = "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/ffprobe.exe"
    
    def download_ff_tool(tool_url, tool_name):
        tool_destination_path = os.path.join(base_directory, tool_name)
        if not os.path.exists(tool_destination_path):
            tool_response = requests.get(tool_url, stream=True)
            total_size = int(tool_response.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=tool_name,
                leave=False,
            )
            with open(tool_destination_path, "wb") as f:
                for chunk in tool_response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()

    download_ff_tool(ffmpeg_url, "ffmpeg.exe")
    download_ff_tool(ffprobe_url, "ffprobe.exe")
