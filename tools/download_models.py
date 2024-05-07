import os
import requests
from tqdm import tqdm

# filenames and URLs
files_to_download = {
    "s2G488k.pth": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s2G488k.pth",
    "s2D488k.pth": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s2D488k.pth",
    "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "chinese-roberta-wwm-ext-large/config.json": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json",
    "chinese-hubert-base/config.json": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin": "https://huggingface.co/blaise-tk/GPT-SoVITS-Fork/resolve/main/chinese-hubert-base/pytorch_model.bin"
}

# Folder to store the downloaded files
base_destination_folder = "GPT_SoVITS/pretrained_models"
os.makedirs(base_destination_folder, exist_ok=True)

# Download the files with progress bar
for file_name, url in files_to_download.items():
    try:
        destination_file_path = os.path.join(base_destination_folder, file_name)
        
        # Check if file already exists
        if os.path.exists(destination_file_path):
            continue
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, leave=False)
            with open(destination_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()
            print(f"File {file_name} downloaded successfully.")
        else:
            print(f"Failed to download file {file_name}. Response status: {response.status_code}")
    except Exception as e:
        print(f"Error downloading file {file_name}: {str(e)}")
