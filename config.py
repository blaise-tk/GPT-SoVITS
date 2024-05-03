import os
import torch

sovits_path = ""
gpt_path = ""
is_half = os.getenv("is_half", "True").lower() == "true"
is_share = os.getenv("is_share", "False").lower() == "true"

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = (
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)

exp_root = "logs"
python_exec = os.getenv("PYTHON_EXEC", "env/python")
infer_device = "cuda" if torch.cuda.is_available() else "cpu"

webui_port_main = 6969
webui_port_subfix = 9696

api_port = 9966

gpu_name = torch.cuda.get_device_name(0) if infer_device == "cuda" else ""
if "cuda" in infer_device and (
    "16" in gpu_name
    and "V100" not in gpu_name.upper()
    or any(x in gpu_name.upper() for x in ["P40", "P10", "1060", "1070", "1080"])
):
    is_half = False

is_half = False if infer_device == "cpu" else is_half


class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device

        self.webui_port_main = webui_port_main
        self.webui_port_subfix = webui_port_subfix

        self.api_port = api_port
