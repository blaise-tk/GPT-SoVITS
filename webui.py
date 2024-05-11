import os
import re
import sys
import json
import site
import yaml
import psutil
import signal
import shutil
import librosa
import warnings
import platform
import traceback
import LangSegment
import numpy as np
import gradio as gr

from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.my_utils import clean_path, load_audio

from subprocess import Popen
from multiprocessing import cpu_count
from tools.asr.config import asr_dict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from config import (
    python_exec,
    is_half,
    exp_root,
    webui_port_subfix,
    is_share,
)


from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

import logging

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

import torch

torch.manual_seed(233333)

if os.path.exists("./logs/gweight.txt"):
    with open("./logs/gweight.txt", "r", encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get("gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
        "gpt_path",
        "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    )

if os.path.exists("./logs/sweight.txt"):
    with open("./logs/sweight.txt", "r", encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get(
        "sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth"
    )

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
cnhubert.cnhubert_base_path = cnhubert_base_path

bert_path = os.environ.get(
    "bert_path",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
)
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
warnings.filterwarnings("ignore")

tmp = os.path.join(now_dir, "logs", "temp")
os.makedirs(tmp, exist_ok=True)
os.environ["temp"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if site_packages_roots == []:
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass


n_cpu = cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "L4",
                "4060",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "%s\t%s" % ("0", "CPU")
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    default_batch_size = psutil.virtual_memory().total / 1024 / 1024 / 1024 / 2
gpus = "-".join([i[0] for i in gpu_infos])

pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = (
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    if "pretrained" not in sovits_path:
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./logs/sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    with open("./logs/gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path)


change_gpt_weights(gpt_path)
change_sovits_weights(sovits_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    i18n("Chinese"): "all_zh",
    i18n("English"): "en",
    i18n("Japanese"): "all_ja",
    i18n("Chinese-English mix"): "zh",
    i18n("Japanese-English mix"): "ja",
    i18n("Multilingual mix"): "auto",
}


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    return phones, bert.to(dtype), norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if text_language != "en" else "." + text

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise OSError(
                i18n(" The reference audio is outside the range of 3 to 10 seconds")
            )
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]

    if how_to_cut == i18n("Split into four lines"):
        text = cut1(text)
    elif how_to_cut == i18n("Split every 50 characters"):
        text = cut2(text)
    elif how_to_cut == i18n("Split by Chinese punctuation marks"):
        text = cut3(text)
    elif how_to_cut == i18n("Split by English punctuation marks"):
        text = cut4(text)
    elif how_to_cut == i18n("Split by punctuation marks"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            )
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, ref_wav_path)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        audio = (
            vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(device).unsqueeze(0),
                refer,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


def cut5(inp):
    inp = inp.strip("\n")
    punds = r"[,.;?!、，。？！;：…]"
    items = re.split(f"({punds})", inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def custom_sort_key(s):
    parts = re.split("(\d+)", s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {
        "choices": sorted(SoVITS_names, key=custom_sort_key),
        "__type__": "update",
    }, {
        "choices": sorted(GPT_names, key=custom_sort_key),
        "__type__": "update",
    }


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = (
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)
SoVITS_weight_root = "logs/weights/SoVITS"
GPT_weight_root = "logs/weights/GPT"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [os.path.join(SoVITS_weight_root, pretrained_sovits_name)]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):
            SoVITS_names.append(os.path.join(SoVITS_weight_root, name))
    GPT_names = [os.path.join(GPT_weight_root, pretrained_gpt_name)]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"):
            GPT_names.append(os.path.join(GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):
            SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"):
            GPT_names.append(name)
    return SoVITS_names, GPT_names


SoVITS_weight_root = "logs/weights/SoVITS"
GPT_weight_root = "logs/weights/GPT"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()


def custom_sort_key(s):
    parts = re.split("(\d+)", s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {
        "choices": sorted(SoVITS_names, key=custom_sort_key),
        "__type__": "update",
    }, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


p_label = None
p_asr = None
p_denoise = None
p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)
        except OSError:
            pass


system = platform.system()


def kill_process(pid):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def change_label(if_label, path_list):
    global p_label
    if if_label == True and p_label == None:
        path_list = clean_path(path_list)
        cmd = '"%s" subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec,
            path_list,
            webui_port_subfix,
            is_share,
        )
        yield i18n("Marking tool WebUI is enabled")

        p_label = Popen(cmd, shell=True)
    elif if_label == False and p_label != None:
        kill_process(p_label.pid)
        p_label = None
        yield i18n("Marking tool WebUI is disabled")


def open_asr(
    asr_inp_dir, asr_opt_dir, asr_model_size, asr_model="Whisper", asr_lang="auto"
):
    global p_asr
    if p_asr == None:
        asr_inp_dir = clean_path(asr_inp_dir)
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f" -s {asr_model_size}"
        cmd += f" -l {asr_lang}"
        cmd += " -p %s" % ("float16" if is_half == True else "float32")

        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }

        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield f"Successfully completed!", {
            "__type__": "update",
            "visible": True,
        }, {"__type__": "update", "visible": False}
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close_asr():
    global p_asr
    if p_asr != None:
        kill_process(p_asr.pid)
        p_asr = None
    return (
        "已终止ASR进程",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = clean_path(denoise_inp_dir)
        denoise_opt_dir = clean_path(denoise_opt_dir)
        cmd = '"%s" tools/cmd_denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec,
            denoise_inp_dir,
            denoise_opt_dir,
            "float16" if is_half == True else "float32",
        )

        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }

        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        yield f"Successfully completed!", {
            "__type__": "update",
            "visible": True,
        }, {"__type__": "update", "visible": False}
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close_denoise():
    global p_denoise
    if p_denoise != None:
        kill_process(p_denoise.pid)
        p_denoise = None
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_SoVITS = None


def open1Ba(
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root
        data["name"] = exp_name
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/sovits_train.py --config "%s"' % (
            python_exec,
            tmp_config_path,
        )
        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }

        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        yield "Successfully completed!", {"__type__": "update", "visible": True}, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS != None:
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS = None
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_GPT = None


def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        with open("GPT_SoVITS/configs/s1longer.yaml") as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1" % s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        cmd = '"%s" GPT_SoVITS/gpt_train.py --config_file "%s" ' % (
            python_exec,
            tmp_config_path,
        )
        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }

        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        yield "Successfully completed!", {"__type__": "update", "visible": True}, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1Bb():
    global p_train_GPT
    if p_train_GPT != None:
        kill_process(p_train_GPT.pid)
        p_train_GPT = None
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps_slice = []


def open_slice(
    inp,
    opt_root,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    _max,
    alpha,
    n_parts,
):
    global ps_slice
    inp = clean_path(inp)
    opt_root = clean_path(opt_root)
    if os.path.exists(inp) == False:
        yield "Input path does not exist", {"__type__": "update", "visible": True}, {
            "__type__": "update",
            "visible": False,
        }
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield "Input path exists but is neither a file nor a folder", {
            "__type__": "update",
            "visible": True,
        }, {"__type__": "update", "visible": False}
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = (
                '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s'
                ""
                % (
                    python_exec,
                    inp,
                    opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    i_part,
                    n_parts,
                )
            )
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "In progress...", {"__type__": "update", "visible": False}, {
            "__type__": "update",
            "visible": True,
        }
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield "Successfully completed!", {"__type__": "update", "visible": True}, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close_slice():
    global ps_slice
    if ps_slice != []:
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice = []
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1a = []


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = clean_path(inp_text)
    inp_wav_dir = clean_path(inp_wav_dir)
    if ps1a == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    "is_half": str(is_half),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec

            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield "In progress...", {"__type__": "update", "visible": False}, {
            "__type__": "update",
            "visible": True,
        }
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        yield "Successfully completed!", {"__type__": "update", "visible": True}, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1a():
    global ps1a
    if ps1a != []:
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a = []
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1b = []


def open1b(inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = clean_path(inp_text)
    inp_wav_dir = clean_path(inp_wav_dir)
    if ps1b == []:
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = (
                '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            )

            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }
        for p in ps1b:
            p.wait()
        ps1b = []
        yield "Successfully completed!", {
            "__type__": "update",
            "visible": True,
        }, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1b():
    global ps1b
    if ps1b != []:
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b = []
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []


def open1c(inp_text, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = clean_path(inp_text)
    if ps1c == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": "GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec

            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "In progress...", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        yield "Successfully completed!", {
            "__type__": "update",
            "visible": True,
        }, {
            "__type__": "update",
            "visible": False,
        }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1c():
    global ps1c
    if ps1c != []:
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c = []
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1abc = []


def open1abc(
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    global ps1abc
    inp_text = clean_path(inp_text)
    inp_wav_dir = clean_path(inp_wav_dir)
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                os.path.exists(path_text) == True
                and len(
                    open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")
                )
                < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec

                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "In progress...", {"__type__": "update", "visible": False}, {
                    "__type__": "update",
                    "visible": True,
                }
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(all_parts):
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
            ps1abc = []
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = (
                    '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
                    % python_exec
                )

                p = Popen(cmd, shell=True)
                ps1abc.append(p)

            for p in ps1abc:
                p.wait()
            ps1abc = []
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True
                and os.path.getsize(path_semantic) < 31
            ):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = (
                        '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'
                        % python_exec
                    )

                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            ps1abc = []
            yield "Successfully completed!", {
                "__type__": "update",
                "visible": True,
            }, {
                "__type__": "update",
                "visible": False,
            }
        except:
            traceback.print_exc()
            close1abc()
            yield "Error", {
                "__type__": "update",
                "visible": True,
            }, {
                "__type__": "update",
                "visible": False,
            }
    else:
        yield "There is already a task in progress.", {
            "__type__": "update",
            "visible": False,
        }, {
            "__type__": "update",
            "visible": True,
        }


def close1abc():
    global ps1abc
    if ps1abc != []:
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc = []
    return (
        "All processes have been terminated.",
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


with gr.Blocks(title="GPT-SoVITS-Fork", theme="remilia/Ghostly") as app:
    gr.Markdown(value=i18n("GPT-SoVITS-Fork"))

    with gr.Tabs():
        with gr.TabItem(i18n("Data Processor")):
            with gr.Accordion(i18n("Audio Splicer")):
                with gr.Row():
                    slice_inp_path = gr.Textbox(
                        label=i18n("Input root"),
                        placeholder=i18n("Folder or audio path"),
                    )
                    slice_opt_root = gr.Textbox(
                        label=i18n("Output root"),
                        value="logs/output/slicer_opt",
                    )
                with gr.Accordion(i18n("Advanced Settings"), open=False):
                    with gr.Column():
                        with gr.Row():
                            threshold = gr.Textbox(
                                label=i18n("Threshold"),
                                value="-34",
                            )
                            min_length = gr.Textbox(
                                label=i18n("Min Length"),
                                value="4000",
                            )
                            min_interval = gr.Textbox(
                                label=i18n("Minimum cutting interval"), value="300"
                            )
                            hop_size = gr.Textbox(
                                label=i18n("Hop Size"),
                                value="10",
                            )
                            max_sil_kept = gr.Textbox(
                                label=i18n("Max Sil"),
                                value="500",
                            )
                        with gr.Row():
                            _max = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("Maximum value after normalization"),
                                value=0.9,
                                interactive=True,
                            )
                            alpha = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n(
                                    "Proportion of normalized audio to be mixed in"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            n_process = gr.Slider(
                                minimum=1,
                                maximum=n_cpu,
                                step=1,
                                label=i18n("Number of processes used for cutting"),
                                value=4,
                                interactive=True,
                            )
                with gr.Column():
                    open_slicer_button = gr.Button(
                        i18n("Run Slicer"), variant="primary", visible=True
                    )
                    close_slicer_button = gr.Button(
                        i18n("Stop Slicer"),
                        variant="primary",
                        visible=False,
                    )
                    slicer_info = gr.Textbox(label=i18n("Output Information"))
            with gr.Accordion(i18n("Optional: Noise Reduction"), open=False):
                with gr.Row():
                    denoise_input_dir = gr.Textbox(
                        label=i18n("Input root"), value="logs/output/slicer_opt"
                    )
                    denoise_output_dir = gr.Textbox(
                        label=i18n("Output root"), value="logs/output/denoise_opt"
                    )
                with gr.Column():
                    open_denoise_button = gr.Button(
                        i18n("Run Noise Reduction"), variant="primary", visible=True
                    )
                    close_denoise_button = gr.Button(
                        i18n("Stop Noise Reduction"), variant="primary", visible=False
                    )
                    denoise_info = gr.Textbox(label=i18n("Output Information"))
            with gr.Accordion(i18n("Automatic text labeling")):
                with gr.Column():
                    with gr.Row():
                        asr_inp_dir = gr.Textbox(
                            label=i18n("Input root"),
                            value="logs/output/slicer_opt",
                            interactive=True,
                        )
                        asr_opt_dir = gr.Textbox(
                            label=i18n("Output root"),
                            value="logs/output/asr_opt",
                            interactive=True,
                        )
                    with gr.Row():
                        asr_size = gr.Dropdown(
                            label=i18n("Labeling Type"),
                            choices=[
                                "tiny",
                                "tiny.en",
                                "base",
                                "base.en",
                                "small",
                                "small.en",
                                "medium",
                                "medium.en",
                                "large",
                                "large-v1",
                                "large-v2",
                                "large-v3",
                            ],
                            interactive=True,
                            value="medium",
                        )
                    with gr.Column():
                        open_asr_button = gr.Button(
                            i18n("Run Labeling"), variant="primary", visible=True
                        )
                        close_asr_button = gr.Button(
                            i18n("Stop Labeling"), variant="primary", visible=False
                        )
                        asr_info = gr.Textbox(label=i18n("Output Information"))

                def change_lang_choices(key):
                    return {
                        "__type__": "update",
                        "choices": asr_dict[key]["lang"],
                        "value": asr_dict[key]["lang"][0],
                    }

                def change_size_choices(key):
                    return {"__type__": "update", "choices": asr_dict[key]["size"]}

            with gr.Accordion(
                i18n("Optional: Manual text labelling reviewer"), open=False
            ):
                if_label = gr.Checkbox(
                    label=i18n("Launch Manual text labelling reviewer WebUI"),
                    show_label=True,
                )
                path_list = gr.Textbox(
                    label=i18n("Root to the list labeled file"),
                    value=os.path.join("list_name.list"),
                    interactive=True,
                )

                label_info = gr.Textbox(label=i18n("Output Information"))
            if_label.change(change_label, [if_label, path_list], [label_info])
            open_asr_button.click(
                open_asr,
                [asr_inp_dir, asr_opt_dir, asr_size],
                [asr_info, open_asr_button, close_asr_button],
            )
            close_asr_button.click(
                close_asr, [], [asr_info, open_asr_button, close_asr_button]
            )
            open_slicer_button.click(
                open_slice,
                [
                    slice_inp_path,
                    slice_opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    n_process,
                ],
                [slicer_info, open_slicer_button, close_slicer_button],
            )
            close_slicer_button.click(
                close_slice, [], [slicer_info, open_slicer_button, close_slicer_button]
            )
            open_denoise_button.click(
                open_denoise,
                [denoise_input_dir, denoise_output_dir],
                [denoise_info, open_denoise_button, close_denoise_button],
            )
            close_denoise_button.click(
                close_denoise,
                [],
                [denoise_info, open_denoise_button, close_denoise_button],
            )

        with gr.TabItem(i18n("Training")):
            with gr.Row():
                exp_name = gr.Textbox(
                    label=i18n("Model Name"), value="my-model", interactive=True
                )
                gpu_info = gr.Textbox(
                    label=i18n("GPU Information"),
                    value=gpu_info,
                    visible=True,
                    interactive=False,
                )
                gpu = gr.Textbox(
                    label=i18n(
                        "Split the GPU card numbers using dashes, and assign one process to each card number."
                    ),
                    value="%s" % (gpus),
                    interactive=True,
                )
                pretrained_s2G = gr.Textbox(
                    label=i18n("Pre-trained G SoVITS model path"),
                    value="GPT_SoVITS/pretrained_models/s2G488k.pth",
                    interactive=True,
                    lines=3,
                )
                pretrained_s2D = gr.Textbox(
                    label=i18n("Pre-trained D SoVITS model path"),
                    value="GPT_SoVITS/pretrained_models/s2D488k.pth",
                    interactive=True,
                    lines=3,
                )
                pretrained_s1 = gr.Textbox(
                    label=i18n("Pre-trained GPT model path"),
                    value="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                    interactive=True,
                    lines=3,
                )
            with gr.Accordion(i18n("Dataset Formatter")):
                with gr.Row():
                    inp_text = gr.Textbox(
                        label=i18n("Text labeling file"),
                        placeholder="Path to the text labeling file",
                        interactive=True,
                    )
                    inp_wav_dir = gr.Textbox(
                        label=i18n("Dataset root"),
                        interactive=True,
                        value="logs/output/slicer_opt",
                    )
                    bert_pretrained_dir = gr.Textbox(
                        label=i18n("Pre-trained Chinese BERT model path"),
                        value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                        interactive=False,
                    )
                    cnhubert_base_dir = gr.Textbox(
                        label=i18n("Pre-trained SSL model path"),
                        value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                        interactive=False,
                    )
                button1abc_open = gr.Button(
                    i18n("Run formatter"),
                    variant="primary",
                    visible=True,
                )
                button1abc_close = gr.Button(
                    i18n("Stop formatter"),
                    variant="primary",
                    visible=False,
                )
                info1abc = gr.Textbox(label=i18n("Output Information"))

            button1abc_open.click(
                open1abc,
                [
                    inp_text,
                    inp_wav_dir,
                    exp_name,
                    gpu,
                    gpu,
                    gpu,
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    pretrained_s2G,
                ],
                [info1abc, button1abc_open, button1abc_close],
            )
            button1abc_close.click(
                close1abc, [], [info1abc, button1abc_open, button1abc_close]
            )

            with gr.Accordion(i18n("SoVITS training")):
                with gr.Column():
                    with gr.Row():
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("Batch size"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        total_epoch = gr.Slider(
                            minimum=1,
                            maximum=25,
                            step=1,
                            label=i18n("Total epoch"),
                            value=8,
                            interactive=True,
                        )
                        text_low_lr_rate = gr.Slider(
                            minimum=0.2,
                            maximum=0.6,
                            step=0.05,
                            label=i18n("Text Module Learning Rate Weights"),
                            value=0.4,
                            interactive=True,
                        )
                        save_every_epoch = gr.Slider(
                            minimum=1,
                            maximum=25,
                            step=1,
                            label=i18n("Save frequency"),
                            value=4,
                            interactive=True,
                        )
                    with gr.Row():
                        if_save_latest = gr.Checkbox(
                            label=i18n("Save only the latest checkpoint"),
                            value=True,
                            interactive=True,
                            show_label=True,
                        )
                        if_save_every_weights = gr.Checkbox(
                            label=i18n(
                                "Save final checkpoints at each save time point"
                            ),
                            value=True,
                            interactive=True,
                            show_label=True,
                        )

                with gr.Column():
                    button1Ba_open = gr.Button(
                        i18n("Run SoVITS training"),
                        variant="primary",
                        visible=True,
                    )
                    button1Ba_close = gr.Button(
                        i18n("Stop SoVITS training"),
                        variant="primary",
                        visible=False,
                    )
                    info1Ba = gr.Textbox(label=i18n("Output Information"))
            with gr.Accordion(i18n("GPT Training")):
                with gr.Column():
                    with gr.Row():
                        batch_size1Bb = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("Batch size"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        total_epoch1Bb = gr.Slider(
                            minimum=2,
                            maximum=50,
                            step=1,
                            label=i18n("Total epoch"),
                            value=10,
                            interactive=True,
                        )

                        save_every_epoch1Bb = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("Save frequency"),
                            value=5,
                            interactive=True,
                        )
                    with gr.Row():
                        if_dpo = gr.Checkbox(
                            label=i18n("Experimental: DPO Training"),
                            info=i18n(
                                "DPO training significantly enhances the model's performance and stability, specifically in non-audio applications. It allows for processing larger text inputs without fragmentation and reduces the occurrence of errors such as word repetition or omission during inference."
                            ),
                            value=False,
                            interactive=True,
                            show_label=True,
                        )
                        if_save_latest1Bb = gr.Checkbox(
                            label=i18n("Save only the latest checkpoint"),
                            value=True,
                            interactive=True,
                            show_label=True,
                        )
                        if_save_every_weights1Bb = gr.Checkbox(
                            label=i18n(
                                "Save final checkpoints at each save time point"
                            ),
                            value=True,
                            interactive=True,
                            show_label=True,
                        )

                with gr.Column():
                    button1Bb_open = gr.Button(
                        i18n("Run GPT training"), variant="primary", visible=True
                    )
                    button1Bb_close = gr.Button(
                        i18n("Stop GPT training"), variant="primary", visible=False
                    )
                    info1Bb = gr.Textbox(label=i18n("Output Information"))
            button1Ba_open.click(
                open1Ba,
                [
                    batch_size,
                    total_epoch,
                    exp_name,
                    text_low_lr_rate,
                    if_save_latest,
                    if_save_every_weights,
                    save_every_epoch,
                    gpu,
                    pretrained_s2G,
                    pretrained_s2D,
                ],
                [info1Ba, button1Ba_open, button1Ba_close],
            )
            button1Ba_close.click(
                close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close]
            )
            button1Bb_open.click(
                open1Bb,
                [
                    batch_size1Bb,
                    total_epoch1Bb,
                    exp_name,
                    if_dpo,
                    if_save_latest1Bb,
                    if_save_every_weights1Bb,
                    save_every_epoch1Bb,
                    gpu,
                    pretrained_s1,
                ],
                [info1Bb, button1Bb_open, button1Bb_close],
            )
            button1Bb_close.click(
                close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close]
            )

        with gr.TabItem(i18n("TTS Inference")):
            with gr.Accordion(i18n("Model Selector")):
                with gr.Row():
                    full_gpt_path = [
                        os.path.join(GPT_weight_root, name) for name in GPT_names
                    ]
                    GPT_dropdown = gr.Dropdown(
                        label=i18n("GPT Model"),
                        choices=full_gpt_path,
                        value=gpt_path,
                        interactive=True,
                    )

                    full_sovits_path = [
                        os.path.join(SoVITS_weight_root, name) for name in SoVITS_names
                    ]
                    SoVITS_dropdown = gr.Dropdown(
                        label=i18n("SoVITS Model"),
                        choices=full_sovits_path,
                        value=sovits_path,
                        interactive=True,
                    )
                    refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[SoVITS_dropdown, GPT_dropdown],
                    )

            with gr.Accordion(i18n("Audio Reference")):
                with gr.Row():
                    with gr.Column():
                        ref_text_free = gr.Checkbox(
                            label=i18n("Disable reference text mode."),
                            info=i18n(
                                "It's recommended to use the fine-tuned GPT model without reference text mode enabled. If you're unable to hear the reference audio to generate text, you can turn this mode on to proceed without it."
                            ),
                            value=False,
                            interactive=True,
                            show_label=True,
                        )
                        inp_ref = gr.Audio(
                            label=i18n("Reference audio within 3-10 seconds."),
                            type="filepath",
                        )

                    with gr.Column():
                        prompt_text = gr.Textbox(
                            label=i18n("Text from the reference audio"),
                            value="",
                            lines=4,
                        )
                        prompt_language = gr.Dropdown(
                            label=i18n("Language of reference audio"),
                            choices=[
                                i18n("English"),
                                i18n("Chinese"),
                                i18n("Japanese"),
                                i18n("Chinese-English mix"),
                                i18n("Japanese-English mix"),
                                i18n("Multilingual mix"),
                            ],
                            value=i18n("English"),
                        )

            with gr.Accordion(i18n("TTS Synthesis")):
                with gr.Column():
                    text = gr.Textbox(
                        label=i18n("Text to be synthesized"),
                        value="",
                        lines=4,
                        placeholder=i18n("Please fill in the text to be synthesized."),
                    )
                    with gr.Row():
                        text_language = gr.Dropdown(
                            label=i18n("Language to be synthesized"),
                            choices=[
                                i18n("English"),
                                i18n("Chinese"),
                                i18n("Japanese"),
                                i18n("Chinese-English mix"),
                                i18n("Japanese-English mix"),
                                i18n("Multilingual mix"),
                            ],
                            value=i18n("English"),
                        )
                        how_to_cut = gr.Radio(
                            label=i18n("How to split the sentence"),
                            choices=[
                                i18n("Do not split"),
                                i18n("Split into four lines"),
                                i18n("Split every 50 characters"),
                                i18n("Split by Chinese punctuation marks"),
                                i18n("Split by English punctuation marks"),
                                i18n("Split by punctuation marks"),
                            ],
                            value=i18n("Split into four lines"),
                            interactive=True,
                        )
                    with gr.Row():
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            label=i18n("top_k"),
                            value=5,
                            interactive=True,
                        )
                        top_p = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            label=i18n("top_p"),
                            value=1,
                            interactive=True,
                        )
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            label=i18n("temperature"),
                            value=1,
                            interactive=True,
                        )
                    inference_button = gr.Button(
                        i18n("Run Inference"), variant="primary"
                    )
                    output = gr.Audio(label=i18n("Output Information"))

            inference_button.click(
                get_tts_wav,
                [
                    inp_ref,
                    prompt_text,
                    prompt_language,
                    text,
                    text_language,
                    how_to_cut,
                    top_k,
                    top_p,
                    temperature,
                    ref_text_free,
                ],
                [output],
            )

    app.launch(
        share="--share" in sys.argv, inbrowser="--open" in sys.argv, server_port=6969
    )
