import os
import sys
import traceback
import numpy as np

from GPT_SoVITS.my_utils import load_audio
from scipy.io import wavfile
from slicer import Slicer


def slice(
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
    all_part,
):
    os.makedirs(opt_root, exist_ok=True)
    if os.path.isfile(inp):
        input = [inp]
    elif os.path.isdir(inp):
        input = [os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "Input path exists but is neither a file nor a folder"
    slicer = Slicer(
        sr=32000,
        threshold=int(threshold),
        min_length=int(min_length),
        min_interval=int(min_interval),
        hop_size=int(hop_size),
        max_sil_kept=int(max_sil_kept),
    )
    _max = float(_max)
    alpha = float(alpha)
    for inp_path in input[int(i_part) :: int(all_part)]:
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path, "fail ->", traceback.format_exc())


slice(*sys.argv[1:])
