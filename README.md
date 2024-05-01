# GPT-SoVITS Fork

## Installation

### Windows

To install the GPT-SoVITS Fork on Windows, simply run `run-install.bat` followed by `run-webui.bat` to start the application.

## Technical Details

### Training:

- **Preprocessing Stage 1:** Audio-to-text conversion using Hubert, followed by text encoding with BERT.
- **Stage 1:** Utilizes Hubert to tokenize audio, integrating text and reference encoder embeddings for audio reconstruction (via Sovits).
- **Preprocessing Stage 2:** Further tokenization of audio using Hubert.
- **Stage 2:** Refinement of token sequences using GPT, incorporating tokens, BERT embeddings, and text (referred to as Soundstorm stage_AR).

### Fine-tuning:

- **Preprocessing Stage:** Conversion of audio to tokens using Hubert, and text to embeddings with BERT.
- **Stage 1:** Encoding of tokens, integrating text and reference encoder embeddings to generate audio output via Sovits_decoder.
- **Stage 2:** Token sequence refinement using GPT, considering tokens, BERT embeddings, and text.

### Inference:

- BERT embeddings extraction from text.
- Conversion of prompt audio to prompt tokens using Sovits_encoder.
- Merging of prompt tokens, task-specific text, and BERT embeddings to generate completed tokens using GPT.
- Combining completed tokens, task-specific text, and reference encoder embeddings to produce the final vocal output via Sovits_decoder.

## Acknowledgements

### Theoretical

- [GPT-SoVITS](https://github.com/rvc-boss/GPT-SoVITS)
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)

### Pretrained Models

- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)

### Text Frontend for Inference

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [LangSegment](https://github.com/juntaosun/LangSegment)

### WebUI Tools

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
