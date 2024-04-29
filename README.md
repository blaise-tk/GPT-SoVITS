**Training:**

- **Preprocessing Stage 1:** Convert audio to text using Hubert, then encode the text with BERT.
- **Stage 1:** Utilize Hubert to encode audio into tokens, supplemented with text and reference encoder embeddings to reconstruct the audio (via Sovits).
- **Preprocessing Stage 2:** Further tokenization of audio using Hubert.
- **Stage 2:** Combine tokens, BERT embeddings, and text to refine token sequences using GPT (referred to as Soundstorm stage_AR).

**Fine-tuning:**

- **Preprocessing Stage:** Convert audio to tokens using Hubert, and text to embeddings with BERT.
- **Stage 1:** Encode tokens, incorporating text and reference encoder embeddings to generate audio output via Sovits_decoder.
- **Stage 2:** Refine token sequences using GPT, considering tokens, BERT embeddings, and text.

**Inference:**

- Extract BERT embeddings from text.
- Convert prompt audio to prompt tokens using Sovits_encoder.
- Merge prompt tokens, task-specific text, and BERT embeddings to generate completed tokens using GPT.
- Combine completed tokens, task-specific text, and reference encoder embeddings to produce the final vocal output via Sovits_decoder.
