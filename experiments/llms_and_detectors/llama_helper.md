
## Running custom LLMs with llama.cpp

Models are downloaded from **Unsloth** repository:
1. Gemma4-E4B https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
2. Gemma3-4B https://huggingface.co/unsloth/gemma-3-4b-it-GGUF
3. Qwen3.5-4B https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
4. Qwen3.6-35B https://huggingface.co/Qwen/Qwen3.6-35B-A3B

Vision LLMs are served locally via [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server`. The detection client (`run_detection_with_llm.py`) talks to an OpenAI-compatible API on **port 8010**.

Start a model before running detection. Paths are relative to wherever your GGUF weights live (adjust as needed):

**Gemma 3 4B**

```bash
llama-server \
  --model Models/gemma/gemma-3-4b-it-GGUF/gemma-3-4b-it-UD-Q4_K_XL.gguf \
  --mmproj Models/gemma/gemma-3-4b-it-GGUF/mmproj-F16.gguf \
  --port 8010 \
  --temp 0.0
```

**Qwen 3.5 4B**

```bash
llama-server \
  --model Models/qwen/Qwen3.5-4B-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --mmproj Models/qwen/Qwen3.5-4B-GGUF/mmproj-F16.gguf \
  --port 8010 \
  --temp 0.0
```

`--mmproj` is required for vision (multimodal) models. `--temp 0.0` keeps outputs deterministic for reproducible detection runs. Once the server is up, run `run_detection_with_llm.py` against `http://127.0.0.1:8010/v1`.
