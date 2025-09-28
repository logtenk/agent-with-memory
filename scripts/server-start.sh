sh /llama.cpp/build/bin/llama-server \
  --model /models/lmstudio-community/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf \
  --mmproj /models/lmstudio-community/gemma-3-12b-it-GGUF/mmproj-model-f16.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --metrics
