# llama2-chatbot-on-cpu-dev

Exploration of concepts in https://www.youtube.com/watch?v=kXuHxI5ZcG0

Building a medical chatbot using all open source tools and llama2 models on machines with only cpu.

Strayed from video and used the gguf model from here:
https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf and re-configured to use a cuda container and the gpu.


## To start in dev
- bind mount for the local directory so you can alter model, etc.
- docker compose up -d
- docker exec -it cuda-llama-chat /bin/bash
- chainlit run ./model.py -w
- compose has 8082:8000 so http://localhost:8082

- then docker compose down when finished.
