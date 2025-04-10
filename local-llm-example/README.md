# Local LLM examples

Prerequisites:
- Download Ollama client: https://ollama.com/download 
- Install python Ollama package: `pip install ollama`
- Load two models:
    - llama3.2 for text processing: `ollama run llama3.2:3b`
    - llama3.2-vision for image processing: `ollama run llama3.2-vision`

Optional:
- Create a new conda environment with python3.11
  - `conda create -n ollama python=3.11`
- Install open-webui platform: https://github.com/open-webui/open-webui
- Launch open-webui in terminal: `open-webui serve`