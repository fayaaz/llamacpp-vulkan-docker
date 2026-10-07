# Llama.cpp Docker Setup (Vulkan)

This repository provides a `docker-compose.yml` configuration to run a `llama.cpp` server with Vulkan support and a convenient Hugging Face model downloader.

This has been tested on a PC with specs:
- CPU: 9950X3D
- RAM: 64GB
- GPU: 9070 XT with 16GB VRAM

and is focused around running Qwen3.8-27B and the Qwen3.6-35B-A3B MoE.

## Services

- **`server`**: Runs the `llama.cpp` server. It is configured to use Vulkan for hardware acceleration.
- **`downloader`**: A Python-based utility container used to download models from Hugging Face using `hf_transfer` for high-speed downloads.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
- An AMD GPU with appropriate Vulkan drivers installed on the host.

## Getting Started


1. **Start the services**:
    Make the .models folder first. If you want this elsewhere change it in the docker-compose.yml

   ```bash
   docker compose up -d
   ```

2. **Download a model**:
    Use the `downloader` service to fetch models from Hugging Face. Replace `<repo_id>` with the Hugging Face repository. Current useful presets in `config.ini`:

    ```bash
    # Qwen 3.5
     docker compose run --rm downloader hf download --local-dir /models bartowski/Qwen_Qwen3.5-27B-GGUF Qwen_Qwen3.5-27B-IQ3_M.gguf
    # Qwen 3.6-35B-A3B
     docker compose run --rm downloader hf download --local-dir /models unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
    # GLM Flash 4.7
     docker compose run --rm downloader hf download --local-dir /models unsloth/GLM-4.7-Flash-GGUF GLM-4.7-Flash-UD-Q4_K_XL.gguf
    # Qwen 3.8-27B uncensored (Heretic abliterated, imatrix)
     docker compose run --rm downloader hf download --local-dir /models JonathanColetti/Qwen3.8-27B-Uncensored-GGUF Qwen3.8-27B-Uncensored-IQ4_XS.gguf

    ```

    *Note: The models will be stored in `~/.models` on your host machine.*

3. **Restart the server**:
   ```bash
   docker compose down && docker compose up -d server
   ```

4. **Access the server**:
   The server will be available at `http://localhost:8080`.

## Configuration

- **Model Storage**: Models are stored in `~/.models` on the host.
- **Cache**: Hugging Face metadata and blobs are cached in `~/.models/.cache`.
