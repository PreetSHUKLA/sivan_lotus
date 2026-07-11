# Sivan AI

Sivan AI is a **local, offline-first desktop chat application** that runs open-source LLMs (Llama 3, DeepSeek Coder, Qwen2.5 Coder, Mistral, Gemma 2, etc.) entirely on your own machine via `llama-cpp-python`. It ships with a modern dark-themed GUI, a RAM-aware mode system, file-based RAG (retrieval-augmented generation), and an optional step-by-step "Think Mode."

No API keys required to run it, no data leaves your computer, and once a model is downloaded the whole thing works fully offline.

## Features

- **Runs 100% locally** — inference happens on your CPU (or GPU, optionally) via `llama.cpp`; nothing is sent to the cloud except the one-time model download from Hugging Face.
- **Model catalogue** — pick from a curated list of coding, general-purpose, and reasoning GGUF models, hot-swappable without losing chat history.
- **RAM Modes** — `lite` / `balanced` / `thinking` presets tune context window size, max output tokens, and history depth to match your hardware.
- **RAG (file grounding)** — load `.txt`, `.py`, `.md`, `.json`, `.csv`, or `.pdf` files and Sivan will retrieve the most relevant chunks (TF-IDF) and inject them into its answers.
- **Think Mode** — toggles chain-of-thought reasoning; the reasoning itself is filtered out of the visible reply and out of stored history, so you only see (and pay context budget for) the final answer.
- **Context Budget bar** — live view of how full the model's context window is, color-coded green/yellow/red.
- **Chat export** — save any conversation to a Markdown file.
- **Packageable** — includes a PyInstaller `.spec` file to build a standalone Windows/Mac/Linux executable.

## Project Structure

| File | Purpose |
|---|---|
| `engine_core.py` | The engine: model catalogue, RAM modes, model download/load, RAG (TF-IDF + PDF extraction), prompt formatting, context budget math, and a CLI (`python engine_core.py`) |
| `sivan_gui.py` | The desktop GUI (built with `customtkinter`) — the recommended way to use Sivan AI |
| `set_keys.py` | One-time helper to store your Hugging Face token in the OS keychain (alternative to the `.env` method below) |
| `SivanAI.spec` | PyInstaller spec for building a standalone `.exe` / app bundle |

## Requirements

- Python 3.10+
- A machine with at least 8 GB RAM (16 GB+ recommended for the `thinking` mode or larger models)
- Optional: an NVIDIA GPU for faster inference (via `n_gpu_layers`)

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/<your-username>/sivan-ai.git
   cd sivan-ai
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install customtkinter huggingface_hub llama-cpp-python scikit-learn python-dotenv pymupdf keyring
   ```
   > `llama-cpp-python` may require build tools on some platforms. If you want GPU acceleration, install it with the appropriate `CMAKE_ARGS` for your hardware (see the [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python)).

## Configuration (Hugging Face token — optional)

A token is only needed for gated/private model repos; the default catalogue works without one. Use **either** method:

**Option A — `.env` file (used by the GUI's Settings tab):**
```
HF_TOKEN=hf_your_token_here
N_GPU_LAYERS=0
DEFAULT_MODE=balanced
MODEL_CACHE_DIR=
```
You can also fill these in directly from the GUI's ⚙️ **Settings** tab and click **Save Config** — it writes this file for you.

**Option B — OS keychain:**
```bash
python set_keys.py
```
This stores `HF_TOKEN` securely via `keyring` instead of a plaintext file. Edit the token value inside `set_keys.py` before running it.

## Usage

### GUI (recommended)
```bash
python sivan_gui.py
```
1. Go to the **🧠 Engine** tab, pick a model and a RAM mode, then click **⚡ Load Selected Model** (this downloads the model on first run — expect a wait depending on size/connection).
2. Go to the **📚 Knowledge** tab and click **📂 Load Files...** to add files for RAG grounding, if needed.
3. Type in the message box and hit **Send 🚀**. Toggle **Think Mode** for step-by-step reasoning.
4. Use **🧹 Clear Chat** to reset the conversation, or **📥 Export Chat** to save it as Markdown.

### CLI
```bash
python engine_core.py
```
Follow the prompts to pick a RAM mode and model, then chat directly in the terminal. Type `help` at any time for the full command list (`load`, `clear-file`, `clear-chat`, `save-chat`, `think-on` / `think-off`, `mode <name>`, `switch-model`, `info`, `exit`).

## Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller SivanAI.spec
```
The packaged app will appear in `dist/SivanAI/`.

## Notes

- Everything (model files) downloads to your local Hugging Face cache (or your custom `MODEL_CACHE_DIR`) — subsequent runs work fully offline.
- Model choice and RAM mode both affect memory usage; if you hit out-of-memory errors, switch to a smaller model or the `lite` mode.
- Larger/quantized models trade quality for speed and RAM usage — check the label next to each model in the catalogue for guidance.
