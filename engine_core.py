"""
SIVAN AI — Engine Core v6
RAG | PDF | Multi-File | Modes | TF-IDF | GPU | .env config
"""

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import sys
import os
import re
import time
import datetime

# =================================================================
# .ENV LOADING  (python-dotenv — gracefully skipped if not installed)
# =================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_OK = True
except ImportError:
    _DOTENV_OK = False

def _env(key, default=None):
    """Read a value from environment / .env file."""
    return os.environ.get(key, default)

print("==================================================")
print("   SIVAN AI — ENGINE CORE v6                     ")
print("   RAG | PDF | Multi-File | Modes | GPU | .env   ")
print("==================================================\n")

if not _DOTENV_OK:
    print("[ENV] python-dotenv not installed — running: pip install python-dotenv")
    os.system("pip install python-dotenv -q")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        _DOTENV_OK = True
        print("[ENV] Loaded .env successfully.\n")
    except Exception:
        print("[ENV] Warning: could not load .env. Proceeding with system environment only.\n")

# =================================================================
# RAM MODES
#
#  MODE       n_ctx   max_tokens  MAX_RAG_CHARS  TOP_K  history
#  lite       2048    256         800            2      6 msgs
#  balanced   4096    512         2000           4      12 msgs
#  thinking   10240   1500        4000           6      20 msgs
# =================================================================
MODES = {
    "lite": {
        "n_ctx":         2048,
        "max_tokens":    256,
        "MAX_RAG_CHARS": 800,
        "TOP_K_CHUNKS":  2,
        "max_history":   6,
        "label":         "LITE     | Low RAM  | Fast replies   | 2K ctx",
    },
    "balanced": {
        "n_ctx":         4096,
        "max_tokens":    512,
        "MAX_RAG_CHARS": 2000,
        "TOP_K_CHUNKS":  4,
        "max_history":   12,
        "label":         "BALANCED | Mid RAM  | Good quality   | 4K ctx",
    },
    "thinking": {
        "n_ctx":         10240,
        "max_tokens":    1500,
        "MAX_RAG_CHARS": 4000,
        "TOP_K_CHUNKS":  6,
        "max_history":   20,
        "label":         "THINKING | High RAM | Deep reasoning | 10K ctx",
    },
}
DEFAULT_MODE = _env("DEFAULT_MODE", "balanced")
if DEFAULT_MODE not in MODES:
    DEFAULT_MODE = "balanced"

# =================================================================
# MODEL CATALOGUE
# Each entry has:
#   name          — display label shown in GUI
#   repo          — HuggingFace repo id
#   file          — GGUF filename inside that repo
#   prompt_format — which prompt template to use:
#                   "llama3" | "mistral" | "chatml" | "gemma"
# =================================================================
MODEL_CATALOGUE = {
    "coding": [
        {
            "name":          "1. Llama-3 8B Instruct — Best all-round coding + chat",
            "repo":          "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
            "file":          "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "prompt_format": "llama3",
        },
        {
            "name":          "2. DeepSeek Coder V2 Lite 7B — Specialised code generation",
            "repo":          "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "file":          "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
            "prompt_format": "chatml",
        },
        {
            "name":          "3. Qwen2.5 Coder 7B Instruct — Strong code + explanation",
            "repo":          "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "file":          "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "prompt_format": "chatml",
        },
        {
            "name":          "4. Qwen2.5 Coder 14B — Best local coding (needs 12 GB RAM)",
            "repo":          "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
            "file":          "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            "prompt_format": "chatml",
        },
        {
            "name":          "5. DeepSeek-R1 Distill Qwen 7B — Reasoning + coding hybrid",
            "repo":          "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "file":          "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "prompt_format": "chatml",
        },
    ],
    "general": [
        {
            "name":          "1. Llama-3.2 1B Instruct — Ultra-Lightweight (8 GB RAM)",
            "repo":          "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "file":          "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "prompt_format": "llama3",
        },
        {
            "name":          "2. Llama-3.2 3B Instruct — Latest Meta general model",
            "repo":          "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "file":          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "prompt_format": "llama3",
        },
        {
            "name":          "3. Llama-3.1 8B Instruct — 128 K context, upgraded Llama",
            "repo":          "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "file":          "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "prompt_format": "llama3",
        },
        {
            "name":          "4. Phi-3.5 Mini 3.8B — Ultra-light, very fast",
            "repo":          "bartowski/Phi-3.5-mini-instruct-GGUF",
            "file":          "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "prompt_format": "chatml",
        },
    ],
    "reasoning": [
        {
            "name":          "1. Mistral 7B Instruct v0.3 — Fast, reliable instruction following",
            "repo":          "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "file":          "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "prompt_format": "mistral",
        },
        {
            "name":          "2. Gemma-2 9B Instruct — Google's best open model at this size",
            "repo":          "bartowski/gemma-2-9b-it-GGUF",
            "file":          "gemma-2-9b-it-Q4_K_M.gguf",
            "prompt_format": "gemma",
        },
        {
            "name":          "3. Llama-3.3 70B Instruct Q2 — Near-GPT-4 quality (32 GB RAM)",
            "repo":          "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "file":          "Llama-3.3-70B-Instruct-Q2_K.gguf",
            "prompt_format": "llama3",
        },
    ],
}


def show_catalogue_and_pick():
    print("\n============ SIVAN MODEL CATALOGUE ============")
    for cat, models in MODEL_CATALOGUE.items():
        print(f"\n--- {cat.upper()} MODELS ---")
        for m in models:
            print(f"  {m['name']}")
    print("\nType category-number  e.g.  coding-1  /  general-3  /  reasoning-2")
    while True:
        choice = input("\nYour choice: ").strip().lower()
        try:
            cat, num = choice.rsplit("-", 1)
            idx = int(num) - 1
            model_info = MODEL_CATALOGUE[cat][idx]
            print(f"\n[OK] Selected: {model_info['name']}\n")
            return model_info
        except Exception:
            print("[!] Invalid. Try e.g.  coding-1  /  reasoning-2")


def pick_mode():
    print("\n============ SELECT RAM MODE ============")
    for key, cfg in MODES.items():
        print(f"  {key:<10} — {cfg['label']}")
    print("\nRecommendation:")
    print("  lite      — 4 GB RAM or less / quick Q&A")
    print("  balanced  — 8 GB RAM / everyday use   [DEFAULT]")
    print("  thinking  — 16 GB+ RAM / deep analysis / large RAG")
    while True:
        choice = input(f"\nMode (lite/balanced/thinking) [{DEFAULT_MODE}]: ").strip().lower()
        if choice == "":
            choice = DEFAULT_MODE
        if choice in MODES:
            print(f"\n[OK] Mode: {MODES[choice]['label']}\n")
            return choice
        print("[!] Invalid. Enter: lite / balanced / thinking")


# =================================================================
# DOWNLOAD + LOAD MODEL
# =================================================================
def download_model(model_info):
    """Download GGUF from HuggingFace, using HF_TOKEN from .env if present."""
    hf_token  = _env("HF_TOKEN")
    cache_dir = _env("MODEL_CACHE_DIR") or None

    token_display = ("(token from .env)" if hf_token and hf_token != "your_huggingface_token_here"
                     else "(no token — public repos only)")
    print(f"Downloading: {model_info['file']} … {token_display}")

    try:
        kwargs = dict(repo_id=model_info["repo"], filename=model_info["file"])
        if hf_token and hf_token != "your_huggingface_token_here":
            kwargs["token"] = hf_token
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        path = hf_hub_download(**kwargs)
        print(f"[OK] Model ready: {path}\n")
        return path
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)


def load_model(model_path, mode_cfg):
    """Load a GGUF model, optionally offloading layers to GPU via N_GPU_LAYERS."""
    n_gpu = int(_env("N_GPU_LAYERS", "0"))
    n_threads = max(1, (os.cpu_count() or 4) - 1)

    gpu_tag = f"n_gpu_layers={n_gpu}" if n_gpu != 0 else "CPU only"
    print(f"Loading engine  [n_ctx={mode_cfg['n_ctx']} | threads={n_threads} | {gpu_tag}] …")

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=mode_cfg["n_ctx"],
            n_threads=n_threads,
            n_batch=512,
            n_gpu_layers=n_gpu,
            use_mmap=True,
            verbose=False,
        )
        print(f"[OK] Engine loaded.\n")
        return llm
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        sys.exit(1)


# =================================================================
# PROMPT FORMAT ROUTER
# Supports: llama3 | mistral | chatml | gemma
# =================================================================
def compile_prompt(history, prompt_format="llama3"):
    """
    Build a model-specific prompt string from the chat history list.
    history = [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    fmt = (prompt_format or "llama3").lower()

    if fmt == "llama3":
        # Meta Llama-3 / 3.1 / 3.2 / 3.3
        out = ""
        for msg in history:
            out += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>"
            )
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out

    elif fmt == "mistral":
        # Mistral Instruct — no system role; prepend system to first user turn
        out = ""
        system_content = ""
        turns = []
        for msg in history:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                turns.append(msg)

        for i, msg in enumerate(turns):
            if msg["role"] == "user":
                content = msg["content"]
                if i == 0 and system_content:
                    content = system_content + "\n\n" + content
                out += f"[INST] {content} [/INST]"
            elif msg["role"] == "assistant":
                out += f" {msg['content']}</s>"
        return out

    elif fmt == "chatml":
        # ChatML — used by Qwen, DeepSeek, Phi-3.5
        out = ""
        for msg in history:
            out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        out += "<|im_start|>assistant\n"
        return out

    elif fmt == "gemma":
        # Gemma-2 Instruct
        out = ""
        system_content = ""
        for msg in history:
            if msg["role"] == "system":
                system_content = msg["content"]
                continue
            if msg["role"] == "user":
                content = msg["content"]
                if system_content:
                    content = system_content + "\n" + content
                    system_content = ""   # only inject once
                out += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif msg["role"] == "assistant":
                out += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
        out += "<start_of_turn>model\n"
        return out

    else:
        # Unknown format — fall back to llama3
        print(f"[WARN] Unknown prompt_format '{prompt_format}', falling back to llama3.")
        return compile_prompt(history, "llama3")


# Keep this alias so the GUI doesn't break during a transition
def compile_llama3_prompt(history):
    return compile_prompt(history, "llama3")


# =================================================================
# SKLEARN TF-IDF RETRIEVER
# =================================================================
def _ensure_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
        from sklearn.metrics.pairwise import cosine_similarity        # noqa
        return True
    except ImportError:
        print("[TF-IDF] sklearn not found — installing scikit-learn …")
        ret = os.system("pip install scikit-learn -q")
        if ret != 0:
            print("[TF-IDF] Install failed. Falling back to word-overlap scorer.")
            return False
        return True


_SKLEARN_OK = _ensure_sklearn()


def retrieve_relevant_chunks(query, rag_store, top_k):
    """
    Score chunks from ALL loaded files with sklearn TF-IDF + cosine similarity.
    Falls back to word-overlap if sklearn is unavailable.
    Returns list of (score, filename, chunk_text).
    """
    all_chunks = []
    for fname, chunks in rag_store.items():
        for c in chunks:
            all_chunks.append((fname, c))

    if not all_chunks:
        return []

    texts = [c for _, c in all_chunks]

    if _SKLEARN_OK:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            corpus = texts + [query]
            vec = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
            )
            tfidf_matrix = vec.fit_transform(corpus)
            query_vec    = tfidf_matrix[-1]
            chunk_vecs   = tfidf_matrix[:-1]
            scores       = cosine_similarity(query_vec, chunk_vecs).flatten()
            top_indices  = scores.argsort()[::-1][:top_k]
            results = [
                (float(scores[i]), all_chunks[i][0], all_chunks[i][1])
                for i in top_indices if scores[i] > 0
            ]
            if results:
                return results
            # All TF-IDF scores were 0 (short query, code-heavy content, etc.)
            # — fall back to returning the top chunks anyway rather than
            # silently injecting nothing, since files ARE loaded and the
            # user expects them to be used.
            return [
                (float(scores[i]), all_chunks[i][0], all_chunks[i][1])
                for i in top_indices
            ]
        except Exception as e:
            print(f"[TF-IDF WARNING] {e} — using fallback scorer.")

    # Fallback: word-overlap
    STOPWORDS = {
        "the","a","an","is","in","of","to","and","for","with","on","at","by",
        "from","that","this","it","are","was","be","as","or","but","not","if",
        "so","do","have","has","can","will","i","you","we","they","he","she",
        "what","how","why","when","which","who","me","my","your","its","their",
    }
    q_words = set(re.findall(r'\w+', query.lower())) - STOPWORDS
    scored = []
    for fname, chunk in all_chunks:
        c_words = set(re.findall(r'\w+', chunk.lower()))
        s = len(c_words & q_words) / max(len(q_words), 1)
        scored.append((s, fname, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


# =================================================================
# PDF EXTRACTOR
# =================================================================
def extract_text_from_pdf(filepath):
    try:
        import fitz
    except ImportError:
        print("[PDF] Installing PyMuPDF …")
        os.system("pip install pymupdf -q")
        import fitz
    try:
        doc = fitz.open(filepath)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text}")
        doc.close()
        full_text = "\n\n".join(pages)
        print(f"[PDF] Extracted {len(pages)} pages, {len(full_text):,} chars.")
        return full_text
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return ""


# =================================================================
# FILE LOADER — path parsing fixed for spaces in folder names
# =================================================================
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80


def _split_paths(path_string):
    """
    Correctly split a string of file paths that may be:
    - space-separated plain paths: /a/b.txt /c/d.py
    - quoted paths with spaces:    "/a/my docs/b.txt" /c/d.py
    Returns a list of clean path strings.
    """
    import shlex
    try:
        return shlex.split(path_string)
    except ValueError:
        # shlex failed (e.g. unmatched quote) — fall back to naive split
        return path_string.strip().split()


def load_single_file(filepath, mode_cfg):
    filepath = filepath.strip().strip("\"' ")
    ext = os.path.splitext(filepath)[1].lower()
    SUPPORTED = {".txt", ".py", ".md", ".json", ".csv", ".pdf"}

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return [], ""
    if ext not in SUPPORTED:
        print(f"[ERROR] Unsupported: {ext}. Supported: {', '.join(SUPPORTED)}")
        return [], ""

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(filepath)
    else:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"[RAG ERROR] {e}")
            return [], ""

    if not raw_text.strip():
        print("[RAG WARNING] File is empty.")
        return [], ""

    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    raw_text = re.sub(r' {4,}', ' ', raw_text)

    chunks, i = [], 0
    while i < len(raw_text):
        chunks.append(raw_text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP

    fname = os.path.basename(filepath)
    print(f"\n[RAG READY] {fname}")
    print(f"  Size   : {len(raw_text):,} chars  |  Chunks: {len(chunks)}")
    print(f"  Inject : top {mode_cfg['TOP_K_CHUNKS']} chunks, "
          f"max {mode_cfg['MAX_RAG_CHARS']} chars  "
          f"[mode: {mode_cfg['label'].split('|')[0].strip()}]")
    return chunks, fname


def load_file_for_rag(path_string, rag_store, mode_cfg):
    paths = _split_paths(path_string)
    loaded = []
    for p in paths:
        chunks, fname = load_single_file(p, mode_cfg)
        if fname:
            rag_store[fname] = chunks
            loaded.append(fname)
    return loaded


def build_rag_context(query, rag_store, mode_cfg):
    if not rag_store:
        return ""
    top = retrieve_relevant_chunks(query, rag_store, mode_cfg["TOP_K_CHUNKS"])
    if not top:
        return ""
    parts = [f"[{fname}  score={score:.3f}]\n{chunk}"
             for score, fname, chunk in top]
    context = "\n---\n".join(parts)
    if len(context) > mode_cfg["MAX_RAG_CHARS"]:
        context = context[:mode_cfg["MAX_RAG_CHARS"]] + "\n[…truncated…]"
    return f"\n\n[RELEVANT CONTENT FROM FILE(S)]:\n{context}\n[END OF FILE CONTEXT]\n"


# =================================================================
# CONTEXT BUDGET + VISUAL BAR
# =================================================================
def estimate_tokens(text):
    return max(1, len(text) // 4)


def context_bar(pct, width=20):
    """Render: [████████░░░░] 78%"""
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    color  = "\033[92m"   # green
    if pct >= 85:
        color = "\033[91m"  # red
    elif pct >= 60:
        color = "\033[93m"  # yellow
    return f"{color}[{bar}] {pct:.1f}%\033[0m"


def check_context_budget(system_msg, chat_history, rag_context, mode_cfg,
                         actual_reply_tokens=None, warn=True):
    # NOTE: chat_history[0] is always the system message, so we exclude it
    # here and add system_msg separately below — otherwise the system
    # prompt gets counted twice and the bar never reflects reality
    # (e.g. it never drops back down after Clear Chat).
    history_text  = " ".join(m["content"] for m in chat_history if m["role"] != "system")
    input_tokens  = (estimate_tokens(system_msg)
                     + estimate_tokens(history_text)
                     + estimate_tokens(rag_context))
    output_tokens = 0 if actual_reply_tokens is not None else mode_cfg["max_tokens"]
    total         = input_tokens + output_tokens
    pct           = min((total / mode_cfg["n_ctx"]) * 100, 100.0)
    if warn and pct > 85:
        print(f"[!] CONTEXT {context_bar(pct)}  — type 'clear-chat' to free space")
    return total, pct


# =================================================================
# MEMORY MANAGEMENT
# =================================================================
def enforce_memory_limit(history, max_messages):
    """Drop oldest user+assistant pairs (never drop index 0 = system msg)."""
    while len(history) > max_messages:
        history.pop(1)
        if len(history) > 1:
            history.pop(1)
    return history


# =================================================================
# CHAT EXPORT
# =================================================================
def export_chat(chat_history, model_name, mode_name):
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sivan_chat_{ts}.md"
    lines    = [
        "# Sivan Chat Export\n\n",
        f"**Model:** {model_name}  \n",
        f"**Mode:** {mode_name}  \n",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n---\n\n",
    ]
    for msg in chat_history:
        role = msg["role"].capitalize()
        if role == "System":
            continue
        lines.append(f"### {role}\n{msg['content']}\n\n")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[SAVED] Chat exported → {filename}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")


# =================================================================
# THINK MODE
# =================================================================
THINK_SUFFIX = (
    " Before answering, briefly reason through the problem inside "
    "<think>...</think> tags — keep this reasoning proportional to the "
    "problem's actual difficulty (a few words for simple messages, more "
    "for hard ones). Then give your final answer after the closing "
    "</think> tag, and nowhere else."
)


def strip_think_block(text):
    """
    Remove <think>...</think> reasoning from model output, returning only
    the final answer. If the closing tag is missing (model cut off mid
    reasoning) or no think block exists at all, returns the text unchanged
    minus any dangling opening tag, so callers never show raw scratch work.
    """
    if "<think>" not in text:
        return text.strip()
    match = re.search(r"<think>.*?</think>", text, flags=re.DOTALL)
    if match:
        return text[match.end():].strip()
    # Opening tag with no closing tag yet — strip from the tag onward,
    # nothing usable to show as a "final answer" in this chunk.
    return text.split("<think>", 1)[0].strip()


# =================================================================
# TERMINAL MAIN LOOP
# =================================================================
def run_sivan(llm, model_info, mode_name):
    mode_cfg       = MODES[mode_name]
    prompt_format  = model_info.get("prompt_format", "llama3")
    rag_store      = {}
    think_mode     = False

    base_system = (
        "You are Sivan, a highly intelligent AI software engineer. "
        "Match your reply's length and depth to the user's message: "
        "brief, casual messages get short, natural replies; substantive "
        "technical questions get complete, thorough answers. Never stop "
        "mid-sentence. When file content is provided in the prompt, use it "
        "to answer."
    )
    system_msg   = base_system
    chat_history = [{"role": "system", "content": system_msg}]
    model_name   = model_info["name"]

    def _print_help():
        print(f"\n[SIVAN ONLINE]  Model  : {model_name}")
        print(f"                Format : {prompt_format}")
        print(f"                Mode   : {mode_cfg['label']}")
        print("─" * 58)
        print("  exit              quit Sivan")
        print("  load <files>      load files for RAG (space/quote-separated)")
        print("  clear-file        unload all RAG files")
        print("  clear-chat        reset conversation history")
        print("  save-chat         export conversation to .md file")
        print("  think-on/off      toggle chain-of-thought mode")
        print("  mode <n>          switch mode: lite / balanced / thinking")
        print("  switch-model      hot-swap model (keeps history)")
        print("  info              show context bar, files, stats")
        print("  help              show this menu")
        print("─" * 58)
        print("  Files: .txt .py .md .json .csv .pdf")
        print("  Multi-load: load script.py notes.md report.pdf\n")

    _print_help()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Shutting down Sivan…]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "exit":
            print("[Shutting down Sivan…]")
            break

        if cmd == "help":
            _print_help()
            continue

        if cmd.startswith("load "):
            loaded = load_file_for_rag(user_input[5:], rag_store, mode_cfg)
            if loaded:
                print(f"[RAG ACTIVE] Loaded: {', '.join(loaded)}")
                print(f"[RAG STORE ] {len(rag_store)} file(s) in memory")
            continue

        if cmd == "clear-file":
            rag_store.clear()
            print("[RAG CLEARED] All files unloaded.")
            continue

        if cmd == "clear-chat":
            chat_history = [{"role": "system", "content": system_msg}]
            print("[Chat cleared.]")
            continue

        if cmd == "save-chat":
            export_chat(chat_history, model_name, mode_name)
            continue

        if cmd == "think-on":
            think_mode = True
            system_msg = base_system + THINK_SUFFIX
            chat_history[0]["content"] = system_msg
            print("[THINK MODE ON] Step-by-step reasoning active.")
            continue

        if cmd == "think-off":
            think_mode = False
            system_msg = base_system
            chat_history[0]["content"] = system_msg
            print("[THINK MODE OFF]")
            continue

        if cmd.startswith("mode"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1] in MODES:
                new_mode = parts[1]
                if new_mode == mode_name:
                    print(f"[MODE] Already in {mode_name}.")
                else:
                    old_ctx  = mode_cfg["n_ctx"]
                    mode_name = new_mode
                    mode_cfg  = MODES[mode_name]
                    print(f"[MODE → {mode_name.upper()}] {mode_cfg['label']}")
                    if mode_cfg["n_ctx"] != old_ctx:
                        print(f"[!] n_ctx changed {old_ctx}→{mode_cfg['n_ctx']}. Reload model? (y/n): ", end="")
                        if input().strip().lower() == "y":
                            llm = load_model(getattr(llm, "model_path", ""), mode_cfg)
                            print("[Model reloaded.]")
                        else:
                            print("[Model NOT reloaded — n_ctx stays until restart.]")
            else:
                print("[!] Usage: mode lite  /  mode balanced  /  mode thinking")
            continue

        if cmd == "switch-model":
            print("[MODEL SWITCHER] Chat history preserved.")
            model_info    = show_catalogue_and_pick()
            new_path      = download_model(model_info)
            llm           = load_model(new_path, mode_cfg)
            model_name    = model_info["name"]
            prompt_format = model_info.get("prompt_format", "llama3")
            print(f"[SWITCHED] Now running: {model_name}  [{prompt_format}]")
            continue

        if cmd == "info":
            history_text = " ".join(m["content"] for m in chat_history)
            sys_tok      = estimate_tokens(system_msg)
            hist_tok     = estimate_tokens(history_text)
            total_tok    = sys_tok + hist_tok
            pct          = min((total_tok / mode_cfg["n_ctx"]) * 100, 100.0)
            scorer       = "sklearn TF-IDF" if _SKLEARN_OK else "word-overlap (fallback)"
            gpu_val      = _env("N_GPU_LAYERS", "0")
            print(f"\n{'═'*52}")
            print(f"  SIVAN INFO")
            print(f"{'─'*52}")
            print(f"  Model         : {model_name}")
            print(f"  Prompt format : {prompt_format}")
            print(f"  Mode          : {mode_name.upper()}  —  {mode_cfg['label']}")
            print(f"  GPU layers    : {gpu_val}")
            print(f"  RAG scorer    : {scorer}")
            print(f"{'─'*52}")
            print(f"  Context used : {context_bar(pct)}  ({total_tok} / {mode_cfg['n_ctx']} tok)")
            print(f"  Remaining    : ~{mode_cfg['n_ctx'] - total_tok} tokens free")
            print(f"  Breakdown:")
            print(f"    System prompt  ~{sys_tok:>5} tok")
            print(f"    Chat history   ~{hist_tok:>5} tok  ({len(chat_history)-1} turns)")
            print(f"  Limits (next reply):")
            print(f"    Output ceiling  {mode_cfg['max_tokens']:>5} tok")
            print(f"    RAG ceiling    ~{mode_cfg['MAX_RAG_CHARS']//4:>5} tok  ({mode_cfg['MAX_RAG_CHARS']} chars, top-{mode_cfg['TOP_K_CHUNKS']} chunks)")
            print(f"    History cap     {mode_cfg['max_history']:>5} turns max")
            print(f"{'─'*52}")
            print(f"  Think mode : {'ON  ← chain-of-thought active' if think_mode else 'OFF'}")
            if rag_store:
                print(f"  RAG files  : {len(rag_store)} loaded")
                for fname, chunks in rag_store.items():
                    print(f"    • {fname}  ({len(chunks)} chunks)")
            else:
                print(f"  RAG files  : none")
            print(f"{'═'*52}\n")
            continue

        # ── NORMAL QUERY ──────────────────────────────────────────
        rag_context = build_rag_context(user_input, rag_store, mode_cfg) if rag_store else ""
        if rag_context:
            scorer_tag = "sklearn" if _SKLEARN_OK else "fallback"
            print(f"[RAG/{scorer_tag}] Injecting {len(rag_context)} chars from: {', '.join(rag_store.keys())}")

        check_context_budget(system_msg, chat_history, rag_context, mode_cfg)

        chat_history.append({"role": "user", "content": user_input + rag_context})
        chat_history = enforce_memory_limit(chat_history, mode_cfg["max_history"])
        prompt       = compile_prompt(chat_history, prompt_format)

        mode_tag = "[THINK] " if think_mode else ""
        print(f"Sivan: {mode_tag}", end="", flush=True)
        try:
            start  = time.time()
            stream = llm(
                prompt,
                max_tokens=mode_cfg["max_tokens"],
                stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>",
                      "<end_of_turn>", "</s>"],
                stream=True,
                temperature=0.3,
                repeat_penalty=1.1,
            )
            full_response = ""
            token_count   = 0
            for chunk in stream:
                token          = chunk["choices"][0]["text"]
                full_response += token
                token_count   += 1
                sys.stdout.write(token)
                sys.stdout.flush()

            elapsed   = time.time() - start
            tok_per_s = token_count / elapsed if elapsed > 0 else 0

            if full_response.strip():
                chat_history.append({"role": "assistant", "content": full_response})
            else:
                print("[WARNING] Empty response.")
                chat_history.pop()
                continue

            _, pct_used = check_context_budget(
                system_msg, chat_history, rag_context, mode_cfg,
                actual_reply_tokens=token_count, warn=True,
            )
            print(f"\n[{elapsed:.1f}s | {token_count} tok | {tok_per_s:.1f} tok/s | ctx {context_bar(pct_used)}]")
            print("─" * 58)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history.pop()


# =================================================================
# BOOT SEQUENCE
# =================================================================
if __name__ == "__main__":
    mode_name  = pick_mode()
    mode_cfg   = MODES[mode_name]
    model_info = show_catalogue_and_pick()
    model_path = download_model(model_info)
    llm        = load_model(model_path, mode_cfg)
    run_sivan(llm, model_info, mode_name)