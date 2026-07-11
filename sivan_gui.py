import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import os

# Import your engine
import engine_core

# Try to load existing .env values to prepopulate settings
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up the modern AI theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SivanDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("Sivan AI - Engine Core v6")
        self.geometry("1200x800")
        self.minsize(900, 600)

        # Grid layout: 1 row, 2 columns (Sidebar + Main Chat)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIVAN ENGINE STATE ---
        self.llm = None
        self.model_info = None
        self.rag_store = {}
        
        # Pull default mode from environment or fallback
        self.mode_name = os.environ.get("DEFAULT_MODE", "balanced")
        if self.mode_name not in engine_core.MODES:
            self.mode_name = "balanced"
        self.mode_cfg = engine_core.MODES[self.mode_name]
        
        self.base_system = (
            "You are Sivan, a highly intelligent AI software engineer. "
            "Match your reply's length and depth to the user's message: "
            "brief, casual messages get short, natural replies; substantive "
            "technical questions get complete, thorough answers. Never stop "
            "mid-sentence. When file content is provided in the prompt, use "
            "it to answer."
        )
        self.chat_history = [{"role": "system", "content": self.base_system}]

        # <think>-tag stream filter state (reset per generation)
        self._stream_buffer = ""
        self._stream_in_think = False

        # Build UI Elements
        self._build_sidebar()
        self._build_main_panel()

        # Initial Status Update
        self._append_to_chat("System", "Sivan UI Engine is Ready.")
        self._append_to_chat("System", "Configure your .env in the 'Settings' tab, select a model in the 'Engine' tab, and click 'Load Model'.")

    def _build_sidebar(self):
        """Constructs the left sidebar for settings and RAG control using tabs."""
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1) 

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SIVAN AI", font=ctk.CTkFont(size=26, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 10))
        
        self.status_indicator = ctk.CTkLabel(self.sidebar_frame, text="Status: OFFLINE", text_color="gray", font=ctk.CTkFont(size=12))
        self.status_indicator.grid(row=1, column=0, pady=(0, 15))

        # Tab View for organizing sidebar controls
        self.tabs = ctk.CTkTabview(self.sidebar_frame, width=300)
        self.tabs.grid(row=2, column=0, padx=15, pady=(0, 20), sticky="nsew")
        
        self.tab_engine = self.tabs.add("🧠 Engine")
        self.tab_rag = self.tabs.add("📚 Knowledge")
        self.tab_settings = self.tabs.add("⚙️ Settings")

        self._build_engine_tab()
        self._build_rag_tab()
        self._build_settings_tab()

        # Context Budget Progress Bar (Always visible at the bottom)
        self.context_label = ctk.CTkLabel(self.sidebar_frame, text="Context Budget: 0%", anchor="w")
        self.context_label.grid(row=3, column=0, padx=20, pady=(5, 0), sticky="ew")
        self.context_progressbar = ctk.CTkProgressBar(self.sidebar_frame, progress_color="#1f538d")
        self.context_progressbar.grid(row=4, column=0, padx=20, pady=(5, 25), sticky="ew")
        self.context_progressbar.set(0)

    def _build_engine_tab(self):
        """Engine controls inside the first tab."""
        # Model Selection
        ctk.CTkLabel(self.tab_engine, text="Select Model:", anchor="w").pack(fill="x", padx=10, pady=(10, 0))
        
        self.catalog_map = {}
        model_names = []
        for cat in engine_core.MODEL_CATALOGUE:
            for m in engine_core.MODEL_CATALOGUE[cat]:
                model_names.append(m["name"])
                self.catalog_map[m["name"]] = m

        self.model_menu = ctk.CTkOptionMenu(self.tab_engine, values=model_names)
        self.model_menu.pack(fill="x", padx=10, pady=(5, 15))
        
        # Default fallback
        if model_names:
            self.model_menu.set(model_names[0])

        # RAM Mode Selection
        ctk.CTkLabel(self.tab_engine, text="RAM Mode (Context Limit):", anchor="w").pack(fill="x", padx=10, pady=(5, 0))
        self.mode_menu = ctk.CTkOptionMenu(
            self.tab_engine, 
            values=["lite", "balanced", "thinking"],
            command=self.ui_change_mode
        )
        self.mode_menu.pack(fill="x", padx=10, pady=(5, 15))
        self.mode_menu.set(self.mode_name)

        # Think Mode Toggle
        self.think_switch = ctk.CTkSwitch(self.tab_engine, text="Think Mode (CoT)")
        self.think_switch.pack(anchor="w", padx=10, pady=(5, 20))

        # Load Button
        self.load_engine_button = ctk.CTkButton(
            self.tab_engine, 
            text="⚡ Load Selected Model", 
            height=40,
            command=self.ui_initialize_engine,
            font=ctk.CTkFont(weight="bold")
        )
        self.load_engine_button.pack(fill="x", padx=10, pady=10)

    def _build_rag_tab(self):
        """RAG controls inside the second tab."""
        ctk.CTkLabel(self.tab_rag, text="RAG Knowledge Base", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        ctk.CTkLabel(self.tab_rag, text="Inject text, PDF, code, or CSVs.", text_color="gray", font=ctk.CTkFont(size=11)).pack(pady=(0, 15))

        self.file_button = ctk.CTkButton(self.tab_rag, text="📂 Load Files...", command=self.ui_load_files)
        self.file_button.pack(fill="x", padx=20, pady=5)
        
        self.clear_file_button = ctk.CTkButton(self.tab_rag, text="🗑️ Clear Memory", command=self.ui_clear_files, fg_color="transparent", border_width=1)
        self.clear_file_button.pack(fill="x", padx=20, pady=5)

        self.loaded_files_label = ctk.CTkTextbox(self.tab_rag, height=150, state="disabled", fg_color="transparent", border_width=1)
        self.loaded_files_label.pack(fill="x", padx=10, pady=15)
        self._update_file_display()

    def _build_settings_tab(self):
        """Environment configs inside the third tab."""
        ctk.CTkLabel(self.tab_settings, text="Environment (.env) Configuration", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 15))

        # HF Token
        ctk.CTkLabel(self.tab_settings, text="HuggingFace Token (Optional):", anchor="w").pack(fill="x", padx=10)
        self.hf_token_entry = ctk.CTkEntry(self.tab_settings, placeholder_text="hf_...", show="*")
        self.hf_token_entry.pack(fill="x", padx=10, pady=(0, 15))
        self.hf_token_entry.insert(0, os.environ.get("HF_TOKEN", ""))

        # GPU Layers
        ctk.CTkLabel(self.tab_settings, text="GPU Offload Layers (0 = CPU, -1 = All):", anchor="w").pack(fill="x", padx=10)
        self.gpu_layers_entry = ctk.CTkEntry(self.tab_settings, placeholder_text="e.g. 0, 15, -1")
        self.gpu_layers_entry.pack(fill="x", padx=10, pady=(0, 15))
        self.gpu_layers_entry.insert(0, os.environ.get("N_GPU_LAYERS", "0"))

        # Default Mode
        ctk.CTkLabel(self.tab_settings, text="Default Boot Mode:", anchor="w").pack(fill="x", padx=10)
        self.default_mode_menu = ctk.CTkOptionMenu(self.tab_settings, values=["lite", "balanced", "thinking"])
        self.default_mode_menu.pack(fill="x", padx=10, pady=(0, 15))
        self.default_mode_menu.set(os.environ.get("DEFAULT_MODE", "balanced"))

        # Cache Directory
        ctk.CTkLabel(self.tab_settings, text="Model Cache Directory (Optional):", anchor="w").pack(fill="x", padx=10)
        
        cache_frame = ctk.CTkFrame(self.tab_settings, fg_color="transparent")
        cache_frame.pack(fill="x", padx=10, pady=(0, 20))
        cache_frame.grid_columnconfigure(0, weight=1)
        
        self.cache_dir_entry = ctk.CTkEntry(cache_frame, placeholder_text="Default HF Cache")
        self.cache_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cache_dir_entry.insert(0, os.environ.get("MODEL_CACHE_DIR", ""))
        
        ctk.CTkButton(cache_frame, text="Browse", width=60, command=self.ui_browse_cache).grid(row=0, column=1)

        # Save Button
        self.save_env_btn = ctk.CTkButton(self.tab_settings, text="💾 Save Config", fg_color="#28a745", hover_color="#218838", command=self.ui_save_env)
        self.save_env_btn.pack(fill="x", padx=10, pady=5)

    def _build_main_panel(self):
        """Constructs the chat display and input area."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Main Chat Window
        self.chat_display = ctk.CTkTextbox(
            self.main_frame, 
            state="disabled", 
            font=ctk.CTkFont(family="Consolas" if os.name == "nt" else "Monospace", size=14), 
            wrap="word",
            spacing1=5,
            spacing3=5
        )
        self.chat_display.grid(row=0, column=0, padx=(0, 20), pady=20, sticky="nsew")

        # Input Box Area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=(0, 20), pady=(0, 10), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message to Sivan...", height=45, font=ctk.CTkFont(size=14))
        self.chat_input.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="ew")
        self.chat_input.bind("<Return>", self.ui_send_message)

        self.send_button = ctk.CTkButton(self.input_frame, text="Send 🚀", width=120, height=45, font=ctk.CTkFont(weight="bold"), command=self.ui_send_message)
        self.send_button.grid(row=0, column=1, padx=0, pady=0)

        # Lower Utilities
        self.cmd_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.cmd_frame.grid(row=2, column=0, padx=(0, 20), pady=(0, 20), sticky="ew")
        
        self.clear_chat_btn = ctk.CTkButton(self.cmd_frame, text="🧹 Clear Chat", command=self.ui_clear_chat, width=120, fg_color="transparent", border_width=1)
        self.clear_chat_btn.pack(side="left", padx=(0, 10))
        
        self.save_chat_btn = ctk.CTkButton(self.cmd_frame, text="📥 Export Chat", command=self.ui_export_chat, width=120, fg_color="transparent", border_width=1)
        self.save_chat_btn.pack(side="left")

    # --- SETTINGS / ENV LOGIC ---

    def ui_browse_cache(self):
        dir_path = filedialog.askdirectory(title="Select Model Cache Directory")
        if dir_path:
            self.cache_dir_entry.delete(0, tk.END)
            self.cache_dir_entry.insert(0, dir_path)

    def ui_save_env(self):
        """Saves current UI configurations directly to the .env file and updates OS environ."""
        token = self.hf_token_entry.get().strip()
        gpu = self.gpu_layers_entry.get().strip()
        mode = self.default_mode_menu.get().strip()
        cache = self.cache_dir_entry.get().strip()

        # Write to .env
        env_content = (
            "# SIVAN AI — Environment Configuration\n"
            f"HF_TOKEN={token}\n"
            f"N_GPU_LAYERS={gpu}\n"
            f"DEFAULT_MODE={mode}\n"
            f"MODEL_CACHE_DIR={cache}\n"
        )
        
        try:
            with open(".env", "w") as f:
                f.write(env_content)
                
            # Apply immediately to runtime environment
            os.environ["HF_TOKEN"] = token
            os.environ["N_GPU_LAYERS"] = gpu
            os.environ["DEFAULT_MODE"] = mode
            os.environ["MODEL_CACHE_DIR"] = cache
            
            self._append_to_chat("System", "✅ Configurations saved to .env and applied to runtime successfully.")
            self.save_env_btn.configure(text="✅ Saved!", fg_color="#218838")
            self.after(2000, lambda: self.save_env_btn.configure(text="💾 Save Config", fg_color="#28a745"))
            
            # Sync mode selection if changed
            self.mode_menu.set(mode)
            self.ui_change_mode(mode)
            
        except Exception as e:
            self._append_to_chat("System", f"❌ Failed to save .env file: {e}")

    # --- THREADED ENGINE INITIALIZATION ---

    def ui_initialize_engine(self):
        """Spawns a background thread to prevent UI freezing during download/load."""
        selected_name = self.model_menu.get()
        self.model_info = self.catalog_map[selected_name]

        self._append_to_chat("System", f"Starting load sequence for: {self.model_info['name']}")
        if not os.environ.get("HF_TOKEN"):
            self._append_to_chat("System", "[Note] No HuggingFace token provided. Attempting to download from public repositories.")
            
        self.status_indicator.configure(text="Status: BOOTING...", text_color="#f39c12")
        self.load_engine_button.configure(state="disabled", text="🔄 Booting Core...")
        
        # Execute the blocking process inside an isolated worker thread
        download_worker = threading.Thread(target=self._bg_load_sequence, daemon=True)
        download_worker.start()

    def _bg_load_sequence(self):
        """Worker function processing the actual core compilation off the main thread."""
        try:
            model_path = engine_core.download_model(self.model_info)
            self.llm = engine_core.load_model(model_path, self.mode_cfg)
            
            # Safe UI updates
            self.after(0, lambda: self._append_to_chat("System", f"✅ Engine core activated natively using {self.mode_name.upper()} logic limiters."))
            self.after(0, lambda: self.load_engine_button.configure(state="normal", text="✅ Reload Model"))
            self.after(0, lambda: self.status_indicator.configure(text="Status: ACTIVE", text_color="#2ecc71"))
        except Exception as e:
            self.after(0, lambda: self._append_to_chat("System", f"[CRITICAL ERROR] Failed during setup: {e}"))
            self.after(0, lambda: self.load_engine_button.configure(state="normal", text="❌ Core Error - Retry"))
            self.after(0, lambda: self.status_indicator.configure(text="Status: ERROR", text_color="#e74c3c"))

    def ui_change_mode(self, selected_mode):
        self.mode_name = selected_mode
        self.mode_cfg = engine_core.MODES[self.mode_name]
        self._append_to_chat("System", f"Context metrics modified to: {self.mode_name.upper()}. (Requires engine reload if running)")
        self._update_context_bar()

    # --- THREADED INFERENCE PIPELINE ---

    def ui_send_message(self, event=None):
        if self.llm is None:
            self._append_to_chat("System", "⚠️ Core engine offline. Select a model and click 'Load Selected Model' first.")
            return

        user_text = self.chat_input.get()
        if not user_text.strip():
            return

        self._append_to_chat("You", user_text)
        self.chat_input.delete(0, tk.END)
        
        # Freeze entry controls during inference
        self.chat_input.configure(state="disabled")
        self.send_button.configure(state="disabled", text="Thinking...")
        self.status_indicator.configure(text="Status: GENERATING", text_color="#3498db")

        # RAG Pipeline
        rag_context = ""
        if self.rag_store:
            rag_context = engine_core.build_rag_context(user_text, self.rag_store, self.mode_cfg)
        
        # System Message formatting & Think mode
        if self.think_switch.get():
            active_system = self.base_system + getattr(engine_core, 'THINK_SUFFIX', " Reason step-by-step.")
        else:
            active_system = self.base_system

        self.chat_history[0]["content"] = active_system
        self.chat_history.append({"role": "user", "content": user_text + rag_context})
        self.chat_history = engine_core.enforce_memory_limit(self.chat_history, self.mode_cfg["max_history"])

        # Reset the incremental <think> filter for this generation
        self._stream_buffer = ""
        self._stream_in_think = False
        
        # Use appropriate prompt format based on the selected model architecture
        fmt = self.model_info.get("prompt_format", "llama3")
        prompt = engine_core.compile_prompt(self.chat_history, prompt_format=fmt)

        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, "\nSivan:\n", "agent_name")
        self.chat_display.configure(state="disabled")

        # Offload text generation loop to a background thread
        generation_worker = threading.Thread(target=self._bg_generation_sequence, args=(prompt,), daemon=True)
        generation_worker.start()

    def _bg_generation_sequence(self, prompt):
        """Handles streaming loops safely without locking user window elements."""
        full_response = ""
        try:
            stream = self.llm(
                prompt,
                max_tokens=self.mode_cfg["max_tokens"],
                stop=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<end_of_turn>", "</s>"],
                stream=True,
                temperature=0.3,
                repeat_penalty=1.1
            )
            
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                full_response += token
                visible = self._filter_think_stream(token)
                if visible:
                    self.after(0, lambda t=visible: self._thread_safe_token_insert(t))

            # Flush any trailing text we were holding back only to guard
            # against a <think> tag starting right at the stream's end.
            if not self._stream_in_think and self._stream_buffer:
                leftover = self._stream_buffer
                self._stream_buffer = ""
                self.after(0, lambda t=leftover: self._thread_safe_token_insert(t))
                
        except Exception as e:
            self.after(0, lambda: self._append_to_chat("System", f"\n[ERROR] Generation halted: {e}"))
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()

        self.after(0, lambda: self._finalize_ui_generation(full_response))

    def _filter_think_stream(self, token):
        """
        Stateful, incremental filter that suppresses everything between
        <think> and </think> as tokens arrive, so reasoning never reaches
        the visible chat window — even when a tag is split across two
        streamed chunks. Returns only the text that should be displayed
        right now (may be empty).
        """
        self._stream_buffer += token
        visible_out = ""

        while True:
            if not self._stream_in_think:
                idx = self._stream_buffer.find("<think>")
                if idx != -1:
                    visible_out += self._stream_buffer[:idx]
                    self._stream_buffer = self._stream_buffer[idx + len("<think>"):]
                    self._stream_in_think = True
                    continue
                # No opening tag yet — hold back a small tail in case a
                # tag is mid-way through arriving across chunks.
                hold = len("<think>") - 1
                if len(self._stream_buffer) > hold:
                    visible_out += self._stream_buffer[:-hold]
                    self._stream_buffer = self._stream_buffer[-hold:]
                break
            else:
                idx = self._stream_buffer.find("</think>")
                if idx != -1:
                    self._stream_buffer = self._stream_buffer[idx + len("</think>"):]
                    self._stream_in_think = False
                    continue
                # Still inside reasoning — nothing to show yet.
                break

        return visible_out

    def _thread_safe_token_insert(self, token):
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, token)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state="disabled")

    def _finalize_ui_generation(self, full_response):
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.configure(state="disabled")

        if full_response.strip():
            clean_response = engine_core.strip_think_block(full_response)
            self.chat_history.append({"role": "assistant", "content": clean_response})
            self._update_context_bar()

        # Re-enable inputs smoothly
        self.chat_input.configure(state="normal")
        self.send_button.configure(state="normal", text="Send 🚀")
        self.status_indicator.configure(text="Status: ACTIVE", text_color="#2ecc71")
        self.chat_input.focus()

    # --- ADMINISTRATIVE OPERATIONS ---

    def ui_load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Files for RAG",
            filetypes=[("Supported Files", "*.txt *.py *.md *.json *.csv *.pdf")]
        )
        if not file_paths:
            return

        self.file_button.configure(state="disabled", text="⏳ Loading...")
        worker = threading.Thread(target=self._bg_load_files, args=(file_paths,), daemon=True)
        worker.start()

    def _bg_load_files(self, file_paths):
        """Runs file/PDF parsing + TF-IDF prep off the UI thread, and makes
        sure errors are shown in-app instead of vanishing into a console
        window that doesn't exist in the packaged .exe."""
        loaded, failed = [], []
        for p in file_paths:
            try:
                chunks, fname = engine_core.load_single_file(p, self.mode_cfg)
                if fname:
                    self.rag_store[fname] = chunks
                    loaded.append(fname)
                else:
                    failed.append(os.path.basename(p))
            except Exception as e:
                failed.append(f"{os.path.basename(p)} ({e})")

        def _finish():
            if loaded:
                self._append_to_chat("System", f"[RAG] Linked contextual matrices: {', '.join(loaded)}\n")
            if failed:
                self._append_to_chat("System", f"⚠️ Failed to load: {', '.join(failed)}\n")
            self._update_file_display()
            self.file_button.configure(state="normal", text="📂 Load Files...")

        self.after(0, _finish)

    def ui_clear_files(self):
        self.rag_store.clear()
        self._append_to_chat("System", "[RAG] Knowledge memory purged.\n")
        self._update_file_display()

    def _update_file_display(self):
        self.loaded_files_label.configure(state="normal")
        self.loaded_files_label.delete("1.0", tk.END)
        if not self.rag_store:
            self.loaded_files_label.insert(tk.END, "No files currently loaded.")
        else:
            self.loaded_files_label.insert(tk.END, "Currently Loaded:\n\n")
            for filename in self.rag_store.keys():
                self.loaded_files_label.insert(tk.END, f"• {filename}\n")
        self.loaded_files_label.configure(state="disabled")

    def ui_clear_chat(self):
        self.chat_history = [{"role": "system", "content": self.base_system}]
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state="disabled")
        self._append_to_chat("System", "Chat logs reset.")
        self._update_context_bar()

    def ui_export_chat(self):
        m_name = self.model_info["name"] if self.model_info else "unloaded"
        engine_core.export_chat(self.chat_history, m_name, self.mode_name)
        self._append_to_chat("System", "Chat exported successfully to root workspace.")

    def _update_context_bar(self):
        # chat_history[0] is always the system message — exclude it here
        # since we add the active system prompt separately below. Counting
        # it twice was why the bar never dropped back to ~0% after Clear Chat.
        history_text = " ".join(m["content"] for m in self.chat_history if m["role"] != "system")
        active_system = self.chat_history[0]["content"] if self.chat_history else self.base_system
        total_tok = engine_core.estimate_tokens(active_system) + engine_core.estimate_tokens(history_text)
        pct = min((total_tok / self.mode_cfg["n_ctx"]) * 100, 100.0)
        
        self.context_progressbar.set(pct / 100)
        
        if pct > 85:
            self.context_progressbar.configure(progress_color="#e74c3c")  # Red
        elif pct > 60:
            self.context_progressbar.configure(progress_color="#f39c12")  # Yellow
        else:
            self.context_progressbar.configure(progress_color="#1f538d")  # Blue standard
            
        self.context_label.configure(text=f"Context Budget: {pct:.1f}% ({total_tok} / {self.mode_cfg['n_ctx']})")

    def _append_to_chat(self, sender, text):
        self.chat_display.configure(state="normal")
        
        # Visually differentiate roles
        if sender == "System":
            self.chat_display.insert(tk.END, f"\n⚙️ {sender}:\n{text}\n")
        elif sender == "You":
            self.chat_display.insert(tk.END, f"\n👤 {sender}:\n{text}\n")
        else:
            self.chat_display.insert(tk.END, f"\n{sender}:\n{text}\n")
            
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    app = SivanDesktopApp()
    app.mainloop()