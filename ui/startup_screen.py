"""
Startup and settings screen.

Responsibilities:
- Let the user choose mode + exchange
- Edit config values before boot
"""
import importlib
import json
import os
import re
import time
import sys
import tkinter as tk
from tkinter import ttk


def _load_config(config_path):
    from config import config as cfg
    importlib.reload(cfg)
    return cfg


def _format_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)

def _mask_key(value):
    if not value:
        return "Not set"
    return "*" * 8


def _parse_symbols(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts

def _load_env_vars(env_path):
    if not os.path.exists(env_path):
        return {}
    data = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            data[key.strip()] = val.strip().strip("\"").strip("'")
    return data

def _update_env_file(env_path, updates):
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    existing = _load_env_vars(env_path)
    merged = dict(existing)
    for key, val in updates.items():
        if val is None or val == "":
            # Keep existing value if user left blank
            continue
        merged[key] = val
    lines = []
    seen = set()
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if "=" in raw and not raw.strip().startswith("#"):
                    k = raw.split("=", 1)[0].strip()
                    if k in merged:
                        lines.append(f"{k}={merged[k]}")
                        seen.add(k)
                    else:
                        lines.append(raw)
                else:
                    lines.append(raw)
    for key, val in merged.items():
        if key not in seen:
            lines.append(f"{key}={val}")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join([ln for ln in lines if ln is not None]) + "\n")

def _normalize_config_path(config_path):
    if os.path.exists(config_path):
        return config_path
    if getattr(sys, "frozen", False):
        temp_markers = ("\\appdata\\local\\temp\\", "/tmp/", "_mei")
        path_lower = config_path.lower()
        if any(marker in path_lower for marker in temp_markers):
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            return os.path.join(base, "TraderBot", "config", "config.py")
    return config_path


def _draw_gradient(canvas, color_top, color_bottom, width, height, steps=120):
    r1, g1, b1 = canvas.winfo_rgb(color_top)
    r2, g2, b2 = canvas.winfo_rgb(color_bottom)
    r_ratio = (r2 - r1) / steps
    g_ratio = (g2 - g1) / steps
    b_ratio = (b2 - b1) / steps
    for i in range(steps):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = f"#{nr >> 8:02x}{ng >> 8:02x}{nb >> 8:02x}"
        y1 = int((height / steps) * i)
        y2 = int((height / steps) * (i + 1))
        canvas.create_rectangle(0, y1, width, y2, outline="", fill=color)


def _show_toast(parent, message, kind="error"):
    toast = tk.Toplevel(parent)
    toast.overrideredirect(True)
    toast.attributes("-topmost", True)
    bg = "#c20000" if kind == "error" else "#008409"
    fg = "#ffffff"
    frame = tk.Frame(toast, bg=bg, padx=12, pady=8)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(frame, text=message, bg=bg, fg=fg, font=("Helvetica", 10, "bold")).pack()
    parent.update_idletasks()
    x = parent.winfo_rootx() + 40
    y = parent.winfo_rooty() + 40
    toast.geometry(f"+{x}+{y}")
    toast.after(2500, toast.destroy)


def _update_config_file(config_path, updates, style_presets_text):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update simple KEY = value lines
    for key, val in updates.items():
        pattern = re.compile(rf"^{key}\s*=\s*.*$", re.MULTILINE)
        if isinstance(val, str):
            new_val = f"{key} = \"{val}\""
        elif isinstance(val, bool):
            new_val = f"{key} = {str(val)}"
        else:
            new_val = f"{key} = {val}"
        if pattern.search(content):
            content = pattern.sub(new_val, content)
        else:
            content += f"\n{new_val}\n"

    # Update STYLE_PRESETS block
    if style_presets_text is not None:
        style_text = style_presets_text.strip()
        if not style_text.startswith("STYLE_PRESETS"):
            style_text = "STYLE_PRESETS = " + style_text
        block = f"# BEGIN STYLE_PRESETS\n{style_text}\n# END STYLE_PRESETS"
        block_pattern = re.compile(
            r"# BEGIN STYLE_PRESETS.*?# END STYLE_PRESETS",
            re.DOTALL
        )
        if block_pattern.search(content):
            content = block_pattern.sub(block, content)
        else:
            content += "\n" + block + "\n"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_startup_ui(config_path="config.py"):
    config_path = _normalize_config_path(config_path)
    cfg = _load_config(config_path)
    env_path = os.path.join(os.path.dirname(config_path), ".env")
    env_vars = _load_env_vars(env_path)

    root = tk.Tk()
    root.title("Ai Trader")
    root.configure(bg="#000000")
    root.geometry("860x860")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#000000", foreground="#e6e6e6")
    style.configure("Header.TLabel", font=("Helvetica", 22, "bold"))
    style.configure("SubHeader.TLabel", font=("Helvetica", 12))
    style.configure("Section.TLabel", font=("Helvetica", 11, "bold"))
    style.configure("Panel.TLabel", background="#000000", foreground="#e6e6e6")
    style.configure("Panel.Header.TLabel", background="#000000", foreground="#e6e6e6", font=("Helvetica", 22, "bold"))
    style.configure("Panel.SubHeader.TLabel", background="#000000", foreground="#c9d1d9", font=("Helvetica", 12))
    style.configure("Panel.Section.TLabel", background="#000000", foreground="#e6e6e6", font=("Helvetica", 11, "bold"))
    style.configure("TButton", padding=6)
    style.configure(
        "Primary.TButton",
        background="#16a34a",
        foreground="#ffffff",
        bordercolor="#16a34a",
        focusthickness=3,
        focuscolor="#86efac"
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#22c55e")],
        bordercolor=[("active", "#22c55e"), ("focus", "#86efac")],
        relief=[("pressed", "sunken"), ("!pressed", "flat")]
    )
    style.configure("TScrollbar", background="#000000", troughcolor="#000000", bordercolor="#000000", arrowcolor="#e6e6e6", lightcolor="#000000", darkcolor="#000000")

    bg_canvas = tk.Canvas(root, highlightthickness=0, bg="#000000")
    bg_canvas.pack(fill=tk.BOTH, expand=True)
    bg_canvas.bind("<Configure>", lambda e: _draw_gradient(bg_canvas, "#000000", "#000000", e.width, e.height))

    card = tk.Frame(bg_canvas, bg="#000000", padx=24, pady=24)
    card_id = bg_canvas.create_window(40, 40, window=card, anchor="nw")

    def _resize_card(event):
        bg_canvas.itemconfigure(card_id, width=min(event.width - 80, 900))
    bg_canvas.bind("<Configure>", _resize_card)

    ttk.Label(card, text="Ai Trader", style="Panel.Header.TLabel").pack(anchor="w")
    ttk.Label(card, text="Select mode and exchange to get started", style="Panel.SubHeader.TLabel").pack(anchor="w", pady=(4, 16))

    ttk.Label(card, text="Select trader mode", style="Panel.Section.TLabel").pack(anchor="w")
    mode_var = tk.StringVar(value=getattr(cfg, "RUN_MODE", "live"))
    mode_frame = tk.Frame(card, bg="#000000")
    mode_frame.pack(anchor="w", pady=(6, 16))

    def _draw_radio(canvas, selected, hover=False):
        canvas.delete("all")
        canvas.configure(bg="#000000")
        ring = "#86efac" if hover else "#8a8a8a"
        canvas.create_oval(8, 8, 36, 36, outline=ring, width=2)
        if selected:
            canvas.create_oval(14, 14, 30, 30, outline="", fill="#22c55e")

    radio_icons = {}

    def _refresh_radios():
        for val, icon in radio_icons.items():
            hover = bool(icon._hover) if hasattr(icon, "_hover") else False
            _draw_radio(icon, mode_var.get() == val, hover=hover)

    def _make_radio(parent, label, value):
        wrapper = tk.Frame(parent, bg="#000000")
        wrapper.pack(side=tk.LEFT, padx=(0, 16))
        icon = tk.Canvas(wrapper, width=44, height=44, highlightthickness=0, bg="#000000")
        icon.pack(side=tk.LEFT)
        text = tk.Label(wrapper, text=label, bg="#000000", fg="#e6e6e6", font=("Helvetica", 11, "bold"))
        text.pack(side=tk.LEFT, padx=(6, 0))

        def _select(_evt=None):
            mode_var.set(value)
            _refresh_radios()

        def _hover_on(_evt=None):
            icon._hover = True
            _refresh_radios()

        def _hover_off(_evt=None):
            icon._hover = False
            _refresh_radios()

        wrapper.bind("<Button-1>", _select)
        icon.bind("<Button-1>", _select)
        text.bind("<Button-1>", _select)
        wrapper.bind("<Enter>", _hover_on)
        wrapper.bind("<Leave>", _hover_off)
        icon.bind("<Enter>", _hover_on)
        icon.bind("<Leave>", _hover_off)
        text.bind("<Enter>", _hover_on)
        text.bind("<Leave>", _hover_off)
        radio_icons[value] = icon
        return icon

    _make_radio(mode_frame, "Live", "live")
    _make_radio(mode_frame, "Sim", "sim")
    _refresh_radios()

    ttk.Label(card, text="Trading profile", style="Panel.Section.TLabel").pack(anchor="w")
    tk.Label(card, text="Autopilot (LLM/RL managed)", bg="#000000", fg="#e6e6e6", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(6, 16))

    ttk.Label(card, text="Select exchange", style="Panel.Section.TLabel").pack(anchor="w")
    exchange_var = tk.StringVar(value=getattr(cfg, "EXCHANGE", "binance"))
    exchange_box = ttk.Combobox(card, textvariable=exchange_var, values=["binance", "kraken"], state="readonly", font=("Helvetica", 11))
    exchange_box.pack(anchor="w", pady=(6, 16))

    def _keys_status_label():
        required = ["OPENAI_API_KEY"]
        exchange = exchange_var.get()
        if exchange == "kraken":
            required += ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"]
        else:
            required += ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
        missing = [k for k in required if not env_vars.get(k)]
        if missing:
            return "Keys: Not set"
        return "Keys: Set"

    keys_status = tk.Label(card, text=_keys_status_label(), bg="#000000", fg="#c9d1d9", font=("Helvetica", 10, "bold"))
    keys_status.pack(anchor="w", pady=(0, 12))

    def _refresh_keys_status(*_):
        keys_status.config(text=_keys_status_label())

    exchange_box.bind("<<ComboboxSelected>>", _refresh_keys_status)

    preview_frame = tk.Frame(card, bg="#000000")
    preview_frame.pack(anchor="w", pady=(4, 16), fill=tk.X)
    preview_title = ttk.Label(preview_frame, text="Autopilot baseline", style="Panel.Section.TLabel")
    preview_title.pack(anchor="w")
    preview_body = tk.Label(preview_frame, text="", bg="#000000", fg="#c9d1d9", justify="left", font=("Helvetica", 10))
    preview_body.pack(anchor="w", pady=(4, 0))
    presets = getattr(cfg, "STYLE_PRESETS", {})
    data = presets.get("autopilot", {})
    lines = [
        f"Min confidence: {data.get('min_confidence_to_order', '-')}",
        f"Size fraction: {data.get('size_fraction', '-')}",
        f"LLM interval: {data.get('llm_check_interval', '-')}",
        f"Stop loss: {data.get('stop_loss_pct_default', '-')}",
        f"Take profit: {data.get('take_profit_pct_default', '-')}",
        f"Trailing stop: {data.get('trailing_stop_pct_default', '-')}",
        f"Cooldown (s): {data.get('cooldown_seconds', '-')}",
        f"Max trades/day: {data.get('max_trades_per_day', '-')}",
        f"Daily loss limit: {data.get('daily_loss_limit_pct', '-')}",
        f"Max drawdown: {data.get('max_drawdown_pct', '-')}",
        f"Max total exposure: {data.get('max_total_exposure_pct', '-')}",
        f"Max symbol exposure: {data.get('max_symbol_exposure_pct', '-')}",
        f"Max open positions: {data.get('max_open_positions', '-')}"
    ]
    preview_body.config(text="\n".join(lines))

    btn_frame = tk.Frame(card, bg="#000000")
    btn_frame.pack(anchor="w", pady=(10, 0))

    def open_settings():
        settings = tk.Toplevel(root)
        settings.title("Settings")
        settings.configure(bg="#000000")
        settings.geometry("720x600")

        canvas = tk.Canvas(settings, bg="#000000", highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#000000", padx=24, pady=24)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        fields = {}
        doc = getattr(cfg, "CONFIG_DOC", {})

        def _refresh_key_labels_local():
            key_labels["openai"].config(text=f"OpenAI API Key: {_mask_key(env_vars.get('OPENAI_API_KEY', ''))}")
            key_labels["kraken_key"].config(text=f"Kraken API Key: {_mask_key(env_vars.get('KRAKEN_API_KEY', ''))}")
            key_labels["kraken_secret"].config(text=f"Kraken API Secret: {_mask_key(env_vars.get('KRAKEN_API_SECRET', ''))}")
            key_labels["binance_key"].config(text=f"Binance API Key: {_mask_key(env_vars.get('BINANCE_API_KEY', ''))}")
            key_labels["binance_secret"].config(text=f"Binance API Secret: {_mask_key(env_vars.get('BINANCE_API_SECRET', ''))}")

        def open_key_editor():
            editor = tk.Toplevel(settings)
            editor.title("Edit API Keys")
            editor.configure(bg="#000000")
            editor.geometry("520x420")

            def add_key_field(label, value):
                ttk.Label(editor, text=label, style="Section.TLabel").pack(anchor="w", padx=20, pady=(12, 4))
                entry = tk.Entry(editor, width=52, bg="#1a1a1a", fg="#e6e6e6", insertbackground="#e6e6e6", show="*")
                if value:
                    entry.insert(0, value)
                entry.pack(anchor="w", padx=20)
                return entry

            e_openai = add_key_field("OpenAI API Key", env_vars.get("OPENAI_API_KEY", ""))
            e_kraken_key = add_key_field("Kraken API Key", env_vars.get("KRAKEN_API_KEY", ""))
            e_kraken_secret = add_key_field("Kraken API Secret", env_vars.get("KRAKEN_API_SECRET", ""))
            e_binance_key = add_key_field("Binance API Key", env_vars.get("BINANCE_API_KEY", ""))
            e_binance_secret = add_key_field("Binance API Secret", env_vars.get("BINANCE_API_SECRET", ""))

            btn_row = tk.Frame(editor, bg="#000000")
            btn_row.pack(anchor="w", padx=20, pady=16)

            def save_keys():
                updates = {
                    "OPENAI_API_KEY": e_openai.get().strip(),
                    "KRAKEN_API_KEY": e_kraken_key.get().strip(),
                    "KRAKEN_API_SECRET": e_kraken_secret.get().strip(),
                    "BINANCE_API_KEY": e_binance_key.get().strip(),
                    "BINANCE_API_SECRET": e_binance_secret.get().strip()
                }
                _update_env_file(env_path, updates)
                nonlocal env_vars
                env_vars = _load_env_vars(env_path)
                _refresh_key_labels_local()
                _show_toast(editor, "Keys saved.", kind="success")
                editor.destroy()

            ttk.Button(btn_row, text="Save Keys", style="Primary.TButton", command=save_keys).pack(side=tk.LEFT)
            ttk.Button(btn_row, text="Cancel", command=editor.destroy).pack(side=tk.LEFT, padx=(10, 0))

        def add_field(label, key, value):
            ttk.Label(scroll_frame, text=label, style="Section.TLabel").pack(anchor="w", pady=(12, 2))
            if isinstance(value, bool):
                value = "true" if value else "false"
            entry = tk.Entry(scroll_frame, width=80, bg="#1a1a1a", fg="#e6e6e6", insertbackground="#e6e6e6")
            entry.insert(0, value)
            entry.pack(anchor="w", pady=(0, 6))
            fields[key] = entry
            if key in doc:
                ttk.Label(scroll_frame, text=doc[key], style="SubHeader.TLabel").pack(anchor="w")

        add_field("RUN_MODE", "RUN_MODE", getattr(cfg, "RUN_MODE", "live"))
        add_field("EXCHANGE", "EXCHANGE", getattr(cfg, "EXCHANGE", "binance"))
        add_field("CAPITAL", "CAPITAL", getattr(cfg, "CAPITAL", 10000))
        add_field("USE_LLM", "USE_LLM", getattr(cfg, "USE_LLM", True))
        add_field("MIN_NOTIONAL", "MIN_NOTIONAL", getattr(cfg, "MIN_NOTIONAL", 10.0))
        add_field("KRAKEN_COST_MIN_USD", "KRAKEN_COST_MIN_USD", getattr(cfg, "KRAKEN_COST_MIN_USD", 0.5))
        add_field("ALLOW_MIN_UPSIZE", "ALLOW_MIN_UPSIZE", getattr(cfg, "ALLOW_MIN_UPSIZE", True))
        add_field("RESET_SIM_WALLET_ON_START", "RESET_SIM_WALLET_ON_START", getattr(cfg, "RESET_SIM_WALLET_ON_START", False))
        add_field("ACCOUNT_INFO_REFRESH_SECONDS", "ACCOUNT_INFO_REFRESH_SECONDS", getattr(cfg, "ACCOUNT_INFO_REFRESH_SECONDS", 60))
        add_field("LIVE_TRADES_REFRESH_SECONDS", "LIVE_TRADES_REFRESH_SECONDS", getattr(cfg, "LIVE_TRADES_REFRESH_SECONDS", 15))
        add_field("UI_REFRESH_SECONDS", "UI_REFRESH_SECONDS", getattr(cfg, "UI_REFRESH_SECONDS", 1))
        add_field("SYMBOLS", "SYMBOLS", ", ".join(getattr(cfg, "SYMBOLS", [])))
        add_field("TIMEFRAME", "TIMEFRAME", getattr(cfg, "TIMEFRAME", "5m"))
        add_field("LLM_CHECK_INTERVAL", "LLM_CHECK_INTERVAL", getattr(cfg, "LLM_CHECK_INTERVAL", 300))
        add_field("STALE_PRICE_SECONDS", "STALE_PRICE_SECONDS", getattr(cfg, "STALE_PRICE_SECONDS", 15))
        add_field("STALE_WARN_INTERVAL_SECONDS", "STALE_WARN_INTERVAL_SECONDS", getattr(cfg, "STALE_WARN_INTERVAL_SECONDS", 60))
        add_field("STALE_GRACE_SECONDS", "STALE_GRACE_SECONDS", getattr(cfg, "STALE_GRACE_SECONDS", 20))
        add_field("BLOCK_ON_STALE_PRICE", "BLOCK_ON_STALE_PRICE", getattr(cfg, "BLOCK_ON_STALE_PRICE", True))

        ttk.Label(scroll_frame, text="API Keys", style="Section.TLabel").pack(anchor="w", pady=(16, 6))
        key_frame = tk.Frame(scroll_frame, bg="#000000")
        key_frame.pack(anchor="w", pady=(0, 8), fill=tk.X)
        key_labels = {}
        key_labels["openai"] = tk.Label(key_frame, text="", bg="#000000", fg="#e6e6e6", font=("Helvetica", 10))
        key_labels["openai"].pack(anchor="w")
        key_labels["kraken_key"] = tk.Label(key_frame, text="", bg="#000000", fg="#e6e6e6", font=("Helvetica", 10))
        key_labels["kraken_key"].pack(anchor="w")
        key_labels["kraken_secret"] = tk.Label(key_frame, text="", bg="#000000", fg="#e6e6e6", font=("Helvetica", 10))
        key_labels["kraken_secret"].pack(anchor="w")
        key_labels["binance_key"] = tk.Label(key_frame, text="", bg="#000000", fg="#e6e6e6", font=("Helvetica", 10))
        key_labels["binance_key"].pack(anchor="w")
        key_labels["binance_secret"] = tk.Label(key_frame, text="", bg="#000000", fg="#e6e6e6", font=("Helvetica", 10))
        key_labels["binance_secret"].pack(anchor="w")
        _refresh_key_labels_local()

        edit_keys_btn = tk.Button(
            key_frame,
            text="Edit Keys",
            command=open_key_editor,
            bg="#111111",
            fg="#ffffff",
            activebackground="#262626",
            activeforeground="#ffffff",
            highlightthickness=2,
            highlightbackground="#22c55e",
            highlightcolor="#22c55e",
            font=("Helvetica", 10, "bold"),
            cursor="hand2"
        )
        edit_keys_btn.pack(anchor="w", pady=(8, 0))

        def save_settings():
            updates = {
                "RUN_MODE": fields["RUN_MODE"].get(),
                "EXCHANGE": fields["EXCHANGE"].get(),
                "TRADING_STYLE": "autopilot",
                "CAPITAL": fields["CAPITAL"].get(),
                "USE_LLM": fields["USE_LLM"].get(),
                "MIN_NOTIONAL": fields["MIN_NOTIONAL"].get(),
                "KRAKEN_COST_MIN_USD": fields["KRAKEN_COST_MIN_USD"].get(),
                "ALLOW_MIN_UPSIZE": fields["ALLOW_MIN_UPSIZE"].get(),
                "RESET_SIM_WALLET_ON_START": fields["RESET_SIM_WALLET_ON_START"].get(),
                "ACCOUNT_INFO_REFRESH_SECONDS": fields["ACCOUNT_INFO_REFRESH_SECONDS"].get(),
                "LIVE_TRADES_REFRESH_SECONDS": fields["LIVE_TRADES_REFRESH_SECONDS"].get(),
                "UI_REFRESH_SECONDS": fields["UI_REFRESH_SECONDS"].get(),
                "STALE_PRICE_SECONDS": fields["STALE_PRICE_SECONDS"].get(),
                "STALE_WARN_INTERVAL_SECONDS": fields["STALE_WARN_INTERVAL_SECONDS"].get(),
                "STALE_GRACE_SECONDS": fields["STALE_GRACE_SECONDS"].get(),
                "BLOCK_ON_STALE_PRICE": fields["BLOCK_ON_STALE_PRICE"].get(),
                "SYMBOLS": fields["SYMBOLS"].get(),
                "TIMEFRAME": fields["TIMEFRAME"].get(),
                "LLM_CHECK_INTERVAL": fields["LLM_CHECK_INTERVAL"].get()
            }

            try:
                updates["CAPITAL"] = float(updates["CAPITAL"])
                updates["USE_LLM"] = str(updates["USE_LLM"]).lower() in ("true", "1", "yes", "y")
                updates["MIN_NOTIONAL"] = float(updates["MIN_NOTIONAL"])
                updates["KRAKEN_COST_MIN_USD"] = float(updates["KRAKEN_COST_MIN_USD"])
                updates["ALLOW_MIN_UPSIZE"] = str(updates["ALLOW_MIN_UPSIZE"]).lower() in ("true", "1", "yes", "y")
                updates["RESET_SIM_WALLET_ON_START"] = str(updates["RESET_SIM_WALLET_ON_START"]).lower() in ("true", "1", "yes", "y")
                updates["ACCOUNT_INFO_REFRESH_SECONDS"] = int(updates["ACCOUNT_INFO_REFRESH_SECONDS"])
                updates["LIVE_TRADES_REFRESH_SECONDS"] = int(updates["LIVE_TRADES_REFRESH_SECONDS"])
                updates["UI_REFRESH_SECONDS"] = int(updates["UI_REFRESH_SECONDS"])
                updates["STALE_PRICE_SECONDS"] = int(updates["STALE_PRICE_SECONDS"])
                updates["STALE_WARN_INTERVAL_SECONDS"] = int(updates["STALE_WARN_INTERVAL_SECONDS"])
                updates["STALE_GRACE_SECONDS"] = int(updates["STALE_GRACE_SECONDS"])
                updates["BLOCK_ON_STALE_PRICE"] = str(updates["BLOCK_ON_STALE_PRICE"]).lower() in ("true", "1", "yes", "y")
                updates["SYMBOLS"] = _parse_symbols(updates["SYMBOLS"])
                updates["LLM_CHECK_INTERVAL"] = int(updates["LLM_CHECK_INTERVAL"])
            except Exception:
                _show_toast(settings, "Invalid value type. Please check numeric and boolean fields.")
                return

            _update_config_file(config_path, updates, None)
            _show_toast(settings, "Settings saved.", kind="success")
            settings.destroy()

        def rebuild_positions():
            try:
                os.makedirs("data", exist_ok=True)
                with open(os.path.join("data", "rebuild_positions.flag"), "w", encoding="utf-8") as f:
                    f.write(str(time.time()))
                _show_toast(settings, "Rebuild requested. Restart the bot.", kind="success")
            except Exception:
                _show_toast(settings, "Failed to create rebuild flag.")

        btn_row = tk.Frame(scroll_frame, bg="#000000")
        btn_row.pack(anchor="w", pady=12)
        ttk.Button(btn_row, text="Save Settings", style="Primary.TButton", command=save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="Rebuild Positions", style="Primary.TButton", command=rebuild_positions).pack(side=tk.LEFT)

    def start_bot():
        updates = {
            "RUN_MODE": mode_var.get(),
            "TRADING_STYLE": "autopilot",
            "EXCHANGE": exchange_var.get()
        }
        _update_config_file(config_path, updates, None)
        root.destroy()

    start_btn = tk.Button(
        btn_frame,
        text="Start",
        command=start_bot,
        bg="#16a34a",
        fg="#ffffff",
        activebackground="#22c55e",
        activeforeground="#ffffff",
        highlightthickness=2,
        highlightbackground="#22c55e",
        highlightcolor="#22c55e",
        relief="flat",
        font=("Helvetica", 11, "bold"),
        cursor="hand2"
    )
    start_btn.configure(width=14, height=2)
    start_btn.pack(side=tk.LEFT, padx=(0, 10))
    def _hover(btn, bg, fg):
        btn.config(bg=bg, fg=fg)
    start_btn.bind("<Enter>", lambda _e: _hover(start_btn, "#0ecd54", "#ffffff"))
    start_btn.bind("<Leave>", lambda _e: _hover(start_btn, "#16a34a", "#ffffff"))
    settings_btn = tk.Button(
        btn_frame,
        text="Settings",
        command=open_settings,
        bg="#000000",
        fg="#ffffff",
        activebackground="#262626",
        activeforeground="#ffffff",
        highlightthickness=2,
        highlightbackground="#22c55e",
        highlightcolor="#22c55e",
        font=("Helvetica", 11, "bold"),
        cursor="hand2"
    )
    settings_btn.configure(width=14, height=2)
    settings_btn.pack(side=tk.LEFT)
    settings_btn.bind("<Enter>", lambda _e: _hover(settings_btn, "#262626", "#ffffff"))
    settings_btn.bind("<Leave>", lambda _e: _hover(settings_btn, "#000000", "#ffffff"))

    root.mainloop()
