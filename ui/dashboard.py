"""
Trading dashboard UI.

Responsibilities:
- Show equity chart, positions, and orders
- Display system status indicators
- Provide STOP/Restart actions
"""
import tkinter as tk
from tkinter import messagebox
import os
import json
import math
import time
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

class RealTimeEquityPlot:
    def __init__(self, trader, csv_file="trades.csv", show_mock_wallet=True, orders_title="Mock Orders"):
        self.trader = trader
        self.show_mock_wallet = show_mock_wallet
        self.bg = "#000000"
        self.panel = "#000000"
        self.text = "#e6e6e6"
        self.muted = "#9aa4b2"
        self.csv_file = csv_file
        self.equity_history = []
        self.equity_timestamps = []
        self.equity_history_path = "data/equity_history.json"
        self._equity_save_counter = 0
        self._history_checked = False
        self.start_equity = None
        self._data_ready = False
        self._chart_ready = False
        self._data_ready_once = False
        self._loading_spinner_active = False
        self._loading_spinner_index = 0
        self._account_first_seen = None
        self._account_ready_wait = 6.0
        self._last_account_value_ready = False
        self._splash_visible = False
        self._splash_spinner_index = 0
        self._heavy_ui_built = False
        self._text_refresh_last = 0.0
        self._text_refresh_interval = 2.0
        self._prompt_refresh_last = 0.0
        self._prompt_refresh_interval = 3.0
        self._chart_refresh_last = 0.0
        self._chart_refresh_interval = 2.0
        self._ui_update_last = 0.0
        self._ui_update_interval = 0.5
        self._selected_range = "ALL"
        self._range_start_ts = None
        self._range_span_seconds = None
        self.market_history = {}
        self._win_loss_counts = (0, 0)
        self._win_loss_mtime = 0.0
        self._win_loss_last_check = 0.0
        self._win_loss_jsonl_mtime = 0.0
        self._win_loss_trades_mtime = 0.0
        self._win_loss_cache_path = os.path.join("data", "wl_cache.json")
        self._realized_cache_mtime = 0.0
        self._realized_cache_total = None
        self._realized_cache_trades = 0
        self._autopilot_tune_mtime = 0.0
        self._autopilot_tune_last = ""
        self._snapshot_stats_mtime = 0.0
        self._snapshot_first_value = None
        self._snapshot_first_ts = None
        self._snapshot_peak_value = None
        self._snapshot_peak_ts = None
        self._load_equity_history()

        # -----------------------------
        # Tkinter root
        # -----------------------------
        self.root = tk.Tk()
        self.root.title("Trading Bot Dashboard")
        self.root.configure(bg=self.bg)
        self.root.geometry("1600x1200")

        # Splash screen (shown immediately while UI builds + data sync)
        self.root.withdraw()
        self.splash = tk.Toplevel(self.root)
        self.splash.configure(bg="#000000")
        self.splash.overrideredirect(True)
        self.splash.geometry("1600x1200+80+40")
        self.splash.attributes("-topmost", True)
        self.splash_spinner = tk.Canvas(
            self.splash,
            width=72,
            height=72,
            bg="#000000",
            highlightthickness=0
        )
        self.splash_spinner.place(relx=0.5, rely=0.5, anchor="center", y=-20)
        self.splash_label = tk.Label(
            self.splash,
            text="Starting dashboard",
            bg="#000000",
            fg=self.text,
            font=("Helvetica", 14, "bold")
        )
        self.splash_label.place(relx=0.5, rely=0.5, anchor="center", y=28)
        self._set_splash_visible(True)
        self._start_splash_spinner()
        try:
            self.splash.update_idletasks()
            self.splash.update()
        except Exception:
            pass

        # Gradient background canvas
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0, bg=self.bg)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        self.bg_canvas.bind("<Configure>", self._draw_gradient_background)

        # Card container
        self.container = tk.Frame(self.bg_canvas, bg=self.panel, padx=16, pady=16)
        self.container_id = self.bg_canvas.create_window(24, 24, window=self.container, anchor="nw")
        self.bg_canvas.bind("<Configure>", self._resize_container)

        # Top status bar
        self.status_frame = tk.Frame(self.container, bg=self.panel)
        self.status_frame.pack(fill=tk.X, pady=(0, 6))
        self.status_badge = tk.Label(
            self.status_frame,
            text="MODE: N/A | STYLE: N/A",
            bg="#1a1a1a",
            fg="#dbeafe",
            padx=8,
            pady=4,
            font=("Helvetica", 10, "bold")
        )
        self.status_badge.pack(side=tk.RIGHT)

        self.running_dot = tk.Canvas(self.status_frame, width=12, height=12, bg=self.panel, highlightthickness=0)
        self.running_dot.pack(side=tk.RIGHT, padx=(0, 8))
        self._dot_state = True
        self.restart_requested = False

        self.restart_btn = tk.Button(
            self.status_frame,
            text="\u2190 Exit to Startup",
            bg="#000000",
            fg="#e6e6e6",
            font=("Helvetica", 10, "bold"),
            command=self._on_restart_click,
            activebackground="#3d3d3d",
            activeforeground="#ffffff",
            highlightthickness=2,
            highlightbackground="#ffffff",
            highlightcolor="#ffffff",
            cursor="hand2",
            relief="flat",
            bd=0
        )
        self.restart_btn.configure(height=3)
        self.restart_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._install_button_hover(self.restart_btn, "#000000", "#2b2b2b", glow="#16a34a", base_glow="#000000", hover_fg="#ffffff", base_fg="#e6e6e6")
        self.restart_notice = tk.Label(
            self.status_frame,
            text="",
            bg=self.panel,
            fg="#eab308",
            font=("Helvetica", 10, "bold")
        )
        self.restart_notice.pack(side=tk.LEFT, padx=(8, 0))
        self._bot_start_toast_shown = False

        # Debug status removed

        # Status indicators
        self.indicator_frame = tk.Frame(self.container, bg=self.panel)
        self.indicator_frame.pack(fill=tk.X, pady=(0, 6))
        self.indicators = {}
        self.market_indicators = {}
        for name in ["Bot", "LLM", "Exchange", "RSS"]:
            box = tk.Frame(self.indicator_frame, bg=self.panel)
            box.pack(side=tk.LEFT, padx=(0, 16))
            dot = tk.Canvas(box, width=10, height=10, bg=self.panel, highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 6))
            label = tk.Label(
                box,
                text=f"{name}: stopped",
                bg=self.panel,
                fg=self.muted,
                font=("Helvetica", 10, "bold")
            )
            label.pack(side=tk.LEFT)
            self.indicators[name] = {"dot": dot, "label": label, "on": False}

        # -----------------------------
        # Account summary frame
        # -----------------------------
        self.header_frame = tk.Frame(self.container, padx=10, pady=5, bg=self.panel)
        self.header_frame.pack(fill=tk.X)
        self._divider(self.container)

        self.total_equity_label = tk.Label(
            self.header_frame, text="Total Value: $0.00", font=("Helvetica", 14, "bold"), bg=self.panel, fg=self.text
        )
        self.total_equity_label.pack(side=tk.LEFT, padx=10)

        self.syncing_label = tk.Label(
            self.header_frame, text="", font=("Helvetica", 12, "bold"), bg=self.panel, fg="#eab308"
        )
        self.syncing_label.pack(side=tk.LEFT, padx=10)

        self.pnl_label = tk.Label(
            self.header_frame, text="Open PnL: $0.00 (0.0%)", font=("Helvetica", 12), bg=self.panel, fg=self.text
        )
        self.pnl_label.pack(side=tk.LEFT, padx=10)
        
        self.realized_pnl_label = tk.Label(
            self.header_frame, text="Historical PnL: $0.00 (0.0%)", font=("Helvetica", 12), bg=self.panel, fg=self.text
        )
        self.realized_pnl_label.pack(side=tk.LEFT, padx=10)

        self.live_account_label = tk.Label(
            self.header_frame, text="Exchange Cash: N/A", font=("Helvetica", 12), bg=self.panel, fg=self.text
        )
        self.live_account_label.pack(side=tk.LEFT, padx=10)

        self.win_loss_label = tk.Label(
            self.header_frame, text="Wins/Losses: 0/0 (0.0%)", font=("Helvetica", 12), bg=self.panel, fg=self.text
        )
        self.win_loss_label.pack(side=tk.LEFT, padx=10)

        self.autopilot_label = tk.Label(
            self.header_frame, text="", font=("Helvetica", 12), bg=self.panel, fg="#22c55e"
        )
        self.autopilot_label.pack(side=tk.LEFT, padx=10)

        self.mock_wallet_label = None
        if self.show_mock_wallet:
            self.mock_wallet_label = tk.Label(
                self.header_frame, text="Wallet Balances: $0.00 | BTC:0.0000 | ETH:0.0000 | SOL:0.0000", font=("Helvetica", 12), bg=self.panel, fg=self.text
            )
            self.mock_wallet_label.pack(side=tk.LEFT, padx=10)

        # Panic button
        self.panic_btn = tk.Button(
            self.header_frame,
            text="⚠ Emergency Stop",
            bg="#ff1100",
            fg="white",
            font=("Helvetica", 12, "bold"),
            command=self._on_panic_click,
            activebackground="#ff6b6b",
            activeforeground="#ffffff",
            highlightthickness=2,
            highlightbackground="#ff1100",
            highlightcolor="#ff1100",
            cursor="hand2",
            relief="flat",
            bd=0
        )
        self.panic_btn.pack(side=tk.RIGHT, padx=10)
        self._install_button_hover(self.panic_btn, "#ff1100", "#ff4d4d", glow="#ff6b6b", base_glow="#ff1100", hover_fg="#ffffff", base_fg="#ffffff")
        self.panic_callback = None

        # Initialize win/loss display from logs on startup
        cached_wins, cached_losses = self._load_win_loss_cache()
        if cached_wins or cached_losses:
            self._win_loss_counts = (cached_wins, cached_losses)
        wins, losses = self._load_win_loss_counts()
        total = wins + losses
        win_rate = (wins / total * 100.0) if total else 0.0
        self.win_loss_label.config(text=f"Wins/Losses: {wins}/{losses} ({win_rate:.1f}%)")

        # -----------------------------
        # Main layout (chart + side panels)
        # -----------------------------
        self.main_frame = tk.Frame(self.container, bg=self.panel)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # 60/40 split for chart vs side panels
        self.main_frame.columnconfigure(0, weight=6, minsize=700)
        self.main_frame.columnconfigure(1, weight=4, minsize=500)
        self.main_frame.rowconfigure(0, weight=1)

        self.chart_frame = tk.Frame(self.main_frame, bg=self.panel)
        self.chart_frame.grid(row=0, column=0, sticky="nsew")

        self.side_frame = tk.Frame(self.main_frame, padx=10, pady=5, bg=self.panel)
        self.side_frame.grid(row=0, column=1, sticky="nsew")
        self.side_frame.rowconfigure(0, weight=1)

        # Placeholder before heavy UI builds
        self.chart_placeholder = tk.Label(
            self.chart_frame,
            text="Preparing dashboard...",
            bg=self.panel,
            fg=self.muted,
            font=("Helvetica", 12, "bold")
        )
        self.chart_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Heavy UI widgets are built lazily after initial sync
        self.fig = None
        self.ax = None
        self.line = None
        self.baseline = max(0, getattr(self.trader, "starting_capital", 0))
        self.baseline_line = None
        self.canvas = None
        self.chart_sync_label = None
        self._chart_sync_index = 0
        self.range_frame = None
        self.range_buttons = {}
        self.positions_frame = None
        self.market_frame = None
        self.market_cols = {}
        self.market_labels = {}
        self.market_axes = {}
        self.market_lines = {}
        self.market_canvases = {}
        self.side_cols = None
        self.side_left = None
        self.side_right = None
        self.orders_frame = None
        self.orders_scroll = None
        self.orders_text = None
        self.attempts_frame = None
        self.attempts_scroll = None
        self.attempts_text = None
        self.prompt_frame = None
        self.prompt_scroll = None
        self.prompt_text = None
        self.prompt_history = []

        # Full-screen loading overlay
        self.loading_overlay = tk.Frame(self.root, bg="#000000")
        self.loading_spinner = tk.Canvas(
            self.loading_overlay,
            width=64,
            height=64,
            bg="#000000",
            highlightthickness=0
        )
        self.loading_spinner.place(relx=0.5, rely=0.5, anchor="center", y=-18)
        self.loading_label = tk.Label(
            self.loading_overlay,
            text="Loading data",
            bg="#000000",
            fg=self.text,
            font=("Helvetica", 14, "bold")
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center", y=28)
        self._start_loading_spinner()
        self._set_loading_visible(True)

        self.ani = None

    def _draw_gradient_background(self, event):
        self.bg_canvas.delete("gradient")
        width = event.width
        height = event.height
        steps = 120
        r1, g1, b1 = self.bg_canvas.winfo_rgb("#000000")
        r2, g2, b2 = self.bg_canvas.winfo_rgb("#000000")
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
            self.bg_canvas.create_rectangle(0, y1, width, y2, outline="", fill=color, tags="gradient")

    def _build_heavy_ui(self, orders_title):
        if self._heavy_ui_built:
            return
        self._heavy_ui_built = True
        if self.chart_placeholder is not None:
            self.chart_placeholder.place_forget()

        # -----------------------------
        # Matplotlib figure for equity chart
        # -----------------------------
        self.fig = Figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(self.bg)
        self.ax.set_facecolor(self.panel)
        self.ax.set_title("Equity Over Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Equity ($)")
        self.ax.tick_params(colors=self.muted)
        self.ax.xaxis.label.set_color(self.muted)
        self.ax.yaxis.label.set_color(self.muted)
        self.ax.title.set_color(self.text)
        self.ax.xaxis.set_major_formatter(FuncFormatter(self._format_time_axis))
        self.line, = self.ax.plot([], [], color="#16a34a")
        self.baseline = max(0, getattr(self.trader, "starting_capital", 0))
        self.baseline_line = self.ax.axhline(
            y=self.baseline, color="#333333", linestyle="--", linewidth=1
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.chart_sync_label = tk.Label(
            self.chart_frame,
            text="Syncing Data...",
            bg=self.panel,
            fg="#eab308",
            font=("Helvetica", 12, "bold")
        )
        self.chart_sync_label.place(relx=0.5, rely=0.5, anchor="center")
        self._chart_sync_index = 0
        self._start_chart_sync_ellipsis()

        # Equity range filters (top-right of chart)
        self.range_frame = tk.Frame(self.chart_frame, bg=self.panel)
        self.range_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=6)
        self.range_buttons = {}
        for label in ["1D", "1W", "1M", "YTD", "ALL"]:
            btn = tk.Button(
                self.range_frame,
                text=label,
                bg="#111111",
                fg=self.text,
                font=("Helvetica", 9, "bold"),
                command=lambda l=label: self._set_range(l),
                activebackground="#2b2b2b",
                activeforeground="#ffffff",
                highlightthickness=1,
                highlightbackground="#2b2b2b",
                highlightcolor="#2b2b2b",
                cursor="hand2",
                relief="flat",
                bd=0,
                padx=8,
                pady=2
            )
            btn.pack(side=tk.LEFT, padx=4)
            self.range_buttons[label] = btn
        self._update_range_buttons()

        # -----------------------------
        # Open positions frame
        # -----------------------------
        self.positions_frame = tk.LabelFrame(self.container, text="Open Positions", padx=10, pady=5, bg=self.panel, fg=self.text)
        self.positions_frame.pack(fill=tk.X, padx=10, pady=5)
        self._divider(self.container)

        # -----------------------------
        # Market overview (3-column row)
        # -----------------------------
        self.market_frame = tk.Frame(self.container, bg=self.panel, padx=10, pady=5)
        self.market_frame.pack(fill=tk.X, padx=10, pady=(2, 6))
        for idx, sym in enumerate(["BTC/USD", "ETH/USD", "SOL/USD"]):
            col = tk.Frame(self.market_frame, bg=self.panel)
            col.grid(row=0, column=idx, sticky="nsew", padx=6)
            self.market_cols[sym] = col
            self.market_frame.columnconfigure(idx, weight=1)
            title = tk.Label(col, text=sym, font=("Helvetica", 10, "bold"), bg=self.panel, fg=self.text)
            title.pack(anchor="w")
            fig = Figure(figsize=(2.6, 1.2))
            fig.patch.set_facecolor(self.bg)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.panel)
            ax.tick_params(colors=self.muted, labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#2b2b2b")
            ax.spines["bottom"].set_color("#2b2b2b")
            line, = ax.plot([], [], color="#16a34a", linewidth=1.2)
            self.market_axes[sym] = ax
            self.market_lines[sym] = line
            canvas = FigureCanvasTkAgg(fig, master=col)
            canvas.get_tk_widget().pack(fill=tk.X, pady=(2, 2))
            self.market_canvases[sym] = canvas
            label = tk.Label(
                col,
                text="Price: $0.00 | RSI: 0.0 | EMA20: 0.0 | VWAP: 0.0 | VolRatio: 0.0",
                font=("Helvetica", 9),
                bg=self.panel,
                fg=self.text
            )
            label.pack(anchor="w", pady=(4, 0))
            self.market_labels[sym] = label
            self.market_history[sym] = []
        self._divider(self.container)

        # -----------------------------
        # Side panels (two-column layout)
        # -----------------------------
        self.side_cols = tk.Frame(self.side_frame, bg=self.panel)
        self.side_cols.pack(fill=tk.BOTH, expand=True)
        self.side_left = tk.Frame(self.side_cols, bg=self.panel)
        self.side_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.side_right = tk.Frame(self.side_cols, bg=self.panel)
        self.side_right.grid(row=0, column=1, sticky="nsew")
        self.side_cols.columnconfigure(0, weight=1)
        self.side_cols.columnconfigure(1, weight=1)
        self.side_cols.rowconfigure(0, weight=1)

        self.orders_frame = tk.LabelFrame(self.side_left, text=orders_title, padx=5, pady=5, bg=self.panel, fg=self.text)
        self.orders_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.orders_scroll = tk.Scrollbar(self.orders_frame)
        self.orders_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.orders_scroll.set(0.0, 1.0)
        self.orders_scroll.config(bg="#000000", troughcolor="#000000", activebackground="#2b2b2b", highlightbackground="#000000")
        self.orders_text = tk.Text(self.orders_frame, height=12, width=90, yscrollcommand=self.orders_scroll.set, bg=self.panel, fg=self.text, insertbackground=self.text)
        self.orders_text.pack(fill=tk.BOTH, expand=True)
        self.orders_scroll.config(command=self.orders_text.yview)
        self.orders_text.tag_configure("FILLED", foreground="#16a34a")
        self.orders_text.tag_configure("PARTIALLY_FILLED", foreground="orange")
        self.orders_text.tag_configure("NEW", foreground="#1f6feb")
        self.orders_text.tag_configure("CANCELED", foreground="#9aa4b2")
        self.orders_text.tag_configure("REJECTED", foreground="#e11d48")
        self.orders_text.tag_configure("BLOCKED", foreground="#f59e0b")

        self.attempts_frame = tk.LabelFrame(self.side_left, text="Attempted Orders", padx=5, pady=5, bg=self.panel, fg=self.text)
        self.attempts_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.attempts_scroll = tk.Scrollbar(self.attempts_frame)
        self.attempts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.attempts_scroll.set(0.0, 1.0)
        self.attempts_scroll.config(bg="#000000", troughcolor="#000000", activebackground="#2b2b2b", highlightbackground="#000000")
        self.attempts_text = tk.Text(self.attempts_frame, height=12, width=90, yscrollcommand=self.attempts_scroll.set, bg=self.panel, fg=self.text, insertbackground=self.text)
        self.attempts_text.pack(fill=tk.BOTH, expand=True)
        self.attempts_scroll.config(command=self.attempts_text.yview)
        self.attempts_text.tag_configure("SUCCESS", foreground="#16a34a")
        self.attempts_text.tag_configure("REJECTED", foreground="#e11d48")
        self.attempts_text.tag_configure("ERROR", foreground="#e11d48")
        self.attempts_text.tag_configure("BLOCKED", foreground="#f59e0b")

        self.prompt_frame = tk.LabelFrame(self.side_right, text="LLM Brain", padx=5, pady=5, bg=self.panel, fg=self.text)
        self.prompt_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.prompt_scroll = tk.Scrollbar(self.prompt_frame)
        self.prompt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.prompt_scroll.set(0.0, 1.0)
        self.prompt_scroll.config(bg="#000000", troughcolor="#000000", activebackground="#2b2b2b", highlightbackground="#000000")
        self.prompt_text = tk.Text(self.prompt_frame, height=18, width=90, yscrollcommand=self.prompt_scroll.set, bg=self.panel, fg=self.text, insertbackground=self.text)
        self.prompt_text.pack(fill=tk.BOTH, expand=True)
        self.prompt_scroll.config(command=self.prompt_text.yview)
        self.prompt_history = []

        # -----------------------------
        # Start Matplotlib animation
        # -----------------------------
        self.ani = FuncAnimation(self.fig, self._animate, interval=int(self._chart_refresh_interval * 1000), cache_frame_data=False)

    def _resize_container(self, event):
        self.bg_canvas.itemconfigure(self.container_id, width=max(300, event.width - 48))

    def _divider(self, parent):
        divider = tk.Frame(parent, bg="#000000", height=1)
        divider.pack(fill=tk.X, padx=6, pady=6)

    def _install_button_hover(self, btn, base_bg, hover_bg, glow=None, base_glow=None, hover_fg=None, base_fg=None):
        def _hover_on(_evt=None):
            btn.config(bg=hover_bg)
            if hover_fg is not None:
                btn.config(fg=hover_fg)
            if glow:
                btn.config(highlightbackground=glow, highlightcolor=glow)
        def _hover_off(_evt=None):
            btn.config(bg=base_bg)
            if base_fg is not None:
                btn.config(fg=base_fg)
            if glow:
                reset = base_glow if base_glow is not None else base_bg
                btn.config(highlightbackground=reset, highlightcolor=reset)
        btn.bind("<Enter>", _hover_on)
        btn.bind("<Leave>", _hover_off)
        btn.bind("<Motion>", _hover_on)

    def _start_loading_spinner(self):
        if self._loading_spinner_active:
            return
        self._loading_spinner_active = True
        self._spin_loading()

    def _spin_loading(self):
        if not self._loading_spinner_active:
            return
        self.loading_spinner.delete("spinner")
        angle = (self._loading_spinner_index * 12) % 360
        self.loading_spinner.create_arc(
            6, 6, 58, 58,
            start=angle,
            extent=280,
            style=tk.ARC,
            width=4,
            outline="#16a34a",
            tags="spinner"
        )
        self._loading_spinner_index += 1
        self.root.after(50, self._spin_loading)

    def _start_chart_sync_ellipsis(self):
        self._spin_chart_sync_ellipsis()

    def _spin_chart_sync_ellipsis(self):
        dots = "." * (self._chart_sync_index % 4)
        self.chart_sync_label.config(text=f"Syncing Data{dots}")
        self._chart_sync_index += 1
        self.root.after(400, self._spin_chart_sync_ellipsis)

    def _start_splash_spinner(self):
        self._spin_splash()

    def _spin_splash(self):
        if not self._splash_visible:
            return
        self.splash_spinner.delete("spinner")
        angle = (self._splash_spinner_index * 12) % 360
        self.splash_spinner.create_arc(
            6, 6, 66, 66,
            start=angle,
            extent=280,
            outline="#22c55e",
            width=4,
            style=tk.ARC,
            tags="spinner"
        )
        self._splash_spinner_index += 1
        self.root.after(50, self._spin_splash)

    def _set_splash_visible(self, visible):
        self._splash_visible = bool(visible)
        if visible:
            try:
                self.splash.deiconify()
                self.splash.lift()
                self.splash.attributes("-topmost", True)
                self.splash.update_idletasks()
                self.splash.update()
            except Exception:
                pass
            self._start_splash_spinner()
        else:
            try:
                self.splash.withdraw()
            except Exception:
                pass

    def _set_loading_visible(self, visible):
        if self._splash_visible:
            return
        if visible:
            self.loading_overlay.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
            self.loading_overlay.lift()
            self._start_loading_spinner()
        else:
            self.loading_overlay.place_forget()

    def _set_range(self, label):
        self._selected_range = label
        self._update_range_buttons()

    def _update_range_buttons(self):
        for label, btn in self.range_buttons.items():
            is_active = label == self._selected_range
            if is_active:
                btn.config(bg="#16a34a", fg="#ffffff", highlightbackground="#16a34a", highlightcolor="#16a34a")
            else:
                btn.config(bg="#111111", fg=self.text, highlightbackground="#2b2b2b", highlightcolor="#2b2b2b")

    def _format_time_axis(self, value, _pos=None):
        if self._range_start_ts is None:
            return f"{int(value)}"
        ts = self._range_start_ts + value
        try:
            # Prefer format based on the selected range.
            if self._selected_range in ("1W", "1M", "YTD"):
                tstruct = time.localtime(ts)
                return f"{time.strftime('%b', tstruct)} {tstruct.tm_mday}"
            if self._selected_range == "ALL":
                span = self._range_span_seconds or 0
                if span >= 180 * 86400:
                    return time.strftime("%b %Y", time.localtime(ts))
                tstruct = time.localtime(ts)
                return f"{time.strftime('%b', tstruct)} {tstruct.tm_mday}"
            return time.strftime("%H:%M", time.localtime(ts))
        except Exception:
            return f"{int(value)}"

    def _load_win_loss_counts(self):
        try:
            wins = 0
            losses = 0
            prev_wins, prev_losses = self._win_loss_counts
            def _maybe_keep_previous(new_wins, new_losses):
                if new_wins < prev_wins or new_losses < prev_losses:
                    return prev_wins, prev_losses
                if new_wins == 0 and new_losses == 0 and (prev_wins or prev_losses):
                    return prev_wins, prev_losses
                return new_wins, new_losses
            # Prefer JSONL with realized_pnl/outcome when available
            jsonl_path = os.path.join("logs", "order_actions.jsonl")
            if os.path.exists(jsonl_path):
                mtime = os.path.getmtime(jsonl_path)
                if mtime != self._win_loss_jsonl_mtime:
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            pnl = row.get("realized_pnl")
                            outcome = (row.get("outcome") or "").strip().upper()
                            if pnl not in (None, "", "None"):
                                try:
                                    pnl_val = float(pnl)
                                except Exception:
                                    pnl_val = None
                                if pnl_val is not None:
                                    if pnl_val > 0:
                                        wins += 1
                                        continue
                                    if pnl_val < 0:
                                        losses += 1
                                        continue
                            if outcome == "W":
                                wins += 1
                            elif outcome == "L":
                                losses += 1
                    self._win_loss_jsonl_mtime = mtime
                    wins, losses = _maybe_keep_previous(wins, losses)
                    self._win_loss_counts = (wins, losses)
                    self._save_win_loss_cache()
                    return (wins, losses)
            # Prefer computed realized PnL from live trades log when available
            trades_path = os.path.join("data", "live_trades.csv")
            if os.path.exists(trades_path):
                mtime = os.path.getmtime(trades_path)
                if mtime != self._win_loss_trades_mtime:
                    pnl_by_symbol = {}
                    qty_by_symbol = {}
                    with open(trades_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            sym = row.get("symbol") or ""
                            side = (row.get("side") or "").upper()
                            try:
                                qty = float(row.get("qty") or 0.0)
                                price = float(row.get("price") or 0.0)
                            except Exception:
                                continue
                            if qty <= 0 or price <= 0:
                                continue
                            cost = pnl_by_symbol.get(sym, 0.0)
                            held = qty_by_symbol.get(sym, 0.0)
                            if side == "BUY":
                                cost += qty * price
                                held += qty
                            elif side == "SELL":
                                if held > 0:
                                    avg_cost = cost / held if held else 0.0
                                    reduce_qty = min(held, qty)
                                    realized = (price - avg_cost) * reduce_qty
                                    if realized > 0:
                                        wins += 1
                                    elif realized < 0:
                                        losses += 1
                                    cost -= avg_cost * reduce_qty
                                    held -= reduce_qty
                            pnl_by_symbol[sym] = cost
                            qty_by_symbol[sym] = held
                    self._win_loss_trades_mtime = mtime
                    wins, losses = _maybe_keep_previous(wins, losses)
                    self._win_loss_counts = (wins, losses)
                    self._save_win_loss_cache()
                    return (wins, losses)

            # Fallback to CSV outcome
            path = os.path.join("logs", "order_actions.csv")
            if not os.path.exists(path):
                return self._win_loss_counts
            mtime = os.path.getmtime(path)
            if mtime == self._win_loss_mtime:
                return self._win_loss_counts
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    outcome = (row.get("outcome") or "").strip().upper()
                    if outcome == "W":
                        wins += 1
                    elif outcome == "L":
                        losses += 1
            self._win_loss_mtime = mtime
            wins, losses = _maybe_keep_previous(wins, losses)
            self._win_loss_counts = (wins, losses)
            self._save_win_loss_cache()
            return (wins, losses)
        except Exception:
            return self._win_loss_counts

    def _compute_realized_stats(self):
        trades_path = os.path.join("data", "live_trades.csv")
        if os.path.exists(trades_path):
            mtime = os.path.getmtime(trades_path)
            if mtime != self._realized_cache_mtime:
                total = 0.0
                trade_count = 0
                pnl_by_symbol = {}
                qty_by_symbol = {}
                try:
                    with open(trades_path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            sym = row.get("symbol") or ""
                            side = (row.get("side") or "").upper()
                            try:
                                qty = float(row.get("qty") or 0.0)
                                price = float(row.get("price") or 0.0)
                            except Exception:
                                continue
                            if qty <= 0 or price <= 0:
                                continue
                            cost = pnl_by_symbol.get(sym, 0.0)
                            held = qty_by_symbol.get(sym, 0.0)
                            if side == "BUY":
                                cost += qty * price
                                held += qty
                            elif side == "SELL":
                                if held > 0:
                                    avg_cost = cost / held if held else 0.0
                                    reduce_qty = min(held, qty)
                                    realized = (price - avg_cost) * reduce_qty
                                    total += realized
                                    trade_count += 1
                                    cost -= avg_cost * reduce_qty
                                    held -= reduce_qty
                            pnl_by_symbol[sym] = cost
                            qty_by_symbol[sym] = held
                    self._realized_cache_total = total
                    self._realized_cache_trades = trade_count
                    self._realized_cache_mtime = mtime
                except Exception:
                    pass
            return self._realized_cache_total, self._realized_cache_trades

        return None, 0

    def _save_win_loss_cache(self):
        try:
            os.makedirs(os.path.dirname(self._win_loss_cache_path), exist_ok=True)
            with open(self._win_loss_cache_path, "w", encoding="utf-8") as f:
                json.dump({"wins": self._win_loss_counts[0], "losses": self._win_loss_counts[1]}, f)
        except Exception:
            pass

    def _load_win_loss_cache(self):
        try:
            if not os.path.exists(self._win_loss_cache_path):
                return (0, 0)
            with open(self._win_loss_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            wins = int(data.get("wins", 0) or 0)
            losses = int(data.get("losses", 0) or 0)
            return (wins, losses)
        except Exception:
            return (0, 0)

    def _load_autopilot_last_tune(self):
        try:
            path = os.path.join("logs", "autopilot_tuning.jsonl")
            if not os.path.exists(path):
                return ""
            mtime = os.path.getmtime(path)
            if mtime == self._autopilot_tune_mtime:
                return self._autopilot_tune_last
            last_line = ""
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
            if not last_line:
                return ""
            payload = json.loads(last_line)
            ts = float(payload.get("timestamp", 0.0) or 0.0)
            reason = payload.get("reason") or "update"
            updates = payload.get("updates") or {}
            updates_str = ", ".join([f"{k}={v}" for k, v in updates.items()]) if isinstance(updates, dict) else ""
            updates_str = updates_str[:120] + ("…" if len(updates_str) > 120 else "")
            tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "unknown"
            message = f"Autopilot tuned {tstr} ({reason}) {updates_str}"
            self._autopilot_tune_mtime = mtime
            self._autopilot_tune_last = message
            return message
        except Exception:
            return self._autopilot_tune_last

    def _load_portfolio_snapshot_stats(self):
        path = os.path.join("logs", "portfolio_snapshots.jsonl")
        if not os.path.exists(path):
            return (None, None, None, None)
        try:
            mtime = os.path.getmtime(path)
            if mtime == self._snapshot_stats_mtime:
                return (
                    self._snapshot_first_value,
                    self._snapshot_first_ts,
                    self._snapshot_peak_value,
                    self._snapshot_peak_ts,
                )
            first_val = None
            first_ts = None
            peak_val = None
            peak_ts = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = float(row.get("timestamp") or 0.0)
                    snap = row.get("snapshot") or {}
                    val = None
                    for key in ("equity_usd", "total_usd", "cash_usd"):
                        if snap.get(key) not in (None, "", "None"):
                            try:
                                val = float(snap.get(key))
                            except Exception:
                                val = None
                            break
                    if val is None or not math.isfinite(val) or val <= 0:
                        continue
                    if first_ts is None or (ts and ts < first_ts):
                        first_ts = ts
                        first_val = val
                    if peak_val is None or val > peak_val:
                        peak_val = val
                        peak_ts = ts
            self._snapshot_stats_mtime = mtime
            self._snapshot_first_value = first_val
            self._snapshot_first_ts = first_ts
            self._snapshot_peak_value = peak_val
            self._snapshot_peak_ts = peak_ts
            return (first_val, first_ts, peak_val, peak_ts)
        except Exception:
            return (
                self._snapshot_first_value,
                self._snapshot_first_ts,
                self._snapshot_peak_value,
                self._snapshot_peak_ts,
            )

    def _get_historical_baseline(self):
        hist_peak = None
        if self.equity_history:
            try:
                hist_peak = max(float(x) for x in self.equity_history if math.isfinite(float(x)))
            except Exception:
                hist_peak = None

        _snap_first_val, _snap_first_ts, snap_peak_val, _snap_peak_ts = self._load_portfolio_snapshot_stats()
        candidates = []
        if hist_peak is not None and hist_peak > 0:
            candidates.append(hist_peak)
        if snap_peak_val is not None and snap_peak_val > 0:
            candidates.append(snap_peak_val)
        if self.start_equity is not None and self.start_equity > 0:
            candidates.append(float(self.start_equity))
        if self.baseline is not None and self.baseline > 0:
            candidates.append(float(self.baseline))
        return float(max(candidates) if candidates else 0.0)

    def _get_filtered_series(self):
        if not self.equity_history:
            return [], []
        if not self.equity_timestamps or len(self.equity_timestamps) != len(self.equity_history):
            self._range_start_ts = None
            self._range_span_seconds = None
            return list(range(len(self.equity_history))), list(self.equity_history)

        now = time.time()
        cutoff = None
        if self._selected_range == "1D":
            cutoff = now - 86400
        elif self._selected_range == "1W":
            cutoff = now - (7 * 86400)
        elif self._selected_range == "1M":
            cutoff = now - (30 * 86400)
        elif self._selected_range == "YTD":
            current = time.localtime(now)
            cutoff = time.mktime((current.tm_year, 1, 1, 0, 0, 0, 0, 0, -1))

        if cutoff is None:
            xs = list(self.equity_timestamps)
            if xs:
                base = xs[0]
                self._range_start_ts = base
                xs = [ts - base for ts in xs]
                if xs:
                    self._range_span_seconds = xs[-1] - xs[0]
            else:
                self._range_start_ts = None
                self._range_span_seconds = None
            return xs, list(self.equity_history)

        xs = []
        ys = []
        for ts, val in zip(self.equity_timestamps, self.equity_history):
            if ts >= cutoff:
                xs.append(ts)
                ys.append(val)
        if xs:
            base = xs[0]
            self._range_start_ts = base
            xs = [ts - base for ts in xs]
            if xs:
                self._range_span_seconds = xs[-1] - xs[0]
        else:
            self._range_start_ts = None
            self._range_span_seconds = None
        if len(xs) < 2:
            xs = list(self.equity_timestamps)
            if xs:
                base = xs[0]
                self._range_start_ts = base
                xs = [ts - base for ts in xs]
                if xs:
                    self._range_span_seconds = xs[-1] - xs[0]
            return xs, list(self.equity_history)
        return xs, ys

    def _pulse_dot(self):
        self.running_dot.delete("dot")
        color = "#16a34a" if self._dot_state else "#1f6feb"
        self.running_dot.create_oval(2, 2, 10, 10, fill=color, outline="")
        self._dot_state = not self._dot_state
        self.running_dot.after(700, self._pulse_dot)

    def _set_indicator(self, name, on):
        entry = self.indicators.get(name)
        if not entry:
            return
        dot = entry["dot"]
        label = entry["label"]
        dot.delete("dot")
        if on:
            dot.create_oval(2, 2, 8, 8, fill="#16a34a", outline="")
            label.config(text=f"{name}: running", fg=self.text)
        else:
            dot.create_oval(2, 2, 8, 8, fill="#3b3b3b", outline="")
            label.config(text=f"{name}: stopped", fg=self.muted)

    def set_indicator(self, name, on):
        self._set_indicator(name, on)

    def _on_panic_click(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Close all Trades and Stop?")
        dialog.configure(bg=self.panel)
        dialog.resizable(False, False)
        dialog.grab_set()

        frame = tk.Frame(dialog, bg=self.panel, padx=16, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            frame,
            text="Close all Trades and Stop?",
            bg=self.panel,
            fg=self.text,
            font=("Helvetica", 12, "bold")
        )
        title.pack(anchor="w")

        body = tk.Label(
            frame,
            text=(
                "This will shut down the bot, close all trades, and assume all current PnL. "
                "Are you sure you want to continue?"
            ),
            bg=self.panel,
            fg=self.muted,
            justify="left",
            wraplength=380
        )
        body.pack(anchor="w", pady=(6, 12))

        btn_row = tk.Frame(frame, bg=self.panel)
        btn_row.pack(fill=tk.X)

        def close_dialog():
            dialog.destroy()

        def confirm_close():
            dialog.destroy()
            if self.panic_callback:
                self.panic_callback()
            self._show_notice("Kill switch engaged…", color="#e11d48")

        cancel_btn = tk.Button(
            btn_row,
            text="Cancel",
            bg="#2a2f3a",
            fg="#e6e6e6",
            font=("Helvetica", 10, "bold"),
            command=close_dialog
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(8, 0))

        confirm_btn = tk.Button(
            btn_row,
            text="CLOSE TRADES",
            bg="#b42318",
            fg="white",
            font=("Helvetica", 10, "bold"),
            command=confirm_close
        )
        confirm_btn.pack(side=tk.RIGHT)

    def _on_restart_click(self):
        if messagebox.askyesno("CONFIRM RESTART", "Exit the bot and return to the startup screen?"):
            self.restart_requested = True
            self._show_notice("Restart scheduled…", color="#eab308")

    def _show_notice(self, text, color="#eab308", duration_ms=3000):
        self.restart_notice.config(text=text, fg=color)
        try:
            self.root.after(duration_ms, lambda: self.restart_notice.config(text=""))
        except Exception:
            pass

    def _load_equity_history(self):
        try:
            if not os.path.exists(self.equity_history_path):
                return
            with open(self.equity_history_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                history = data.get("equity") or []
                timestamps = data.get("timestamps") or []
                if isinstance(history, list):
                    self.equity_history = [float(x) for x in history][-5000:]
                if isinstance(timestamps, list):
                    self.equity_timestamps = [float(x) for x in timestamps][-5000:]
                if self.equity_history and (not self.equity_timestamps or len(self.equity_timestamps) != len(self.equity_history)):
                    now = time.time()
                    step = 1.0
                    start = now - step * (len(self.equity_history) - 1)
                    self.equity_timestamps = [start + step * i for i in range(len(self.equity_history))]
            elif isinstance(data, list):
                self.equity_history = [float(x) for x in data][-5000:]
                now = time.time()
                step = 1.0
                start = now - step * (len(self.equity_history) - 1)
                self.equity_timestamps = [start + step * i for i in range(len(self.equity_history))]
            # Drop extreme outliers on load (keeps history but removes spikes)
            if self.equity_history and self.equity_timestamps:
                cleaned_vals = []
                cleaned_ts = []
                window = []
                for ts, val in zip(self.equity_timestamps, self.equity_history):
                    if not window:
                        cleaned_vals.append(val)
                        cleaned_ts.append(ts)
                        window.append(val)
                        continue
                    median = sorted(window)[len(window) // 2]
                    if median > 0:
                        delta = abs(val - median) / median
                        if delta > 0.2:
                            continue
                    cleaned_vals.append(val)
                    cleaned_ts.append(ts)
                    window.append(val)
                    if len(window) > 20:
                        window.pop(0)
                self.equity_history = cleaned_vals[-5000:]
                self.equity_timestamps = cleaned_ts[-5000:]
        except Exception:
            pass

    def _save_equity_history(self):
        try:
            os.makedirs(os.path.dirname(self.equity_history_path), exist_ok=True)
            with open(self.equity_history_path, "w") as f:
                payload = {
                    "equity": self.equity_history[-5000:],
                    "timestamps": self.equity_timestamps[-5000:]
                }
                json.dump(payload, f)
        except Exception:
            pass

    def update(self, live_prices, sentiments=None, indicators=None, account_info=None, wallet_snapshot=None, orders=None, attempted_orders=None, prompt_text=None, status_mode=None, status_style=None, status_flags=None, rss_last_ts=None, price_timestamps=None):
        self.live_prices = live_prices
        self.sentiments = sentiments or {}
        self.market_indicators = indicators or {}
        self.account_info = account_info or {}
        self.wallet_snapshot = wallet_snapshot or {}
        self.price_timestamps = price_timestamps or {}
        if not self.account_info or self.account_info.get("total_usd") is None:
            self._show_notice("Syncing balances...", color="#94a3b8", duration_ms=1500)

        # Determine data readiness for charting
        required_symbols = list(self.market_labels.keys()) if self.market_labels else ["BTC/USD", "ETH/USD", "SOL/USD"]
        have_prices = all(self.live_prices.get(sym) is not None for sym in required_symbols)
        have_equity = bool(self.wallet_snapshot) or (self.account_info.get("total_usd") is not None)
        have_indicators = True
        for sym in required_symbols:
            ind = self.market_indicators.get(sym, {})
            if ind.get("ema20") is None or ind.get("vwap") is None or ind.get("vol_ratio") is None:
                have_indicators = False
                break
        self._data_ready = bool(have_prices and have_equity and have_indicators)
        have_account = self.account_info and (self.account_info.get("total_usd") is not None) and (self.account_info.get("cash_usd") is not None)
        if have_account and self._account_first_seen is None:
            self._account_first_seen = time.time()
        positions_ready = bool(self.account_info.get("positions_ready")) if isinstance(self.account_info, dict) else False
        total_usd = float(self.account_info.get("total_usd") or 0.0) if self.account_info else 0.0
        cash_usd = float(self.account_info.get("cash_usd") or 0.0) if self.account_info else 0.0
        positions_count = self.account_info.get("positions_count") if isinstance(self.account_info, dict) else None
        has_non_cash = bool(self.account_info.get("has_non_cash_balances")) if isinstance(self.account_info, dict) else False
        value_has_positions = (total_usd - cash_usd) >= max(1.0, 0.005 * max(total_usd, 1.0))
        if positions_count is not None and positions_count == 0 and not has_non_cash:
            if self._account_first_seen is not None:
                wait_elapsed = (time.time() - self._account_first_seen) >= self._account_ready_wait
            else:
                wait_elapsed = False
            value_has_positions = bool(wait_elapsed and total_usd >= cash_usd and total_usd > 0)
        account_value_ready = bool(
            have_account
            and positions_ready
            and have_prices
            and have_indicators
            and value_has_positions
        )
        skip_updates = False
        if not account_value_ready:
            self._chart_ready = False
            if self.chart_sync_label is not None:
                self.chart_sync_label.lift()
            self.syncing_label.config(text="Syncing data...")
            self._set_splash_visible(True)
            self._set_loading_visible(True)
            skip_updates = True
        else:
            if not self._heavy_ui_built:
                self._build_heavy_ui(self.orders_frame.cget("text") if self.orders_frame else "Live Trades")
            if self.syncing_label.cget("text"):
                self.syncing_label.config(text="")
            if not self._chart_ready:
                self._chart_ready = True
                if self.chart_sync_label is not None:
                    self.chart_sync_label.place_forget()
                if not self._last_account_value_ready:
                    self._show_notice("Synced", color="#16a34a", duration_ms=2000)
            if self._splash_visible:
                self._set_splash_visible(False)
                try:
                    self.root.deiconify()
                    self.root.lift()
                    self.root.focus_force()
                except Exception:
                    pass
            self._set_loading_visible(not self._data_ready)
        self._last_account_value_ready = account_value_ready

        # Update equity history
        if skip_updates:
            self.root.update_idletasks()
            try:
                self.root.update()
            except Exception:
                pass
            if self.canvas is not None:
                self.canvas.draw_idle()
            return
        wallet_total_usd = None
        equity = None
        if self.wallet_snapshot:
            total_usd = 0.0
            balances = self.wallet_snapshot.get("balances", {})
            for asset, bal in balances.items():
                if asset == "USD":
                    total_usd += float(bal)
                else:
                    pair = f"{asset}/USD"
                    price = self.live_prices.get(pair)
                    if price is None and "/" in pair:
                        price = self.live_prices.get(pair.replace("/", ""))
                    if price is not None:
                        total_usd += float(bal) * float(price)
            wallet_total_usd = float(total_usd)
            equity = wallet_total_usd
        if equity is None:
            equity = self.trader.equity(live_prices)
        # Prefer account total when available (live source of truth)
        if self.account_info and self.account_info.get("total_usd") is not None:
            try:
                equity = float(self.account_info.get("total_usd"))
            except Exception:
                pass
        # If account total is just cash but positions exist, compute from positions to avoid cash-only spike.
        try:
            if self.account_info and self.account_info.get("cash_usd") is not None and hasattr(self.trader, "positions"):
                cash_only = float(self.account_info.get("cash_usd") or 0.0)
                if equity <= cash_only + 0.01 and getattr(self.trader, "positions", {}):
                    pos_total = 0.0
                    for sym, pos in (self.trader.positions or {}).items():
                        size = float(pos.get("size", 0.0) or 0.0)
                        if size <= 0:
                            continue
                        price = live_prices.get(sym) or live_prices.get(sym.replace("/", ""))
                        if price is None:
                            price = float(pos.get("entry_price", 0.0) or 0.0)
                        pos_total += size * float(price or 0.0)
                    equity = cash_only + pos_total
        except Exception:
            pass

        if not math.isfinite(equity) or equity <= 0:
            equity = self.equity_history[-1] if self.equity_history else 0.0

        if not self._history_checked:
            if self.equity_history:
                last = self.equity_history[-1]
                baseline = max(1.0, wallet_total_usd or 0.0, getattr(self.trader, "starting_capital", 1.0))
                if (not self.trader.positions) and (abs(equity - baseline) <= 0.05 * baseline):
                    if abs(last - equity) > max(0.25 * baseline, 2000):
                        self.equity_history = [equity]
                        self.start_equity = baseline
                elif abs(last - equity) > max(0.35 * max(equity, 1.0), 5000):
                    self.equity_history = [equity]
                    self.start_equity = baseline
            self._history_checked = True

        if self.start_equity is None:
            baseline = max(1.0, wallet_total_usd or 0.0, getattr(self.trader, "starting_capital", 1.0))
            self.start_equity = baseline
        if self.start_equity < 1000 and getattr(self.trader, "starting_capital", 0) >= 1000:
            self.start_equity = float(self.trader.starting_capital)

        # If balances jump a lot with no positions, wait before appending to avoid spikes.
        if self._data_ready and self.equity_history:
            last_equity = self.equity_history[-1]
            try:
                no_positions = not bool(self.trader.positions)
            except Exception:
                no_positions = False
            if no_positions and last_equity > 0:
                delta_pct = abs(equity - last_equity) / max(last_equity, 1.0)
                if delta_pct >= 0.2:
                    # Skip this tick; keep history intact.
                    return

        if self._data_ready and not self._data_ready_once:
            self._data_ready_once = True
        if self._data_ready:
            self.equity_history.append(equity)
            self.equity_timestamps.append(time.time())
            self._equity_save_counter += 1
            if self._equity_save_counter >= 10:
                self._save_equity_history()
                self._equity_save_counter = 0

        # Update account summary
        open_pnl = self.trader.unrealized_pnl(live_prices)
        denom = equity if equity and equity > 0 else 0.0
        if (not denom) and self.account_info and self.account_info.get("total_usd"):
            denom = float(self.account_info.get("total_usd")) or 0.0
        if not denom:
            denom = self.start_equity if self.start_equity and self.start_equity > 0 else max(1.0, getattr(self.trader, "starting_capital", 1.0))
        pnl_percent = (open_pnl / denom) * 100
        self.total_equity_label.config(text=f"Total Value: ${equity:.2f}")
        self.pnl_label.config(text=f"Open PnL: ${open_pnl:.2f} ({pnl_percent:.2f}%)")
        realized_total = getattr(self.trader, "realized_pnl_total", None)
        if realized_total is None:
            realized_map = getattr(self.trader, "realized_pnl_by_symbol", {}) or {}
            realized_total = sum(float(v or 0.0) for v in realized_map.values()) if isinstance(realized_map, dict) else 0.0
        # Historical PnL: current total value vs. historical baseline
        current_total = None
        if isinstance(account_info, dict):
            current_total = account_info.get("total_usd")
        if current_total is None:
            current_total = float(self.trader.equity(live_prices) or 0.0) if hasattr(self.trader, "equity") else None
        if current_total is None:
            current_total = 0.0
        baseline_value = self._get_historical_baseline()
        hist_pnl = float(current_total) - baseline_value if baseline_value > 0 else 0.0
        hist_pct = (hist_pnl / baseline_value * 100.0) if baseline_value > 0 else 0.0
        self.realized_pnl_label.config(text=f"Historical PnL: ${hist_pnl:.2f} ({hist_pct:.2f}%)")
        self.baseline = max(0, getattr(self.trader, "starting_capital", 0))
        self.baseline_line.set_ydata([self.baseline, self.baseline])

        # Win/Loss counters (persisted via logs)
        now = time.time()
        if (now - self._win_loss_last_check) >= 5:
            wins, losses = self._load_win_loss_counts()
            total = wins + losses
            win_rate = (wins / total * 100.0) if total else 0.0
            self.win_loss_label.config(text=f"Wins/Losses: {wins}/{losses} ({win_rate:.1f}%)")
            self._win_loss_last_check = now
            tune_msg = self._load_autopilot_last_tune()
            self.autopilot_label.config(text=tune_msg)

        if self.account_info:
            cash = self.account_info.get("cash_usd")
            if cash is not None:
                self.live_account_label.config(text=f"Exchange Cash: ${cash:.2f}")
            else:
                self.live_account_label.config(text="Exchange Cash: N/A")

        if self.mock_wallet_label is not None:
            if self.wallet_snapshot:
                balances = self.wallet_snapshot.get("balances", {})
                usd_val = float(balances.get("USD", 0.0))
                btc_val = float(balances.get("BTC", 0.0))
                eth_val = float(balances.get("ETH", 0.0))
                sol_val = float(balances.get("SOL", 0.0))
                self.mock_wallet_label.config(
                    text=(
                        f"Wallet Balances: ${usd_val:.2f} | BTC:{btc_val:.4f} | "
                        f"ETH:{eth_val:.4f} | SOL:{sol_val:.4f}"
                    )
                )
            else:
                self.mock_wallet_label.config(text="Wallet Balances: N/A")

        # -----------------------------
        # Update open positions
        # -----------------------------
        if self.positions_frame is None:
            return
        for widget in self.positions_frame.winfo_children():
            widget.destroy()

        sorted_positions = sorted(
            self.trader.positions.values(), key=lambda x: x["timestamp"], reverse=True
        )
        if not sorted_positions:
            label = tk.Label(
                self.positions_frame,
                text="No positions open",
                anchor="w",
                bg=self.panel,
                fg=self.muted
            )
            label.pack(fill=tk.X)
        for pos in sorted_positions:
            sym = pos["symbol"]
            entry = pos["entry_price"]
            size = pos["size"]
            current_price = live_prices.get(sym)
            if current_price is None and isinstance(sym, str) and "/" not in sym:
                slash_sym = f"{sym[:-3]}/{sym[-3:]}" if len(sym) > 3 else sym
                current_price = live_prices.get(slash_sym)
            if current_price is None:
                current_price = entry
            entry_value = size * entry
            current_value = size * current_price
            pnl = (current_price - entry) * size
            pnl_pct = (current_price - entry) / entry * 100
            llm_decision = pos.get("llm_decision") if isinstance(pos, dict) else None
            if not isinstance(llm_decision, dict):
                llm_decision = {}
            conf = llm_decision.get("confidence", pos.get("llm_confidence", 0.0))
            source = pos.get("entry_source", "unknown")
            trailing_pct = pos.get("trailing_stop_pct", 0.0)
            stop_loss = pos.get("stop_loss")
            take_profit = pos.get("take_profit")
            trailing_stop = pos.get("trailing_stop")
            sl_text = f"{stop_loss:.2f}" if stop_loss is not None else "N/A"
            tp_text = f"{take_profit:.2f}" if take_profit is not None else "N/A"
            trail_text = f"{trailing_stop:.2f}" if trailing_stop is not None else "N/A"
            row = tk.Frame(self.positions_frame, bg=self.panel)
            row.pack(fill=tk.X)
            arrow = "▲" if pnl_pct >= 0 else "▼"
            arrow_color = "#16a34a" if pnl_pct >= 0 else "#e11d48"
            arrow_label = tk.Label(
                row,
                text=arrow,
                anchor="w",
                width=2,
                bg=self.panel,
                fg=arrow_color,
                font=("Helvetica", 10, "bold")
            )
            arrow_label.pack(side=tk.LEFT)
            label = tk.Label(
                row,
                text=(
                    f"{sym} | Value: ${current_value:.2f} (Entry: ${entry_value:.2f}) | "
                    f"Entry: {entry:.2f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | "
                    f"Conf: {conf:.2f} | Source: {source} | SL: {sl_text} | TP: {tp_text} | "
                    f"Trail: {trailing_pct:.3f} ({trail_text})"
                ),
                anchor="w",
                bg=self.panel,
                fg=self.text
            )
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -----------------------------
        # Update market overview
        # -----------------------------
        for sym in self.market_labels.keys():
            price = live_prices.get(sym, 0.0)
            rsi = self.market_indicators.get(sym, {}).get("rsi", 0.0)
            ema20 = self.market_indicators.get(sym, {}).get("ema20", 0.0)
            vwap = self.market_indicators.get(sym, {}).get("vwap", 0.0)
            vol_ratio = self.market_indicators.get(sym, {}).get("vol_ratio", 0.0)
            history = self.market_history.get(sym, [])
            if price:
                history.append(float(price))
                if len(history) > 120:
                    history[:] = history[-120:]
                self.market_history[sym] = history
            self.market_labels[sym].config(
                text=(
                    f"Price: ${price:.2f} | RSI: {rsi:.2f} | EMA20: {ema20:.2f} | "
                    f"VWAP: {vwap:.2f} | VolRatio: {vol_ratio:.2f}"
                )
            )
            series = self.market_history.get(sym, [])
            ax = self.market_axes.get(sym)
            line = self.market_lines.get(sym)
            if ax and line and series:
                xs = list(range(len(series)))
                line.set_data(xs, series)
                min_val = min(series)
                max_val = max(series)
                span = max(max_val - min_val, 0.01)
                pad = span * 0.1
                ax.set_ylim(min_val - pad, max_val + pad)
                ax.set_xlim(0, max(10, len(series)))
        for canvas in self.market_canvases.values():
            try:
                canvas.draw_idle()
            except Exception:
                pass

        # Update side panels
        now = time.time()
        if self.orders_text is not None and orders is not None and (now - self._text_refresh_last) >= self._text_refresh_interval:
            self.orders_text.delete("1.0", tk.END)
            self.orders_text.insert(tk.END, f"Total Orders: {len(orders)}\n")
            for o in orders[-50:]:
                status = o.get("status")
                ts = o.get("timestamp") or o.get("transactTime", "")
                line = (
                    f"{ts} | {o.get('symbol')} {o.get('side')} {o.get('type')} {status} "
                    f"qty={o.get('origQty')} px={o.get('price')}\n"
                )
                tag = status if status in ("FILLED", "PARTIALLY_FILLED", "NEW", "CANCELED", "REJECTED") else None
                if tag:
                    self.orders_text.insert(tk.END, line, tag)
                else:
                    self.orders_text.insert(tk.END, line)

        if self.attempts_text is not None and attempted_orders is not None and (now - self._text_refresh_last) >= self._text_refresh_interval:
            self.attempts_text.delete("1.0", tk.END)
            self.attempts_text.insert(tk.END, f"Attempts: {len(attempted_orders)}\n")
            for o in attempted_orders[-80:]:
                if o.get("status") == "SUCCESS":
                    continue
                ts = o.get("timestamp", "")
                sym = o.get("symbol", "")
                side = o.get("side", "")
                status = o.get("status", "")
                reason = o.get("reason", "")
                qty = o.get("qty", "")
                price = o.get("price", "")
                line = f"{ts} | {sym} {side} {status} {reason} qty={qty} px={price}\n"
                tag = status if status in ("SUCCESS", "REJECTED", "ERROR", "BLOCKED") else None
                if tag:
                    self.attempts_text.insert(tk.END, line, tag)
                else:
                    self.attempts_text.insert(tk.END, line)

        if self.prompt_text is not None and prompt_text is not None and (now - self._prompt_refresh_last) >= self._prompt_refresh_interval:
            if isinstance(prompt_text, (list, tuple)):
                display_text = "\n".join([str(item) for item in prompt_text])
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert(tk.END, display_text)
                self.prompt_text.see(tk.END)
                self._prompt_refresh_last = now
            else:
                if isinstance(prompt_text, dict):
                    prompt_text = "\n".join([f"{k}: {v}" for k, v in prompt_text.items()])
                if not self.prompt_history or self.prompt_history[-1] != prompt_text:
                    self.prompt_history.append(prompt_text)
                if len(self.prompt_history) > 50:
                    self.prompt_history = self.prompt_history[-50:]
                self.prompt_text.delete("1.0", tk.END)
                for entry in self.prompt_history:
                    self.prompt_text.insert(tk.END, entry + "\n" + ("-" * 40) + "\n")
                self.prompt_text.see(tk.END)
                self._prompt_refresh_last = now
        if (now - self._text_refresh_last) >= self._text_refresh_interval:
            self._text_refresh_last = now

        now = time.time()
        if (now - self._ui_update_last) >= self._ui_update_interval:
            self.root.update_idletasks()
            try:
                self.root.update()
            except Exception:
                pass
            self._ui_update_last = now
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _animate(self, i):
        if not self._chart_ready or not self._heavy_ui_built or self.ax is None or self.line is None:
            return tuple()
        now = time.time()
        if (now - self._chart_refresh_last) < self._chart_refresh_interval:
            return tuple()
        self._chart_refresh_last = now
        xs, ys = self._get_filtered_series()
        if not ys:
            return self.line,

        self.line.set_data(xs, ys)
        min_val = min(ys)
        max_val = max(ys)
        span = max(max_val - min_val, 0.01)
        pad = max(span * 0.1, max_val * 0.02, 1.0)
        self.ax.set_ylim(min_val - pad, max_val + pad)
        if xs:
            self.ax.set_xlim(min(xs), max(xs))
        return self.line,

    def set_status(self, status_flags=None, status_mode=None, status_style=None, rss_last_ts=None):
        if status_mode or status_style:
            mode_text = status_mode.upper() if status_mode else "N/A"
            style_text = status_style.title() if status_style else "N/A"
            self.status_badge.config(text=f"MODE: {mode_text} | STYLE: {style_text}")
        if not hasattr(self, "_pulse_started"):
            self._pulse_started = True
            self._pulse_dot()
        if status_flags is not None:
            for key, on in status_flags.items():
                if key in self.indicators:
                    self._set_indicator(key, bool(on))
            if status_flags.get("TradingReady") and not self._bot_start_toast_shown:
                self._bot_start_toast_shown = True
                self._show_notice("Bot starting...", color="#22c55e", duration_ms=2500)
        if rss_last_ts is not None and rss_last_ts > 0:
            self._set_indicator("RSS", True)
