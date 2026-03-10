import tkinter as tk
from tkinter import ttk, messagebox
import requests
import urllib.parse
import pandas as pd
from io import StringIO
import threading
from datetime import datetime, timedelta
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HOST    = "10.40.42.27"
PORT    = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

PAGE_SIZE = 100

COL_WIDTHS = {
    "No": 50, "_time": 140, "TIME_EX": 140, "MACHINENAME": 110,
    "MACHINETYPE": 90, "UNITNAME": 90, "CARRIER": 100, "COMMANDID": 80,
    "COMMAND": 100, "OPERATION_NAME": 110, "MESSAGENAME": 110,
    "PROCESS": 70, "TRANSACTIONID": 120, "TEXT": 300, "THREAD": 70,
    "LEVEL": 60, "XML": 60, "SECSII": 60, "RESULTCODE": 80,
}

LEVEL_COLORS = {
    "ERROR": "#ffe0e0",
    "FATAL": "#ffb3b3",
    "WARN":  "#fff3cd",
    "WELL":  "#e8f5e9",
}

# ─────────────────────────────────────────────
def run_query(query: str, timeout=120):
    clean_q = " ".join(query.split())
    encoded = urllib.parse.quote(clean_q, safe="")
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    print("=" * 80)
    print("[QUERY]", clean_q)
    print("[URL  ]", url[:300])
    print("=" * 80)
    resp = requests.get(url, verify=False, timeout=timeout)
    print(f"[STATUS] {resp.status_code}")
    if resp.status_code != 200:
        print("[RESPONSE]", resp.text[:500])
    if resp.status_code == 200:
        return pd.read_csv(StringIO(resp.text))
    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")


def build_query(from_dt, to_dt, carrier, text_kw):
    from_s = from_dt.strip()
    to_s   = to_dt.strip()
    c = carrier.strip()
    t = text_kw.strip()

    if not c and not t:
        messagebox.showwarning("입력 오류", "CARRIER 또는 TEXT 검색어를 입력하세요.")
        return None

    if c and t:
        cond = f'((CARRIER=="{c}") or (TEXT=="{t}"))'
    elif c:
        cond = f'((CARRIER=="{c}") or (TEXT=="{c}"))'
    else:
        cond = f'((TEXT=="{t}"))'

    q = (
        f'fulltext from={from_s} to={to_s} '
        f'(LEVEL=="WELL" or LEVEL=="WARN" or LEVEL=="ERROR" or LEVEL=="FATAL") '
        f'and {cond} '
        f'from ts_data_view_m14a, ts_data_view_m14b, ts_data_view_m16, ts_data_view_m16b '
        f'| fields _time, TIME_EX, MACHINENAME, MACHINETYPE, UNITNAME, CARRIER, COMMANDID, COMMAND, OPERATION_NAME, MESSAGENAME, PROCESS, TRANSACTIONID, TEXT, THREAD, LEVEL, XML, SECSII, RESULTCODE '
        f'| sort _time '
        f'| limit 0 1000 '
        f'| eval No = seq() + 0'
    )
    return q


# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Logpresso Carrier Viewer")
        self.geometry("1400x860")
        self.configure(bg="#1e1e2e")
        self.df_result = None
        self.current_page = 0   # 0-based
        self._build_ui()

    def _build_ui(self):
        # 헤더
        hdr = tk.Frame(self, bg="#313244", pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔍  Logpresso Carrier Viewer",
                 bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)

        # 검색 패널
        panel = tk.Frame(self, bg="#1e1e2e", pady=10)
        panel.pack(fill="x", padx=12)

        def lbl(parent, text):
            return tk.Label(parent, text=text, bg="#1e1e2e", fg="#a6adc8",
                            font=("Segoe UI", 9))

        def entry(parent, width=18, **kw):
            e = tk.Entry(parent, width=width, bg="#313244", fg="#cdd6f4",
                         insertbackground="#cdd6f4", relief="flat",
                         font=("Consolas", 10), **kw)
            e.config(highlightthickness=1, highlightbackground="#45475a",
                     highlightcolor="#89b4fa")
            return e

        # 날짜
        r1 = tk.Frame(panel, bg="#1e1e2e")
        r1.pack(anchor="w", pady=3)
        lbl(r1, "FROM  (yyyyMMddHHmmss)").pack(side="left")
        self.e_from = entry(r1, 16)
        self.e_from.insert(0, (datetime.now() - timedelta(days=1)).strftime("%Y%m%d000000"))
        self.e_from.pack(side="left", padx=(4, 20))
        lbl(r1, "TO  (yyyyMMddHHmmss)").pack(side="left")
        self.e_to = entry(r1, 16)
        self.e_to.insert(0, datetime.now().strftime("%Y%m%d235959"))
        self.e_to.pack(side="left", padx=(4, 20))
        for label, delta in [("오늘", 0), ("어제", 1), ("2일전", 2), ("3일전", 3)]:
            tk.Button(r1, text=label, bg="#45475a", fg="#cdd6f4",
                      relief="flat", padx=8, font=("Segoe UI", 9),
                      command=lambda d=delta: self._set_date(d)
                      ).pack(side="left", padx=2)

        # 검색어
        r2 = tk.Frame(panel, bg="#1e1e2e")
        r2.pack(anchor="w", pady=3)
        lbl(r2, "CARRIER").pack(side="left")
        self.e_carrier = entry(r2, 16)
        self.e_carrier.insert(0, "4PDK2966")
        self.e_carrier.pack(side="left", padx=(4, 20))
        lbl(r2, "TEXT 검색 (선택)").pack(side="left")
        self.e_text = entry(r2, 24)
        self.e_text.pack(side="left", padx=(4, 20))
        self.btn_search = tk.Button(
            r2, text="  ▶  조회  ", bg="#89b4fa", fg="#1e1e2e",
            font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=4,
            cursor="hand2", command=self._search_click)
        self.btn_search.pack(side="left", padx=6)
        tk.Button(r2, text="초기화", bg="#45475a", fg="#cdd6f4",
                  font=("Segoe UI", 9), relief="flat", padx=8, pady=4,
                  cursor="hand2", command=self._reset).pack(side="left", padx=2)

        # 상태바
        self.status_var = tk.StringVar(value="대기 중...")
        status_bar = tk.Frame(self, bg="#181825", pady=4)
        status_bar.pack(fill="x")
        self.lbl_status = tk.Label(status_bar, textvariable=self.status_var,
                                   bg="#181825", fg="#a6e3a1",
                                   font=("Segoe UI", 9), anchor="w")
        self.lbl_status.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=120)
        self.progress.pack(side="left", padx=8)

        # 테이블
        self._build_table()

        # ── 페이지네이션 바 ──────────────────────
        self.page_bar = tk.Frame(self, bg="#181825", pady=5)
        self.page_bar.pack(fill="x")

        self.btn_prev = tk.Button(self.page_bar, text="◀ 이전", bg="#45475a", fg="#cdd6f4",
                                   relief="flat", padx=10, pady=3,
                                   font=("Segoe UI", 9), cursor="hand2",
                                   command=self._prev_page)
        self.btn_prev.pack(side="left", padx=(12, 4))

        self.page_btn_frame = tk.Frame(self.page_bar, bg="#181825")
        self.page_btn_frame.pack(side="left")

        self.btn_next = tk.Button(self.page_bar, text="다음 ▶", bg="#45475a", fg="#cdd6f4",
                                   relief="flat", padx=10, pady=3,
                                   font=("Segoe UI", 9), cursor="hand2",
                                   command=self._next_page)
        self.btn_next.pack(side="left", padx=4)

        self.lbl_page_info = tk.Label(self.page_bar, text="", bg="#181825", fg="#6c7086",
                                       font=("Segoe UI", 9))
        self.lbl_page_info.pack(side="left", padx=12)

        # 하단 쿼리 미리보기
        bot = tk.Frame(self, bg="#0f0f1a")
        bot.pack(fill="x", side="bottom")
        tk.Label(bot, text="쿼리:", bg="#0f0f1a", fg="#6c7086",
                 font=("Segoe UI", 8)).pack(side="left", padx=8)
        self.lbl_query = tk.Label(bot, text="", bg="#0f0f1a", fg="#585b70",
                                  font=("Consolas", 8), anchor="w")
        self.lbl_query.pack(side="left", fill="x", expand=True, padx=4, pady=3)

    def _build_table(self):
        frame = tk.Frame(self, bg="#1e1e2e")
        frame.pack(fill="both", expand=True, padx=12, pady=(6, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background="#181825", foreground="#cdd6f4",
                         fieldbackground="#181825", rowheight=22,
                         font=("Consolas", 9))
        style.configure("Dark.Treeview.Heading",
                         background="#313244", foreground="#89b4fa",
                         font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", "#45475a")],
                  foreground=[("selected", "#cdd6f4")])

        cols = list(COL_WIDTHS.keys())
        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  style="Dark.Treeview", selectmode="browse")
        for col in cols:
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_col(c, False))
            self.tree.column(col, width=COL_WIDTHS[col], minwidth=40,
                             anchor="w" if col in ("TEXT", "MACHINENAME") else "center")

        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self._show_detail)
        for lvl, color in LEVEL_COLORS.items():
            self.tree.tag_configure(lvl, background=color, foreground="#1e1e2e")

    # ── 페이지네이션 ──────────────────────────
    def _total_pages(self):
        if self.df_result is None or len(self.df_result) == 0:
            return 1
        import math
        return math.ceil(len(self.df_result) / PAGE_SIZE)

    def _render_page(self, page):
        if self.df_result is None:
            return
        total = self._total_pages()
        page = max(0, min(page, total - 1))
        self.current_page = page

        # 테이블 갱신
        for row in self.tree.get_children():
            self.tree.delete(row)

        cols = list(COL_WIDTHS.keys())
        start = page * PAGE_SIZE
        end   = start + PAGE_SIZE
        chunk = self.df_result.iloc[start:end]

        for _, row in chunk.iterrows():
            vals = [str(row.get(c, "")) if not pd.isna(row.get(c, float("nan"))) else ""
                    for c in cols]
            lvl = str(row.get("LEVEL", "")).strip().upper()
            tag = lvl if lvl in LEVEL_COLORS else ""
            self.tree.insert("", "end", values=vals, tags=(tag,))

        # 페이지 번호 버튼 갱신
        for w in self.page_btn_frame.winfo_children():
            w.destroy()

        # 표시할 페이지 범위 (현재 ±4, 최대 9개)
        btn_start = max(0, page - 4)
        btn_end   = min(total, btn_start + 9)
        if btn_end - btn_start < 9:
            btn_start = max(0, btn_end - 9)

        for p in range(btn_start, btn_end):
            is_cur = (p == page)
            btn = tk.Button(
                self.page_btn_frame,
                text=str(p + 1),
                bg="#89b4fa" if is_cur else "#313244",
                fg="#1e1e2e" if is_cur else "#cdd6f4",
                font=("Segoe UI", 9, "bold" if is_cur else "normal"),
                relief="flat", padx=8, pady=2, cursor="hand2",
                command=lambda pp=p: self._render_page(pp)
            )
            btn.pack(side="left", padx=1)

        # 이전/다음 버튼 상태
        self.btn_prev.config(state="normal" if page > 0 else "disabled")
        self.btn_next.config(state="normal" if page < total - 1 else "disabled")

        # 페이지 정보
        self.lbl_page_info.config(
            text=f"{page+1} / {total} 페이지  │  전체 {len(self.df_result)}건  │  {start+1}~{min(end, len(self.df_result))}번"
        )

    def _prev_page(self):
        self._render_page(self.current_page - 1)

    def _next_page(self):
        self._render_page(self.current_page + 1)

    # ── 이벤트 핸들러 ─────────────────────────
    def _set_date(self, days_ago):
        d = datetime.now() - timedelta(days=days_ago)
        self.e_from.delete(0, "end"); self.e_from.insert(0, d.strftime("%Y%m%d000000"))
        self.e_to.delete(0, "end");   self.e_to.insert(0, d.strftime("%Y%m%d235959"))

    def _reset(self):
        self.e_carrier.delete(0, "end")
        self.e_text.delete(0, "end")
        self.status_var.set("초기화 완료")
        self.lbl_query.config(text="")
        self.df_result = None
        for row in self.tree.get_children():
            self.tree.delete(row)
        for w in self.page_btn_frame.winfo_children():
            w.destroy()
        self.lbl_page_info.config(text="")

    def _search_click(self):
        q = build_query(self.e_from.get().strip(),
                        self.e_to.get().strip(),
                        self.e_carrier.get().strip(),
                        self.e_text.get().strip())
        if q is None:
            return
        self.lbl_query.config(text=q[:180] + ("..." if len(q) > 180 else ""))
        self.btn_search.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("조회 중...")
        self.lbl_status.config(fg="#f38ba8")
        threading.Thread(target=self._do_query, args=(q,), daemon=True).start()

    def _do_query(self, q):
        try:
            df = run_query(q)
            self.df_result = df
            self.after(0, lambda: self._on_load(df))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _on_load(self, df):
        self.progress.stop()
        self.btn_search.config(state="normal")
        self.status_var.set(f"✅  {len(df)}건 조회 완료  │  {datetime.now().strftime('%H:%M:%S')}")
        self.lbl_status.config(fg="#a6e3a1")
        self.current_page = 0
        self._render_page(0)

    def _on_error(self, msg):
        self.progress.stop()
        self.btn_search.config(state="normal")
        self.status_var.set(f"❌  오류: {msg[:80]}")
        self.lbl_status.config(fg="#f38ba8")
        messagebox.showerror("조회 오류", msg)

    def _sort_col(self, col, reverse):
        if self.df_result is None:
            return
        try:
            self.df_result = self.df_result.sort_values(
                by=col, ascending=not reverse,
                key=lambda x: x.astype(str)
            ).reset_index(drop=True)
            self._render_page(self.current_page)
            # 다음 클릭 시 반대 방향
            self.tree.heading(col, command=lambda: self._sort_col(col, not reverse))
        except Exception:
            pass

    def _show_detail(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        cols = list(COL_WIDTHS.keys())

        win = tk.Toplevel(self)
        win.title("상세 보기")
        win.geometry("700x600")
        win.configure(bg="#1e1e2e")
        tk.Label(win, text="📋  Row Detail", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 12, "bold")).pack(pady=8)

        txt_frame = tk.Frame(win, bg="#1e1e2e")
        txt_frame.pack(fill="both", expand=True, padx=12, pady=4)
        text = tk.Text(txt_frame, bg="#181825", fg="#cdd6f4",
                       font=("Consolas", 10), wrap="word",
                       relief="flat", padx=10, pady=8)
        sb = ttk.Scrollbar(txt_frame, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        for col, val in zip(cols, vals):
            text.insert("end", f"{col:20s}", "key")
            text.insert("end", f"  {val}\n", "val")
        text.tag_configure("key", foreground="#89b4fa", font=("Consolas", 10, "bold"))
        text.tag_configure("val", foreground="#cdd6f4")
        text.config(state="disabled")
        tk.Button(win, text="닫기", bg="#45475a", fg="#cdd6f4",
                  relief="flat", padx=16, pady=6,
                  command=win.destroy).pack(pady=8)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()