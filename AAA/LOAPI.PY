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

# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────
HOST    = "10.40.42.27"
PORT    = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

TABLES = "ts_data_view_m14a, ts_data_view_m14b, ts_data_view_m16, ts_data_view_m16b"
FIELDS = "_time, TIME_EX, MACHINENAME, MACHINETYPE, UNITNAME, CARRIER, COMMANDID, COMMAND, OPERATION_NAME, MESSAGENAME, PROCESS, TRANSACTIONID, TEXT, THREAD, LEVEL, XML, SECSII, RESULTCODE"
LEVEL_FILTER = '(LEVEL=="WELL" or LEVEL=="WARN" or LEVEL=="ERROR" or LEVEL=="FATAL")'

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
#  API
# ─────────────────────────────────────────────
def run_query(query: str, timeout=120):
    # 공백 정규화 (줄바꿈 등 제거)
    clean_q = " ".join(query.split())
    encoded  = urllib.parse.quote(clean_q, safe="")
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    # 디버그 출력 (콘솔 확인용)
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

    # 원본 쿼리 구조 그대로 - 조건만 변수로
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
#  메인 앱
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Logpresso Carrier Viewer")
        self.geometry("1400x820")
        self.configure(bg="#1e1e2e")
        self.df_result = None
        self._build_ui()

    # ── UI 구성 ──────────────────────────────
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

        # 빠른 날짜 버튼
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

        # 하단: 쿼리 미리보기
        bot = tk.Frame(self, bg="#181825")
        bot.pack(fill="x", side="bottom")
        tk.Label(bot, text="쿼리:", bg="#181825", fg="#6c7086",
                 font=("Segoe UI", 8)).pack(side="left", padx=8)
        self.lbl_query = tk.Label(bot, text="", bg="#181825", fg="#585b70",
                                  font=("Consolas", 8), anchor="w")
        self.lbl_query.pack(side="left", fill="x", expand=True, padx=4, pady=3)

    def _build_table(self):
        frame = tk.Frame(self, bg="#1e1e2e")
        frame.pack(fill="both", expand=True, padx=12, pady=6)

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

        # 행 더블클릭 → 상세 팝업
        self.tree.bind("<Double-1>", self._show_detail)

        # 태그(LEVEL 색상)
        for lvl, color in LEVEL_COLORS.items():
            self.tree.tag_configure(lvl, background=color, foreground="#1e1e2e")

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
        for row in self.tree.get_children():
            self.tree.delete(row)

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
            self.after(0, lambda: self._load_table(df))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _load_table(self, df):
        for row in self.tree.get_children():
            self.tree.delete(row)

        cols = list(COL_WIDTHS.keys())
        for _, row in df.iterrows():
            vals = [str(row.get(c, "")) if not pd.isna(row.get(c, float("nan"))) else ""
                    for c in cols]
            lvl = str(row.get("LEVEL", "")).strip().upper()
            tag = lvl if lvl in LEVEL_COLORS else ""
            self.tree.insert("", "end", values=vals, tags=(tag,))

        self.progress.stop()
        self.btn_search.config(state="normal")
        self.status_var.set(f"✅  {len(df)}건 조회 완료  │  {datetime.now().strftime('%H:%M:%S')}")
        self.lbl_status.config(fg="#a6e3a1")

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
            data = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
            data.sort(reverse=reverse, key=lambda x: x[0])
            for i, (_, k) in enumerate(data):
                self.tree.move(k, "", i)
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