# -*- coding: utf-8 -*-
"""데모스 런처 — 매일 실행하는 서버/스크립트 원클릭 실행 UI

사용법:
    python demos_launcher.py

기능:
  * 카테고리(탭)별로 항목을 정리해서 버튼 하나로 실행 (새 CMD 창이 열려 로그 확인 가능)
  * [전체 실행] : 모든 카테고리의 모든 항목을 순서대로 실행
  * 항목 / 카테고리 추가 · 수정 · 삭제 · 순서변경 가능
  * 실행 상태(● 실행중) 표시, [중지] 버튼으로 해당 CMD 창과 하위 프로세스까지 종료
  * 모든 설정은 이 파일과 같은 폴더의 launcher_config.json 에 자동 저장
    (런처를 닫아도 이미 실행한 서버 CMD 창은 그대로 유지됨)
"""

import itertools
import json
import os
import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

APP_TITLE = "데모스 런처"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "launcher_config.json")
IS_WIN = (os.name == "nt")

DEFAULT_CONFIG = {
    "categories": [
        {
            "name": "데모스",
            "items": [
                {"name": "데모스 메인 (scientific-assistant)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\scientific-assistant",
                 "cmd": "python app.py"},
                {"name": "RAG 서버 (rag_server)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\scientific-assistant\rag_server",
                 "cmd": "python app.py"},
            ],
        },
        {
            "name": "RPA",
            "items": [
                {"name": "RPA 플로우 (run_flow.py)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\RPA",
                 "cmd": "python run_flow.py"},
            ],
        },
        {
            "name": "리포트",
            "items": [
                {"name": "레포트10060",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\레포트1000_4000\레포트10060",
                 "cmd": "python server.py"},
                {"name": "레포트10050",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\레포트1000_4000\레포트10050",
                 "cmd": "python server.py"},
                {"name": "레포트10040",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\레포트1000_4000\레포트10040",
                 "cmd": "python server.py"},
                {"name": "레포트10030",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\레포트1000_4000\레포트10030",
                 "cmd": "python server.py"},
            ],
        },
        {
            "name": "관제시스템",
            "items": [
                {"name": "아바타 2D (avatar_2d)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs\avatar_2d",
                 "cmd": "python run.py"},
                {"name": "실시간 AMHS 서버",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs",
                 "cmd": "python server.py"},
                {"name": "월드모델 파생",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs\월드모델\월드모델파생",
                 "cmd": "python main.py"},
                {"name": "QA",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs\qa",
                 "cmd": "python app.py"},
                {"name": "LLM WIKI MCP (app.py)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs\LLM_WIKI_MCP",
                 "cmd": "python app.py"},
                {"name": "LLM WIKI MCP 서버 (mcp_server.py)",
                 "dir": r"C:\연구과제\CODE\데모스_분석툴\관제시스템\real_time_amhs\LLM_WIKI_MCP",
                 "cmd": "python mcp_server.py"},
            ],
        },
    ]
}

_key_counter = itertools.count(1)


def _tag(cfg):
    """각 항목에 런타임 전용 키를 부여 (저장 안 됨)."""
    for cat in cfg.get("categories", []):
        for it in cat.get("items", []):
            it["_key"] = next(_key_counter)
    return cfg


def save_config(cfg):
    data = {"categories": [
        {"name": c["name"],
         "items": [{"name": i["name"], "dir": i["dir"], "cmd": i["cmd"]}
                   for i in c["items"]]}
        for c in cfg["categories"]]}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg.get("categories"), list):
                return _tag(cfg)
        except Exception:
            pass  # 설정이 깨졌으면 기본값으로 재생성
    cfg = _tag(json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False)))
    save_config(cfg)
    return cfg


class ItemDialog(tk.Toplevel):
    """항목 추가/수정 다이얼로그"""

    def __init__(self, parent, title, item=None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None

        frm = ttk.Frame(self, padding=14)
        frm.grid(row=0, column=0)

        ttk.Label(frm, text="이름").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 8))
        self.e_name = ttk.Entry(frm, width=54)
        self.e_name.grid(row=0, column=1, columnspan=2, sticky="we", pady=4)

        ttk.Label(frm, text="실행 폴더").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 8))
        self.e_dir = ttk.Entry(frm, width=46)
        self.e_dir.grid(row=1, column=1, sticky="we", pady=4)
        ttk.Button(frm, text="찾아보기…", command=self.browse).grid(row=1, column=2, padx=(6, 0))

        ttk.Label(frm, text="명령어").grid(row=2, column=0, sticky="w", pady=4, padx=(0, 8))
        self.e_cmd = ttk.Entry(frm, width=54)
        self.e_cmd.grid(row=2, column=1, columnspan=2, sticky="we", pady=4)
        self.e_cmd.insert(0, "python ")

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, pady=(12, 0))
        ttk.Button(btns, text="저장", width=10, command=self.ok).pack(side="left", padx=4)
        ttk.Button(btns, text="취소", width=10, command=self.destroy).pack(side="left", padx=4)

        if item:
            self.e_name.insert(0, item["name"])
            self.e_dir.insert(0, item["dir"])
            self.e_cmd.delete(0, "end")
            self.e_cmd.insert(0, item["cmd"])

        self.e_name.focus_set()
        self.bind("<Return>", lambda e: self.ok())
        self.bind("<Escape>", lambda e: self.destroy())

    def browse(self):
        d = filedialog.askdirectory(initialdir=self.e_dir.get() or "C:\\", parent=self)
        if d:
            self.e_dir.delete(0, "end")
            self.e_dir.insert(0, os.path.normpath(d))

    def ok(self):
        name = self.e_name.get().strip()
        d = self.e_dir.get().strip().strip('"')
        cmd = self.e_cmd.get().strip()
        if not name or not d or not cmd:
            messagebox.showwarning(APP_TITLE, "이름 / 실행 폴더 / 명령어를 모두 입력하세요.", parent=self)
            return
        self.result = {"name": name, "dir": d, "cmd": cmd}
        self.destroy()


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.procs = {}          # item _key -> Popen
        self.status_labels = {}  # item _key -> 상태 Label
        self._tab_frames = []

        root.title(APP_TITLE)
        root.geometry("900x640")
        root.minsize(760, 480)

        if IS_WIN:
            for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    tkfont.nametofont(fname).configure(family="맑은 고딕", size=10)
                except Exception:
                    pass

        base = tkfont.nametofont("TkDefaultFont")
        self.bold_font = tkfont.Font(**base.actual())
        self.bold_font.configure(weight="bold")
        self.title_font = tkfont.Font(**base.actual())
        self.title_font.configure(size=13, weight="bold")

        top = ttk.Frame(root, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text="데모스 런처", font=self.title_font).pack(side="left")
        ttk.Button(top, text="＋ 카테고리 추가", command=self.add_category).pack(side="right", padx=3)
        ttk.Button(top, text="■ 전체 중지", command=self.stop_everything).pack(side="right", padx=3)
        ttk.Button(top, text="▶ 전체 실행 (모든 카테고리)", command=self.start_everything).pack(side="right", padx=3)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self.rebuild()
        self.poll()

    # ---------- UI 구성 ----------

    def rebuild(self):
        try:
            sel = self.nb.index("current") if self.nb.tabs() else 0
        except Exception:
            sel = 0
        for fr in self._tab_frames:
            fr.destroy()
        self._tab_frames = []
        self.status_labels.clear()
        for cat in self.cfg["categories"]:
            self.nb.add(self.build_tab(cat), text=" %s " % cat["name"])
        if self.nb.tabs():
            self.nb.select(min(sel, len(self.nb.tabs()) - 1))

    def build_tab(self, cat):
        outer = ttk.Frame(self.nb, padding=8)
        self._tab_frames.append(outer)

        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="▶ 이 탭 전체 실행", command=lambda c=cat: self.start_all(c)).pack(side="left", padx=(0, 3))
        ttk.Button(bar, text="■ 이 탭 전체 중지", command=lambda c=cat: self.stop_all(c)).pack(side="left", padx=3)
        ttk.Button(bar, text="＋ 항목 추가", command=lambda c=cat: self.add_item(c)).pack(side="left", padx=3)
        ttk.Button(bar, text="카테고리 삭제", command=lambda c=cat: self.delete_category(c)).pack(side="right", padx=3)
        ttk.Button(bar, text="이름 변경", command=lambda c=cat: self.rename_category(c)).pack(side="right", padx=3)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        rows = ttk.Frame(canvas)
        rows.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
        win = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.bind("<Configure>", lambda e, c=canvas, w=win: c.itemconfigure(w, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _wheel(e, c=canvas):
            c.yview_scroll(int(-e.delta / 120) if e.delta else (1 if e.num == 5 else -1), "units")
        canvas.bind("<Enter>", lambda e, c=canvas: (c.bind_all("<MouseWheel>", _wheel),
                                                    c.bind_all("<Button-4>", _wheel),
                                                    c.bind_all("<Button-5>", _wheel)))
        canvas.bind("<Leave>", lambda e, c=canvas: (c.unbind_all("<MouseWheel>"),
                                                    c.unbind_all("<Button-4>"),
                                                    c.unbind_all("<Button-5>")))

        if not cat["items"]:
            ttk.Label(rows, text="항목이 없습니다. [＋ 항목 추가]를 눌러 등록하세요.",
                      foreground="#888").pack(pady=24)
        for idx, item in enumerate(cat["items"]):
            self.build_row(rows, cat, idx, item)
        return outer

    def build_row(self, parent, cat, idx, item):
        row = ttk.Frame(parent, padding=(6, 6))
        row.pack(fill="x")
        ttk.Separator(parent).pack(fill="x")

        st = ttk.Label(row, text="● 대기", foreground="#999999", width=8)
        st.pack(side="left")
        self.status_labels[item["_key"]] = st

        # 버튼을 먼저 배치해서 창이 좁아도 버튼이 잘리지 않게 함
        ttk.Button(row, text="✕", width=3, command=lambda c=cat, i=item: self.delete_item(c, i)).pack(side="right", padx=(6, 0))
        ttk.Button(row, text="↓", width=3, command=lambda c=cat, j=idx: self.move_item(c, j, +1)).pack(side="right", padx=1)
        ttk.Button(row, text="↑", width=3, command=lambda c=cat, j=idx: self.move_item(c, j, -1)).pack(side="right", padx=1)
        ttk.Button(row, text="✎ 수정", width=7, command=lambda c=cat, i=item: self.edit_item(c, i)).pack(side="right", padx=2)
        ttk.Button(row, text="■ 중지", width=7, command=lambda i=item: self.stop_item(i)).pack(side="right", padx=2)
        ttk.Button(row, text="▶ 실행", width=7, command=lambda i=item: self.start_item(i)).pack(side="right", padx=2)

        txt = ttk.Frame(row)
        txt.pack(side="left", fill="x", expand=True)
        name_lbl = ttk.Label(txt, text=item["name"], font=self.bold_font)
        name_lbl.pack(anchor="w")
        ttk.Label(txt, text="%s  ▸  %s" % (item["dir"], item["cmd"]),
                  foreground="#888888").pack(anchor="w")
        name_lbl.bind("<Double-Button-1>", lambda e, i=item: self.start_item(i))

    # ---------- 프로세스 제어 ----------

    def _running(self, item):
        p = self.procs.get(item["_key"])
        return p is not None and p.poll() is None

    def _launch(self, item):
        d = item["dir"]
        if not os.path.isdir(d):
            messagebox.showerror(APP_TITLE, "실행 폴더가 없습니다:\n%s" % d)
            return False
        try:
            if IS_WIN:
                safe_title = item["name"].replace('"', "'").replace("&", "-")
                inner = "title %s & %s" % (safe_title, item["cmd"])
                p = subprocess.Popen('cmd /k "%s"' % inner, cwd=d,
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                p = subprocess.Popen(item["cmd"], cwd=d, shell=True)
            self.procs[item["_key"]] = p
            return True
        except Exception as e:
            messagebox.showerror(APP_TITLE, "실행 실패: %s\n%s" % (item["name"], e))
            return False

    def start_item(self, item):
        if self._running(item):
            if not messagebox.askyesno(APP_TITLE, "'%s' 이(가) 이미 실행중입니다.\n중지 후 다시 실행할까요?" % item["name"]):
                return
            self.stop_item(item, quiet=True)
        self._launch(item)

    def stop_item(self, item, quiet=False):
        p = self.procs.get(item["_key"])
        if p is None or p.poll() is not None:
            if not quiet:
                messagebox.showinfo(APP_TITLE, "'%s' 은(는) 실행중이 아닙니다." % item["name"])
            return
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                p.terminate()
        except Exception:
            pass

    def _start_seq(self, items, delay=700):
        """항목들을 delay(ms) 간격으로 순차 실행 (동시 기동 폭주 방지)."""
        def go(i=0):
            if i < len(items):
                self._launch(items[i])
                self.root.after(delay, lambda: go(i + 1))
        go()

    def start_all(self, cat):
        items = [i for i in cat["items"] if not self._running(i)]
        if items:
            self._start_seq(items)

    def stop_all(self, cat):
        for i in cat["items"]:
            self.stop_item(i, quiet=True)

    def start_everything(self):
        items = [i for c in self.cfg["categories"] for i in c["items"] if not self._running(i)]
        if not items:
            messagebox.showinfo(APP_TITLE, "실행할 항목이 없습니다. (모두 실행중)")
            return
        self._start_seq(items)

    def stop_everything(self):
        if messagebox.askyesno(APP_TITLE, "실행중인 모든 항목을 중지할까요?"):
            for c in self.cfg["categories"]:
                self.stop_all(c)

    # ---------- 항목/카테고리 관리 ----------

    def add_item(self, cat):
        dlg = ItemDialog(self.root, "항목 추가")
        self.root.wait_window(dlg)
        if dlg.result:
            dlg.result["_key"] = next(_key_counter)
            cat["items"].append(dlg.result)
            save_config(self.cfg)
            self.rebuild()

    def edit_item(self, cat, item):
        dlg = ItemDialog(self.root, "항목 수정", item)
        self.root.wait_window(dlg)
        if dlg.result:
            item.update(dlg.result)
            save_config(self.cfg)
            self.rebuild()

    def delete_item(self, cat, item):
        msg = "'%s' 항목을 삭제할까요?" % item["name"]
        if self._running(item):
            msg = "'%s' 이(가) 실행중입니다.\n중지하고 삭제할까요?" % item["name"]
        if not messagebox.askyesno(APP_TITLE, msg):
            return
        self.stop_item(item, quiet=True)
        cat["items"].remove(item)
        save_config(self.cfg)
        self.rebuild()

    def move_item(self, cat, idx, d):
        j = idx + d
        if 0 <= j < len(cat["items"]):
            cat["items"][idx], cat["items"][j] = cat["items"][j], cat["items"][idx]
            save_config(self.cfg)
            self.rebuild()

    def add_category(self):
        name = simpledialog.askstring(APP_TITLE, "새 카테고리 이름:", parent=self.root)
        if name and name.strip():
            self.cfg["categories"].append({"name": name.strip(), "items": []})
            save_config(self.cfg)
            self.rebuild()
            self.nb.select(len(self.nb.tabs()) - 1)

    def rename_category(self, cat):
        name = simpledialog.askstring(APP_TITLE, "카테고리 이름:", initialvalue=cat["name"], parent=self.root)
        if name and name.strip():
            cat["name"] = name.strip()
            save_config(self.cfg)
            self.rebuild()

    def delete_category(self, cat):
        if len(self.cfg["categories"]) <= 1:
            messagebox.showwarning(APP_TITLE, "마지막 카테고리는 삭제할 수 없습니다.")
            return
        if not messagebox.askyesno(APP_TITLE, "'%s' 카테고리와 항목 %d개를 삭제할까요?" % (cat["name"], len(cat["items"]))):
            return
        for i in cat["items"]:
            self.stop_item(i, quiet=True)
        self.cfg["categories"].remove(cat)
        save_config(self.cfg)
        self.rebuild()

    # ---------- 상태 갱신 ----------

    def poll(self):
        for cat in self.cfg["categories"]:
            for item in cat["items"]:
                lbl = self.status_labels.get(item["_key"])
                if lbl is None or not lbl.winfo_exists():
                    continue
                if self._running(item):
                    lbl.configure(text="● 실행중", foreground="#1a9c46")
                else:
                    lbl.configure(text="● 대기", foreground="#999999")
        self.root.after(1000, self.poll)


def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
