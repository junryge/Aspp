#!/usr/bin/env python3
"""
OHT Layout Viewer - Standalone EXE Version
SK Hynix M14 OHT System Layout Visualization

PyInstaller 빌드:
  pip install pyinstaller
  pyinstaller --onefile --windowed --name "OHT_Layout_Viewer" oht_viewer_exe.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import xml.etree.ElementTree as ET
import re
import math

class OHTLayoutViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("OHT Layout Viewer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0e14')
        
        # Data
        self.addresses = {}
        self.stations = {}
        self.connections = []
        self.bounds = {'minX': 0, 'maxX': 1000, 'minY': 0, 'maxY': 1000}
        
        # View state
        self.scale = 0.5
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        
        # Options
        self.show_rails = tk.BooleanVar(value=True)
        self.show_stations = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.show_welcome()
    
    def setup_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg='#111820', height=40)
        toolbar.pack(fill='x', side='top')
        toolbar.pack_propagate(False)
        
        # Title
        tk.Label(toolbar, text="OHT Layout Viewer", fg='#00d4ff', bg='#111820', 
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=10)
        
        # Buttons
        btn_style = {'bg': '#21262d', 'fg': '#e6edf3', 'relief': 'flat', 
                     'padx': 10, 'pady': 3, 'font': ('Segoe UI', 9)}
        
        tk.Button(toolbar, text="📂 Open", command=self.open_file, **btn_style).pack(side='left', padx=5, pady=5)
        tk.Button(toolbar, text="Fit", command=self.fit_view, **btn_style).pack(side='left', padx=2, pady=5)
        tk.Button(toolbar, text="Reset", command=self.reset_view, **btn_style).pack(side='left', padx=2, pady=5)
        
        # Checkboxes
        tk.Checkbutton(toolbar, text="Rails", variable=self.show_rails, 
                       command=self.render, bg='#111820', fg='#8b949e',
                       selectcolor='#21262d', activebackground='#111820').pack(side='left', padx=5)
        tk.Checkbutton(toolbar, text="Stations", variable=self.show_stations,
                       command=self.render, bg='#111820', fg='#8b949e',
                       selectcolor='#21262d', activebackground='#111820').pack(side='left', padx=5)
        
        # Search
        tk.Label(toolbar, text="Search:", fg='#8b949e', bg='#111820').pack(side='left', padx=(20,5))
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(toolbar, textvariable=self.search_var, width=15,
                                bg='#0a0e14', fg='#e6edf3', insertbackground='#e6edf3',
                                relief='flat', font=('Segoe UI', 9))
        search_entry.pack(side='left', padx=2, pady=5)
        search_entry.bind('<Return>', lambda e: self.search())
        tk.Button(toolbar, text="Find", command=self.search, **btn_style).pack(side='left', padx=5, pady=5)
        
        # Stats
        self.stats_label = tk.Label(toolbar, text="", fg='#8b949e', bg='#111820', font=('Segoe UI', 9))
        self.stats_label.pack(side='right', padx=10)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg='#0a0e14', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        # Canvas events
        self.canvas.bind('<ButtonPress-1>', self.on_drag_start)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_drag_end)
        self.canvas.bind('<MouseWheel>', self.on_scroll)  # Windows
        self.canvas.bind('<Button-4>', self.on_scroll)     # Linux up
        self.canvas.bind('<Button-5>', self.on_scroll)     # Linux down
        self.canvas.bind('<Configure>', lambda e: self.render())
        
        # Info panel
        self.info_frame = tk.Frame(self.root, bg='#111820')
        self.info_frame.place(x=10, y=50, width=160, height=80)
        tk.Label(self.info_frame, text="Info", fg='#00d4ff', bg='#111820', 
                 font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=8, pady=5)
        self.info_label = tk.Label(self.info_frame, text="드래그: 이동\n휠: 줌", 
                                   fg='#8b949e', bg='#111820', font=('Segoe UI', 8),
                                   justify='left')
        self.info_label.pack(anchor='w', padx=8)
    
    def show_welcome(self):
        self.canvas.delete('all')
        w = self.canvas.winfo_width() or 800
        h = self.canvas.winfo_height() or 600
        
        self.canvas.create_text(w//2, h//2 - 30, text="📂 OHT Layout Viewer",
                                fill='#00d4ff', font=('Segoe UI', 24, 'bold'))
        self.canvas.create_text(w//2, h//2 + 20, text="Open 버튼을 클릭하여 XML 파일을 선택하세요",
                                fill='#8b949e', font=('Segoe UI', 12))
    
    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="OHT Layout XML 파일 선택",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
        )
        if file_path:
            self.load_xml(file_path)
    
    def load_xml(self, file_path):
        try:
            self.root.config(cursor='wait')
            self.root.update()
            
            self.parse_xml(file_path)
            
            self.stats_label.config(
                text=f"Addr: {len(self.addresses):,} | Stn: {len(self.stations):,} | Conn: {len(self.connections):,}"
            )
            
            self.fit_view()
            self.root.config(cursor='')
            
        except Exception as e:
            self.root.config(cursor='')
            messagebox.showerror("Error", f"파일 로드 실패:\n{str(e)}")
    
    def parse_xml(self, file_path):
        self.addresses = {}
        self.stations = {}
        self.connections = []
        
        context = ET.iterparse(file_path, events=('start', 'end'))
        
        current_addr = None
        current_addr_no = None
        current_station = None
        current_station_no = None
        current_next_addr = None
        
        addr_data = {}
        station_data = {}
        next_addr_data = {}
        
        for event, elem in context:
            if event == 'start':
                if elem.tag == 'group':
                    name = elem.get('name', '')
                    cls = elem.get('class', '')
                    
                    if 'Addr' in name and 'address.Addr' in cls and 'NextAddr' not in name:
                        match = re.search(r'Addr(\d+)', name)
                        if match:
                            current_addr_no = int(match.group(1))
                            addr_data = {'x': 0, 'y': 0, 'stations': [], 'next_addrs': []}
                            current_addr = True
                    
                    elif 'Station' in name and 'Station' in cls:
                        match = re.search(r'Station(\d+)', name)
                        if match:
                            current_station_no = int(match.group(1))
                            station_data = {'port_id': '', 'type': 0}
                            current_station = True
                    
                    elif 'NextAddr' in name and 'NextAddr' in cls:
                        current_next_addr = True
                        next_addr_data = {'next_address': None}
            
            elif event == 'end':
                if elem.tag == 'param':
                    key = elem.get('key', '')
                    value = elem.get('value', '')
                    
                    if current_addr and not current_station and not current_next_addr:
                        if key == 'draw-x':
                            try: addr_data['x'] = float(value)
                            except: pass
                        elif key == 'draw-y':
                            try: addr_data['y'] = float(value)
                            except: pass
                    
                    elif current_station:
                        if key == 'port-id':
                            station_data['port_id'] = value
                        elif key == 'type':
                            try: station_data['type'] = int(value)
                            except: pass
                    
                    elif current_next_addr:
                        if key == 'next-address':
                            try: next_addr_data['next_address'] = int(value)
                            except: pass
                
                elif elem.tag == 'group':
                    name = elem.get('name', '')
                    
                    if current_next_addr and 'NextAddr' in name:
                        if next_addr_data.get('next_address') is not None:
                            addr_data['next_addrs'].append(next_addr_data['next_address'])
                        current_next_addr = False
                    
                    elif current_station and 'Station' in name:
                        if current_station_no:
                            self.stations[current_station_no] = station_data.copy()
                            if current_addr_no:
                                addr_data['stations'].append(current_station_no)
                        current_station = False
                        current_station_no = None
                    
                    elif current_addr and 'Addr' in name and 'NextAddr' not in name:
                        if current_addr_no is not None:
                            self.addresses[current_addr_no] = addr_data.copy()
                            for next_addr in addr_data['next_addrs']:
                                self.connections.append((current_addr_no, next_addr))
                        current_addr = False
                        current_addr_no = None
                
                elem.clear()
        
        # Calculate bounds
        xs = [a['x'] for a in self.addresses.values() if a['x'] != 0]
        ys = [a['y'] for a in self.addresses.values() if a['y'] != 0]
        
        if xs and ys:
            self.bounds = {
                'minX': min(xs), 'maxX': max(xs),
                'minY': min(ys), 'maxY': max(ys)
            }
    
    def world_to_screen(self, wx, wy):
        h = self.canvas.winfo_height()
        x = (wx - self.bounds['minX']) * self.scale + self.offset_x
        y = h - ((wy - self.bounds['minY']) * self.scale + self.offset_y)
        return x, y
    
    def screen_to_world(self, sx, sy):
        h = self.canvas.winfo_height()
        wx = (sx - self.offset_x) / self.scale + self.bounds['minX']
        wy = (h - sy - self.offset_y) / self.scale + self.bounds['minY']
        return wx, wy
    
    def render(self):
        self.canvas.delete('all')
        
        if not self.addresses:
            self.show_welcome()
            return
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Draw rails
        if self.show_rails.get():
            for from_addr, to_addr in self.connections:
                if from_addr in self.addresses and to_addr in self.addresses:
                    a = self.addresses[from_addr]
                    b = self.addresses[to_addr]
                    x1, y1 = self.world_to_screen(a['x'], a['y'])
                    x2, y2 = self.world_to_screen(b['x'], b['y'])
                    
                    if (x1 > -100 and x1 < w + 100 and y1 > -100 and y1 < h + 100) or \
                       (x2 > -100 and x2 < w + 100 and y2 > -100 and y2 < h + 100):
                        self.canvas.create_line(x1, y1, x2, y2, fill='#238636', width=max(1, self.scale * 0.5))
        
        # Draw stations
        if self.show_stations.get():
            r = max(2, self.scale * 0.8)
            for addr in self.addresses.values():
                if addr['stations']:
                    x, y = self.world_to_screen(addr['x'], addr['y'])
                    if x > -20 and x < w + 20 and y > -20 and y < h + 20:
                        for sno in addr['stations']:
                            if sno in self.stations:
                                st = self.stations[sno]
                                color = '#ff6b6b' if 5 <= st['type'] <= 9 else '#ffd93d'
                                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='')
    
    def fit_view(self):
        if not self.addresses:
            return
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        lw = self.bounds['maxX'] - self.bounds['minX']
        lh = self.bounds['maxY'] - self.bounds['minY']
        
        if lw > 0 and lh > 0:
            self.scale = min((w - 100) / lw, (h - 100) / lh)
            self.offset_x = (w - lw * self.scale) / 2
            self.offset_y = (h - lh * self.scale) / 2
            self.render()
    
    def reset_view(self):
        self.scale = 0.5
        self.offset_x = 0
        self.offset_y = 0
        self.render()
    
    def search(self):
        q = self.search_var.get().strip().upper()
        if not q:
            return
        
        for sno, st in self.stations.items():
            if st['port_id'] and q in st['port_id'].upper():
                for ano, addr in self.addresses.items():
                    if sno in addr['stations']:
                        w = self.canvas.winfo_width()
                        h = self.canvas.winfo_height()
                        x, y = self.world_to_screen(addr['x'], addr['y'])
                        self.offset_x += w / 2 - x
                        self.offset_y += h / 2 - (h - y)
                        self.info_label.config(text=f"{st['port_id']}\nNo: {sno}\nType: {st['type']}")
                        self.render()
                        return
        
        self.info_label.config(text="검색 결과 없음")
    
    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)
    
    def on_drag(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y -= dy
            self.drag_start = (event.x, event.y)
            self.render()
    
    def on_drag_end(self, event):
        self.drag_start = None
    
    def on_scroll(self, event):
        # Windows: event.delta, Linux: event.num
        if event.delta:
            zoom = 1.1 if event.delta > 0 else 0.9
        else:
            zoom = 0.9 if event.num == 5 else 1.1
        
        mx, my = event.x, event.y
        wx, wy = self.screen_to_world(mx, my)
        
        self.scale *= zoom
        self.scale = max(0.05, min(10, self.scale))
        
        sx, sy = self.world_to_screen(wx, wy)
        h = self.canvas.winfo_height()
        self.offset_x += mx - sx
        self.offset_y -= my - (h - sy)
        
        self.render()


def main():
    root = tk.Tk()
    app = OHTLayoutViewer(root)
    root.mainloop()


if __name__ == '__main__':
    main()