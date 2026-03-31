#!/usr/bin/env python3
"""
OHT 시뮬레이터 히트맵 패치
===========================
원본 파일에 히트맵 기능을 추가하기 위한 패치 코드.

적용 방법:
1. 원본 .py 파일에서 아래 3곳을 수정
2. 또는 이 스크립트를 실행하여 자동 패치: python heatmap_patch.py <원본파일경로>

변경사항:
  [서버] get_state()에 edgeHeatmap 데이터 추가
  [HTML] 헤더에 히트맵 드롭다운 UI 추가
  [JS]   히트맵 모드별 렌더링 로직 추가
"""

import sys
import re

def apply_patch(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # ============================================================
    # 패치 1: get_state()에 edgeHeatmap 데이터 추가
    # 찾을 위치: return { 'timestamp': ... } 블록 안에 'zoneStatusMap' 바로 뒤
    # ============================================================
    old_zone_status_map = """            # Zone별 현재 상태 (렌더링용)
            'zoneStatusMap': {
                z.zoneId: {
                    'status': z.status,
                    'vehicleCount': z.vehicleCount,
                    'occupancyRate': round(z.occupancyRate, 1)
                }
                for z in self.hid_zones.values()
            }
        }"""

    new_zone_status_map = """            # Zone별 현재 상태 (렌더링용)
            'zoneStatusMap': {
                z.zoneId: {
                    'status': z.status,
                    'vehicleCount': z.vehicleCount,
                    'occupancyRate': round(z.occupancyRate, 1)
                }
                for z in self.hid_zones.values()
            },
            # 히트맵용 엣지 데이터
            'edgeHeatmap': self._get_edge_heatmap_data()
        }"""

    if old_zone_status_map in content:
        content = content.replace(old_zone_status_map, new_zone_status_map)
        print("[패치1] get_state() return에 edgeHeatmap 추가 ✓")
    else:
        print("[패치1] ⚠ get_state() return 블록을 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 2: _get_edge_heatmap_data() 메서드 추가
    # 찾을 위치: generate_udp_message() 메서드 바로 위
    # ============================================================
    heatmap_method = '''
    def _get_edge_heatmap_data(self) -> list:
        """히트맵용 엣지별 데이터 반환 (프론트엔드 렌더링용)"""
        edge_data = []
        for (from_n, to_n), buf in self.rail_buffer.items():
            edge_id = f"{FAB_ID}:RE:{MCP_NAME}:{from_n}-{to_n}"
            rail_edge = self.rail_edge_map.get(edge_id)

            pass_cnt = buf['pass_cnt']
            avg_vel = 0
            if buf['velocities']:
                avg_vel = sum(buf['velocities']) / len(buf['velocities'])

            density = 0
            vhl_count = 0
            if rail_edge:
                density = rail_edge.getDensity()
                vhl_count = len(rail_edge.vhlIdMap)

            # 데이터가 있는 엣지만 전송 (대역폭 절약)
            if pass_cnt > 0 or vhl_count > 0:
                edge_data.append({
                    'f': from_n,          # fromNode
                    't': to_n,            # toNode
                    'p': pass_cnt,        # passCount (통과량)
                    'd': round(density, 1),  # density (밀도 %)
                    'v': round(avg_vel, 1),  # avgVelocity (평균속도)
                    'c': vhl_count        # vhlCount (현재 차량수)
                })

        return edge_data

'''
    # generate_udp_message 바로 앞에 삽입
    marker = "    def generate_udp_message(self, v: Vehicle) -> str:"
    if marker in content:
        content = content.replace(marker, heatmap_method + marker)
        print("[패치2] _get_edge_heatmap_data() 메서드 추가 ✓")
    else:
        print("[패치2] ⚠ generate_udp_message() 를 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 3: HTML 헤더에 히트맵 드롭다운 추가
    # 찾을 위치: 3D 효과 버튼과 곡선 버튼 사이 또는 뒤
    # ============================================================
    old_header_buttons = """            <button id="btnToggleCurve" style="padding:6px 14px;background:#00d4ff;color:#000;border:none;border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;">곡선 ON</button>"""

    new_header_buttons = """            <button id="btnToggleCurve" style="padding:6px 14px;background:#00d4ff;color:#000;border:none;border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;">곡선 ON</button>
            <select id="heatmapMode" style="padding:6px 10px;background:#1a1a3e;color:#ff6600;border:1px solid #ff6600;border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;" title="히트맵 모드">
                <option value="off">히트맵 OFF</option>
                <option value="traffic">통과량</option>
                <option value="density">밀도</option>
                <option value="velocity">속도(저속=빨강)</option>
                <option value="gaussian">OHT 밀집도</option>
            </select>"""

    if old_header_buttons in content:
        content = content.replace(old_header_buttons, new_header_buttons)
        print("[패치3] 헤더에 히트맵 드롭다운 추가 ✓")
    else:
        print("[패치3] ⚠ 헤더 버튼 영역을 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 4: 사이드바 범례에 히트맵 범례 추가
    # 찾을 위치: #sidebar 안 '범례' 섹션 뒤
    # ============================================================
    old_legend_end = """            <div class="legend-item"><div class="legend-dot" style="background:#ff0000;animation:blink 0.5s infinite"></div>정체 (JAM)</div>
        </div>
    </div>"""

    new_legend_end = """            <div class="legend-item"><div class="legend-dot" style="background:#ff0000;animation:blink 0.5s infinite"></div>정체 (JAM)</div>
        </div>
        <!-- 히트맵 범례 -->
        <div id="heatmapLegend" style="display:none;margin-top:12px;padding-top:10px;border-top:1px solid #333;">
            <div style="font-size:11px;color:#ff6600;font-weight:bold;margin-bottom:6px;">히트맵 범례</div>
            <div id="heatmapLegendContent" style="font-size:10px;color:#aaa;"></div>
            <div style="margin-top:6px;height:12px;border-radius:3px;background:linear-gradient(90deg, #00ff88, #ffff00, #ff6600, #ff0000);"></div>
            <div style="display:flex;justify-content:space-between;font-size:9px;color:#888;margin-top:2px;">
                <span id="heatLegendLow">낮음</span>
                <span id="heatLegendHigh">높음</span>
            </div>
        </div>
    </div>"""

    if old_legend_end in content:
        content = content.replace(old_legend_end, new_legend_end)
        print("[패치4] 사이드바 히트맵 범례 추가 ✓")
    else:
        print("[패치4] ⚠ 범례 영역을 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 5: JS - WebSocket update에서 edgeHeatmap 데이터 수신
    # 찾을 위치: window.zoneStatusMap = msg.data.zoneStatusMap || {};
    # ============================================================
    old_zone_status_js = "        // Zone 상태 맵 저장 (렌더링용)\n        window.zoneStatusMap = msg.data.zoneStatusMap || {};"

    new_zone_status_js = """        // Zone 상태 맵 저장 (렌더링용)
        window.zoneStatusMap = msg.data.zoneStatusMap || {};

        // 히트맵 데이터 저장 (렌더링용)
        if (msg.data.edgeHeatmap) {
            window.edgeHeatmapData = {};
            msg.data.edgeHeatmap.forEach(e => {
                const key = e.f + '-' + e.t;
                window.edgeHeatmapData[key] = e;
            });
        }"""

    if old_zone_status_js in content:
        content = content.replace(old_zone_status_js, new_zone_status_js)
        print("[패치5] WebSocket에서 edgeHeatmap 수신 코드 추가 ✓")
    else:
        print("[패치5] ⚠ zoneStatusMap JS 코드를 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 6: JS - 히트맵 모드 변수 및 드롭다운 이벤트
    # 찾을 위치: let curveRadius = 30; 바로 앞
    # ============================================================
    old_curve_radius = "let curveRadius = 30;  // 곡률반경 (픽셀)"

    new_curve_radius = """// ============================================================
// 히트맵 모드 설정
// ============================================================
let heatmapMode = 'off';  // off, traffic, density, velocity, gaussian
window.edgeHeatmapData = {};

// 히트맵 색상 보간 함수 (0~1 → 초록~빨강)
function heatColor(ratio) {
    ratio = Math.max(0, Math.min(1, ratio));
    // 초록(0) → 노랑(0.33) → 주황(0.66) → 빨강(1)
    let r, g, b;
    if (ratio < 0.33) {
        const t = ratio / 0.33;
        r = Math.round(t * 255);
        g = 255;
        b = Math.round(136 * (1 - t));
    } else if (ratio < 0.66) {
        const t = (ratio - 0.33) / 0.33;
        r = 255;
        g = Math.round(255 * (1 - t * 0.4));
        b = 0;
    } else {
        const t = (ratio - 0.66) / 0.34;
        r = 255;
        g = Math.round(153 * (1 - t));
        b = 0;
    }
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

// 히트맵 드롭다운 이벤트
document.getElementById('heatmapMode').addEventListener('change', function(e) {
    heatmapMode = e.target.value;
    const legend = document.getElementById('heatmapLegend');
    const legendContent = document.getElementById('heatmapLegendContent');
    const lowLabel = document.getElementById('heatLegendLow');
    const highLabel = document.getElementById('heatLegendHigh');

    if (heatmapMode === 'off') {
        legend.style.display = 'none';
    } else {
        legend.style.display = 'block';
        switch(heatmapMode) {
            case 'traffic':
                legendContent.textContent = '레일 통과 횟수 (많을수록 빨강)';
                lowLabel.textContent = '0회';
                highLabel.textContent = '최다';
                break;
            case 'density':
                legendContent.textContent = '엣지 차량 밀도 (밀집할수록 빨강)';
                lowLabel.textContent = '0%';
                highLabel.textContent = '100%';
                break;
            case 'velocity':
                legendContent.textContent = '평균 속도 (느릴수록 빨강)';
                lowLabel.textContent = '300 m/min';
                highLabel.textContent = '0 m/min';
                break;
            case 'gaussian':
                legendContent.textContent = 'OHT 위치 기반 밀집도 (가우시안)';
                lowLabel.textContent = '소수';
                highLabel.textContent = '밀집';
                break;
        }
    }
    render();
});

let curveRadius = 30;  // 곡률반경 (픽셀)"""

    if old_curve_radius in content:
        content = content.replace(old_curve_radius, new_curve_radius)
        print("[패치6] 히트맵 모드 변수 및 이벤트 추가 ✓")
    else:
        print("[패치6] ⚠ curveRadius 변수를 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 7: JS - render() 함수에 히트맵 렌더링 로직 추가
    # drawCurvedRails() 호출 부분을 히트맵 모드 분기로 교체
    # ============================================================
    old_render_rails = """    // 엣지 (레일) - 곡선 처리 적용
    drawCurvedRails();"""

    new_render_rails = """    // 엣지 (레일) - 히트맵 모드에 따라 다르게 렌더링
    if (heatmapMode === 'off') {
        drawCurvedRails();
    } else if (heatmapMode === 'gaussian') {
        // 가우시안 모드: 기본 레일 먼저 그리고, OHT 루프 뒤에 오버레이
        drawCurvedRails();
    } else {
        // 엣지 기반 히트맵 (traffic, density, velocity)
        drawHeatmapRails();
    }"""

    if old_render_rails in content:
        content = content.replace(old_render_rails, new_render_rails)
        print("[패치7] render()에 히트맵 분기 추가 ✓")
    else:
        print("[패치7] ⚠ render() 레일 그리기 부분을 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 8: JS - drawHeatmapRails() 함수 + 가우시안 오버레이 함수 추가
    # 찾을 위치: drawCurvedRails() 함수 바로 앞에 삽입
    # ============================================================
    heatmap_functions = """
// ============================================================
// 히트맵 레일 렌더링 (엣지 색상을 데이터에 따라 변경)
// ============================================================
function drawHeatmapRails() {
    if (!layout || !layout.edges) return;

    const heatData = window.edgeHeatmapData || {};

    // 최대값 계산 (정규화용)
    let maxVal = 1;
    if (heatmapMode === 'traffic') {
        Object.values(heatData).forEach(e => { if (e.p > maxVal) maxVal = e.p; });
    } else if (heatmapMode === 'density') {
        maxVal = 100;  // 밀도는 0~100%
    } else if (heatmapMode === 'velocity') {
        maxVal = 300;  // 최대 300 m/min
    }

    layout.edges.forEach(e => {
        const from = nodeMap[e.from], to = nodeMap[e.to];
        if (!from || !to) return;

        const key = e.from + '-' + e.to;
        const data = heatData[key];
        let ratio = 0;
        let hasData = false;

        if (data) {
            hasData = true;
            switch(heatmapMode) {
                case 'traffic':
                    ratio = data.p / maxVal;
                    break;
                case 'density':
                    ratio = data.d / 100;
                    break;
                case 'velocity':
                    // 속도: 빠르면 초록, 느리면 빨강 (반전)
                    ratio = data.v > 0 ? 1 - (data.v / maxVal) : 0;
                    break;
            }
        }

        // 데이터 없으면 기본 어두운 색
        const edgeColor = hasData ? heatColor(ratio) : '#1a2a3a';
        const lineWidth = hasData ? Math.max(2, 1.5 + ratio * 4) / scale : 1 / scale;

        if (isPseudo3D) {
            draw3DRail(from.x, from.y, to.x, to.y, edgeColor);
        } else {
            ctx.strokeStyle = edgeColor;
            ctx.lineWidth = lineWidth;
            ctx.globalAlpha = hasData ? 0.6 + ratio * 0.4 : 0.3;
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
    });
}

// ============================================================
// 가우시안 히트맵 오버레이 (OHT 위치 기반)
// ============================================================
function drawGaussianHeatmap() {
    if (!vehicles || heatmapMode !== 'gaussian') return;

    const vehList = Object.values(vehicles);
    if (vehList.length === 0) return;

    // 오프스크린 캔버스에서 히트맵 생성
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = canvas.width;
    heatCanvas.height = canvas.height;
    const hCtx = heatCanvas.getContext('2d');

    // 각 OHT 위치에 가우시안 원 그리기
    const radius = Math.max(30, 80 / scale);  // 줌에 따라 반경 조절

    vehList.forEach(v => {
        // 화면 좌표로 변환
        let sx, sy;
        if (isPseudo3D) {
            const isoPos = toIso(v.dispX, v.dispY, RAIL_HEIGHT / scale);
            sx = isoPos.x * scale + offsetX;
            sy = isoPos.y * scale + offsetY;
        } else {
            sx = v.dispX * scale + offsetX;
            sy = v.dispY * scale + offsetY;
        }

        const screenRadius = radius * scale;

        // 가우시안 그라데이션
        const grad = hCtx.createRadialGradient(sx, sy, 0, sx, sy, screenRadius);
        grad.addColorStop(0, 'rgba(255, 0, 0, 0.15)');
        grad.addColorStop(0.4, 'rgba(255, 100, 0, 0.08)');
        grad.addColorStop(0.7, 'rgba(255, 200, 0, 0.03)');
        grad.addColorStop(1, 'rgba(0, 255, 0, 0)');

        hCtx.fillStyle = grad;
        hCtx.beginPath();
        hCtx.arc(sx, sy, screenRadius, 0, Math.PI * 2);
        hCtx.fill();
    });

    // 메인 캔버스에 오버레이 합성
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);  // 변환 초기화 (화면 좌표계)
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.7;
    ctx.drawImage(heatCanvas, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0;
    ctx.restore();
}

"""

    # drawCurvedRails 함수 앞에 삽입
    old_curved_rails_marker = "// 곡선 레일 그리기 (코너에만 모따기 적용)\nfunction drawCurvedRails() {"
    if old_curved_rails_marker in content:
        content = content.replace(old_curved_rails_marker, heatmap_functions + old_curved_rails_marker)
        print("[패치8] drawHeatmapRails() + drawGaussianHeatmap() 함수 추가 ✓")
    else:
        # 대체 마커 시도
        alt_marker = "function drawCurvedRails() {"
        if alt_marker in content:
            content = content.replace(alt_marker, heatmap_functions + alt_marker, 1)
            print("[패치8] drawHeatmapRails() + drawGaussianHeatmap() 함수 추가 ✓ (대체 마커)")
        else:
            print("[패치8] ⚠ drawCurvedRails() 함수를 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 패치 9: render() 함수 끝부분에 가우시안 오버레이 호출 추가
    # ctx.restore() 후, zoomLevel 표시 전에 삽입
    # ============================================================
    old_render_end = """    ctx.restore();

    document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
}"""

    new_render_end = """    ctx.restore();

    // 가우시안 히트맵 오버레이 (OHT 렌더링 후에 그려야 함)
    if (heatmapMode === 'gaussian') {
        drawGaussianHeatmap();
    }

    document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
}"""

    if old_render_end in content:
        content = content.replace(old_render_end, new_render_end)
        print("[패치9] render() 끝에 가우시안 오버레이 호출 추가 ✓")
    else:
        print("[패치9] ⚠ render() 끝부분을 찾을 수 없음 - 수동 패치 필요")

    # ============================================================
    # 저장
    # ============================================================
    output_path = filepath.replace('.html', '_heatmap.html').replace('.py', '_heatmap.py')
    if output_path == filepath:
        output_path = filepath + '_heatmap_patched'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ 패치 완료! 저장 위치: {output_path}")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python heatmap_patch.py <원본파일경로>")
        print("예: python heatmap_patch.py Untitled_txt__7_.html")
        sys.exit(1)

    apply_patch(sys.argv[1])
