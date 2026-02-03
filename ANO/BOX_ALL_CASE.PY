"""
모든 위치 이탈 케이스
- 상/하/좌/우
- 대각선 (좌상, 우상, 좌하, 우하)
- 회전
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

def create_topdown_box(filename, box_x, box_y, angle=0):
    """위에서 본 박스 이미지"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 컨베이어 벨트 배경 (단색)
    ax.set_facecolor('#3D3D3D')
    
    # 박스 크기
    box_w = 0.28
    box_h = 0.22
    
    # 중심점
    cx = box_x + box_w/2
    cy = box_y + box_h/2
    
    if angle != 0:
        t = Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    else:
        t = ax.transData
    
    # 박스 본체
    box = patches.Rectangle(
        (box_x, box_y), box_w, box_h,
        facecolor='#C4883A',
        edgecolor='#8B5A2B',
        linewidth=3,
        transform=t
    )
    ax.add_patch(box)
    
    # 테이프 (가로)
    tape_h = patches.Rectangle(
        (box_x + 0.02, box_y + box_h/2 - 0.015), box_w - 0.04, 0.03,
        facecolor='#D2B48C',
        edgecolor='none',
        transform=t
    )
    ax.add_patch(tape_h)
    
    # 테이프 (세로)
    tape_v = patches.Rectangle(
        (box_x + box_w/2 - 0.015, box_y + 0.02), 0.03, box_h - 0.04,
        facecolor='#D2B48C',
        edgecolor='none',
        transform=t
    )
    ax.add_patch(tape_v)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=200, bbox_inches='tight', 
                facecolor='#3D3D3D', pad_inches=0)
    plt.close()
    print(f"Saved: {filename}")


def main():
    out = '/home/claude/'
    
    # 정상 위치 (중앙)
    center_x = 0.36
    center_y = 0.39
    shift = 0.15  # 이탈 거리
    
    # === 정상 ===
    create_topdown_box(f'{out}box_01_normal.png', 
                       center_x, center_y, angle=0)
    
    # === 상하좌우 이탈 ===
    # 오른쪽
    create_topdown_box(f'{out}box_02_right.png', 
                       center_x + shift, center_y, angle=0)
    # 왼쪽
    create_topdown_box(f'{out}box_03_left.png', 
                       center_x - shift, center_y, angle=0)
    # 위
    create_topdown_box(f'{out}box_04_up.png', 
                       center_x, center_y + shift, angle=0)
    # 아래
    create_topdown_box(f'{out}box_05_down.png', 
                       center_x, center_y - shift, angle=0)
    
    # === 대각선 이탈 ===
    diag = 0.12
    # 우상
    create_topdown_box(f'{out}box_06_up_right.png', 
                       center_x + diag, center_y + diag, angle=0)
    # 좌상
    create_topdown_box(f'{out}box_07_up_left.png', 
                       center_x - diag, center_y + diag, angle=0)
    # 우하
    create_topdown_box(f'{out}box_08_down_right.png', 
                       center_x + diag, center_y - diag, angle=0)
    # 좌하
    create_topdown_box(f'{out}box_09_down_left.png', 
                       center_x - diag, center_y - diag, angle=0)
    
    # === 회전 (위치는 중앙) ===
    create_topdown_box(f'{out}box_10_rotate_10.png', 
                       center_x, center_y, angle=10)
    create_topdown_box(f'{out}box_11_rotate_20.png', 
                       center_x, center_y, angle=20)
    create_topdown_box(f'{out}box_12_rotate_45.png', 
                       center_x, center_y, angle=45)
    create_topdown_box(f'{out}box_13_rotate_m15.png', 
                       center_x, center_y, angle=-15)
    create_topdown_box(f'{out}box_14_rotate_m30.png', 
                       center_x, center_y, angle=-30)
    
    # === 이탈 + 회전 복합 ===
    create_topdown_box(f'{out}box_15_right_rotate.png', 
                       center_x + shift, center_y, angle=15)
    create_topdown_box(f'{out}box_16_left_rotate.png', 
                       center_x - shift, center_y, angle=-20)
    create_topdown_box(f'{out}box_17_up_rotate.png', 
                       center_x, center_y + shift, angle=25)
    create_topdown_box(f'{out}box_18_corner_rotate.png', 
                       center_x + diag, center_y + diag, angle=35)
    
    print(f"\nDone! 18 images created.")


if __name__ == "__main__":
    main()