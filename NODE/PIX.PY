"""
열심히 일하는 픽셀 개발자 - Pygame 애니메이션
실행: pip install pygame 후 python working_pixel.py
조작: 1=일하기, 2=휴식, R=리셋, ESC=종료
"""

import pygame
import sys
import math
import random

# ─── 초기화 ───
pygame.init()
PIXEL = 8
W, H = 56 * PIXEL, 62 * PIXEL
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("🖥️ PIXEL WORKER 🖥️")
clock = pygame.time.Clock()

# ─── 색상 팔레트 ───
C = {
    "skin":       (245, 197, 163),
    "skin_dark":  (212, 160, 118),
    "hair":       (59, 36, 23),
    "shirt":      (74, 144, 217),
    "shirt_dark": (54, 112, 176),
    "pants":      (44, 62, 80),
    "pants_dark": (26, 37, 47),
    "shoe":       (28, 28, 28),
    "desk":       (139, 105, 20),
    "desk_dark":  (107, 79, 16),
    "desk_light": (168, 130, 10),
    "monitor":    (45, 45, 45),
    "mon_screen": (26, 58, 42),
    "screen_glow":(51, 255, 102),
    "screen_text":(34, 204, 85),
    "keyboard":   (58, 58, 58),
    "key":        (85, 85, 85),
    "chair":      (68, 68, 68),
    "chair_dark": (51, 51, 51),
    "coffee":     (245, 240, 225),
    "coffee_liq": (111, 78, 55),
    "bg1":        (27, 27, 47),
    "bg2":        (22, 36, 71),
    "floor":      (42, 42, 58),
    "lamp":       (255, 215, 0),
    "plant":      (45, 139, 70),
    "plant_dark": (30, 107, 50),
    "pot":        (194, 120, 78),
    "white":      (255, 255, 255),
    "sweat":      (136, 204, 255),
    "paper":      (240, 232, 208),
    "clock_c":    (204, 204, 204),
    "red":        (255, 68, 68),
    "black":      (0, 0, 0),
    "gray":       (102, 102, 102),
    "wall_tex":   (255, 255, 255),
}


def px(x, y, color):
    """픽셀 하나 그리기"""
    pygame.draw.rect(screen, color, (x * PIXEL, y * PIXEL, PIXEL, PIXEL))


def rect(x, y, w, h, color):
    """픽셀 블록 그리기"""
    pygame.draw.rect(screen, color, (x * PIXEL, y * PIXEL, w * PIXEL, h * PIXEL))


# ─── 배경 ───
def draw_background():
    for y in range(62):
        t = y / 62
        if t < 0.6:
            r = int(27 + (22 - 27) * (t / 0.6))
            g = int(27 + (36 - 27) * (t / 0.6))
            b = int(47 + (71 - 47) * (t / 0.6))
        else:
            tt = (t - 0.6) / 0.4
            r = int(22 + (42 - 22) * tt)
            g = int(36 + (42 - 36) * tt)
            b = int(71 + (58 - 71) * tt)
        pygame.draw.rect(screen, (r, g, b), (0, y * PIXEL, W, PIXEL))

    # 벽 텍스처
    for y in range(40):
        for x in range(56):
            if (x + y) % 4 == 0:
                s = pygame.Surface((PIXEL, PIXEL), pygame.SRCALPHA)
                s.fill((255, 255, 255, 5))
                screen.blit(s, (x * PIXEL, y * PIXEL))

    # 바닥 라인
    s = pygame.Surface((56 * PIXEL, PIXEL), pygame.SRCALPHA)
    s.fill((255, 255, 255, 20))
    screen.blit(s, (0, 52 * PIXEL))


# ─── 램프 ───
def draw_lamp(frame):
    rect(5, 10, 1, 30, C["gray"])
    rect(3, 8, 5, 2, (136, 136, 136))
    # 빛 글로우
    intensity = int(30 + math.sin(frame * 0.05) * 10)
    glow = pygame.Surface((16 * PIXEL, 16 * PIXEL), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 215, 0, intensity),
                       (8 * PIXEL, 8 * PIXEL), 8 * PIXEL)
    screen.blit(glow, ((5 - 8 + 0.5) * PIXEL, (12 - 8) * PIXEL))
    px(5, 10, C["lamp"])


# ─── 시계 ───
def draw_clock(frame):
    rect(38, 8, 6, 6, C["clock_c"])
    rect(39, 9, 4, 4, C["white"])
    px(41, 10, C["chair_dark"])
    px(41, 11, C["chair_dark"])
    if (frame % 60) < 30:
        px(42, 11, C["red"])
    else:
        px(40, 11, C["red"])


# ─── 의자 ───
def draw_chair():
    rect(24, 30, 1, 10, C["chair_dark"])
    rect(25, 28, 1, 12, C["chair"])
    rect(26, 28, 1, 2, C["chair"])
    rect(20, 40, 8, 2, C["chair"])
    rect(20, 42, 1, 8, C["chair_dark"])
    rect(27, 42, 1, 8, C["chair_dark"])
    px(19, 50, C["chair_dark"])
    px(28, 50, C["chair_dark"])
    rect(20, 50, 1, 1, C["key"])
    rect(27, 50, 1, 1, C["key"])


# ─── 책상 ───
def draw_desk():
    rect(8, 38, 40, 2, C["desk_dark"])
    rect(8, 36, 40, 2, C["desk"])
    rect(8, 35, 40, 1, C["desk_light"])
    rect(9, 40, 2, 12, C["desk_dark"])
    rect(45, 40, 2, 12, C["desk_dark"])
    rect(38, 40, 8, 6, C["desk"])
    rect(41, 42, 2, 1, C["desk_light"])


# ─── 모니터 ───
def draw_monitor(frame):
    rect(22, 33, 2, 3, C["monitor"])
    rect(20, 35, 6, 1, C["monitor"])
    rect(14, 20, 18, 13, C["monitor"])
    rect(15, 21, 16, 11, C["mon_screen"])

    # 코드 라인 스크롤
    lines = [
        [3, 8, 5, 3, 7], [5, 3, 8, 2], [2, 6, 4, 5, 3], [7, 3, 5],
        [3, 4, 6, 2, 5], [4, 8, 3], [6, 3, 5, 4], [2, 5, 7, 3],
    ]
    scroll = (frame // 8) % 4
    for i in range(7):
        li = (i + scroll) % len(lines)
        xoff = 16
        for seg in lines[li]:
            if xoff + seg < 30:
                br = 40 + random.randint(0, 20)
                color = (br, 180 + random.randint(0, 40), br + 40)
                rect(xoff, 22 + i + (1 if i > 3 else 0), seg, 1, color)
            xoff += seg + 1

    if frame % 30 < 15:
        px(29, 21, C["screen_glow"])


# ─── 키보드 ───
def draw_keyboard(frame, typing):
    rect(17, 36, 12, 2, C["keyboard"])
    for kx in range(10):
        pressed = typing and (frame + kx * 3) % 6 < 2
        px(18 + kx, 36, C["screen_glow"] if pressed else C["key"])
    for kx in range(8):
        pressed = typing and (frame + kx * 5 + 2) % 7 < 2
        px(19 + kx, 37, C["screen_glow"] if pressed else C["key"])


# ─── 커피 ───
def draw_coffee(frame):
    rect(33, 33, 3, 3, C["coffee"])
    rect(34, 33, 1, 3, C["coffee_liq"])
    px(36, 34, C["coffee"])
    if frame % 20 < 10:
        sy = 31 - (1 if frame % 10 > 5 else 0)
        s = pygame.Surface((PIXEL, PIXEL), pygame.SRCALPHA)
        s.fill((200, 200, 200, 100))
        screen.blit(s, (34 * PIXEL, sy * PIXEL))
        s.fill((200, 200, 200, 60))
        screen.blit(s, (33 * PIXEL, (sy - 1) * PIXEL))


# ─── 화분 ───
def draw_plant():
    rect(42, 33, 4, 3, C["pot"])
    rect(43, 32, 2, 1, C["pot"])
    for (x, y, c) in [
        (43, 31, "plant"), (44, 31, "plant"), (42, 30, "plant_dark"),
        (45, 30, "plant"), (43, 29, "plant"), (44, 29, "plant_dark"),
        (41, 29, "plant"), (46, 29, "plant_dark"), (43, 28, "plant_dark"),
    ]:
        px(x, y, C[c])


# ─── 서류 ───
def draw_papers():
    rect(10, 33, 5, 3, C["paper"])
    rect(11, 34, 3, 1, C["clock_c"])
    rect(11, 35, 2, 1, C["clock_c"])
    rect(9, 32, 5, 3, C["paper"])
    rect(10, 33, 3, 1, C["clock_c"])


# ─── 캐릭터 ───
def draw_character(frame, state):
    typing = state == "working"
    bob = -1 if typing and frame % 8 < 4 else 0
    arm_f = (frame // 3) % 4 if typing else 0

    # 몸통
    rect(21, 32 + bob, 4, 5, C["shirt"])
    rect(22, 32 + bob, 2, 5, C["shirt_dark"])
    px(22, 32 + bob, C["white"])
    px(23, 32 + bob, C["white"])

    # 팔
    if typing:
        ly = 36 if arm_f < 2 else 37
        rect(18, 34 + bob, 3, 1, C["shirt"])
        px(17, ly, C["skin"]); px(18, ly, C["skin"])
        ry = 36 if arm_f >= 2 else 37
        rect(25, 34 + bob, 3, 1, C["shirt"])
        px(27, ry, C["skin"]); px(28, ry, C["skin"])
    else:
        rect(19, 34 + bob, 2, 2, C["shirt"]); px(19, 36, C["skin"])
        rect(25, 34 + bob, 2, 2, C["shirt"]); px(26, 36, C["skin"])

    # 다리 & 신발
    rect(21, 37, 4, 3, C["pants"])
    rect(20, 40, 3, 2, C["pants"]); rect(24, 40, 3, 2, C["pants"])
    rect(19, 42, 3, 1, C["shoe"]); rect(24, 42, 3, 1, C["shoe"])

    # 머리
    hy = 26 + bob
    rect(21, hy, 5, 6, C["skin"])
    rect(20, hy + 1, 1, 4, C["skin"])
    rect(26, hy + 1, 1, 4, C["skin"])

    # 머리카락
    rect(20, hy - 1, 7, 2, C["hair"])
    rect(20, hy, 1, 2, C["hair"])
    rect(26, hy, 1, 2, C["hair"])

    # 눈 (깜빡임)
    blink = frame % 80
    if blink < 76:
        px(22, hy + 2, C["mon_screen"])
        px(24, hy + 2, (26, 26, 26))
    else:
        px(22, hy + 2, C["skin_dark"])
        px(24, hy + 2, C["skin_dark"])

    # 안경
    s = pygame.Surface((2 * PIXEL, 2 * PIXEL), pygame.SRCALPHA)
    s.fill((100, 150, 255, 50))
    screen.blit(s, (21 * PIXEL, (hy + 1) * PIXEL))
    screen.blit(s, (24 * PIXEL, (hy + 1) * PIXEL))
    px(23, hy + 2, C["gray"])

    # 입
    if typing and frame % 20 < 5:
        px(23, hy + 4, C["skin_dark"])
    else:
        px(22, hy + 4, C["skin_dark"])
        px(23, hy + 4, C["skin_dark"])

    # 땀방울
    if typing:
        sf = frame % 30
        if sf < 15:
            px(27, hy + 1 + sf // 5, C["sweat"])
        if frame % 45 < 10:
            px(19, hy + 2 + (frame % 45) // 4, C["sweat"])


# ─── 상태 텍스트 & 생산성 바 ───
def draw_status(frame, productivity):
    font = pygame.font.SysFont("monospace", 13, bold=True)
    msgs = ["CODING...", "DEBUGGING!", "SHIPPING!", "COMMIT!", "PUSH!", "DEPLOY!"]
    msg = msgs[(frame // 60) % len(msgs)]
    float_y = 6 + math.sin(frame * 0.08) * 2

    txt = font.render(msg, True, C["screen_glow"])
    tx = W // 2 - txt.get_width() // 2
    # 글로우 효과
    glow = font.render(msg, True, (51, 255, 102))
    glow_s = pygame.Surface(glow.get_size(), pygame.SRCALPHA)
    glow_s.blit(glow, (0, 0))
    glow_s.set_alpha(80)
    screen.blit(glow_s, (tx - 1, float_y * PIXEL - 1))
    screen.blit(glow_s, (tx + 1, float_y * PIXEL + 1))
    screen.blit(txt, (tx, float_y * PIXEL))

    # 생산성 바
    bar_x, bar_y = 10, 55
    rect(bar_x, bar_y, 36, 2, (51, 51, 51))
    fill_w = int(36 * productivity / 100)
    if productivity > 80:
        bar_c = C["screen_glow"]
    elif productivity > 50:
        bar_c = C["lamp"]
    else:
        bar_c = C["red"]
    rect(bar_x, bar_y, fill_w, 2, bar_c)

    small = pygame.font.SysFont("monospace", 10)
    label = small.render(f"생산성: {productivity}%", True, C["white"])
    screen.blit(label, (W // 2 - label.get_width() // 2, 59 * PIXEL))


# ─── HUD (조작법) ───
def draw_hud(state):
    small = pygame.font.SysFont("monospace", 10)
    controls = [
        ("[1] 일하기" if state != "working" else "▶ 일하는 중"),
        ("[2] 휴식" if state != "idle" else "▶ 쉬는 중"),
        "[R] 리셋",
        "[ESC] 종료",
    ]
    for i, txt in enumerate(controls):
        color = C["screen_glow"] if "▶" in txt else (120, 120, 120)
        label = small.render(txt, True, color)
        screen.blit(label, (4, H - 50 + i * 12))


# ═══════════════ 메인 루프 ═══════════════
def main():
    frame = 0
    productivity = 42
    state = "working"

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif ev.key == pygame.K_1:
                    state = "working"
                elif ev.key == pygame.K_2:
                    state = "idle"
                elif ev.key == pygame.K_r:
                    productivity = 0
                    frame = 0

        # 그리기
        draw_background()
        draw_lamp(frame)
        draw_clock(frame)
        draw_chair()
        draw_desk()
        draw_monitor(frame)
        draw_keyboard(frame, state == "working")
        draw_coffee(frame)
        draw_plant()
        draw_papers()
        draw_character(frame, state)
        draw_status(frame, productivity)
        draw_hud(state)

        pygame.display.flip()

        # 생산성 증가
        if state == "working" and frame % 15 == 0:
            productivity = min(100, productivity + 1)

        frame += 1
        clock.tick(12)


if __name__ == "__main__":
    main()