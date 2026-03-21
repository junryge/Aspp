"""
소스코드 텍스트를 Base64로 인코딩/디코딩
DLP 우회용 - 웹 업로드 시 내용 스캔 방지

사용법:
  인코딩: python b64_code.py encode input.txt encoded.txt
  디코딩: python b64_code.py decode encoded.txt decoded.txt
"""

import sys
import base64


def encode_file(src, dst):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')

    # 80자씩 줄바꿈 (붙여넣기 편하게)
    lines = [encoded[i:i+80] for i in range(0, len(encoded), 80)]

    with open(dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[OK] 인코딩 완료: {src} -> {dst}")


def decode_file(src, dst):
    with open(src, 'r', encoding='utf-8') as f:
        encoded = f.read().replace('\n', '').replace('\r', '').strip()

    decoded = base64.b64decode(encoded).decode('utf-8')

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(decoded)

    print(f"[OK] 디코딩 완료: {src} -> {dst}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("사용법:")
        print("  인코딩: python b64_code.py encode input.txt encoded.txt")
        print("  디코딩: python b64_code.py decode encoded.txt decoded.txt")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == 'encode':
        encode_file(sys.argv[2], sys.argv[3])
    elif cmd == 'decode':
        decode_file(sys.argv[2], sys.argv[3])
    else:
        print(f"[ERROR] encode 또는 decode만 가능: {cmd}")
