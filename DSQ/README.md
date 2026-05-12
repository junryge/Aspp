# DSQ — Open Design FULL Split-Zip Bundle

이 폴더는 `open-design v0.6.0` 풀 소스 백업의 분할 zip 색인입니다.
실제 zip 파일은 용량 제한(파일당 100MB) 때문에 git 트리에 직접 커밋되지 않고, 본 레포의 **GitHub Releases** 자산으로 업로드되어 있습니다.

## Assets

다운로드 위치: <https://github.com/junryge/Aspp/releases/tag/DSQ-open-design-v0.6.0>

| File | Size | SHA-256 |
|------|------|---------|
| `open-design-FULL.zip.001` | 1,048,576,000 B (~1000 MB) | `E9A23FB27D6BEB0D6724F4FCDAE2FD067D38AA98B2F6C4CA59452FEB5B667AEE` |
| `open-design-FULL.zip.002` | 1,048,576,000 B (~1000 MB) | `3714763D08FA1825414D269672A25D71B7683C6C86B8B9551397212BEEC8558F` |
| `open-design-FULL.zip.003` |   975,152,476 B (~930 MB)  | `D871A9559FF26D69040C3C35FC9ED2661B5A94528B68C4E94F1447B8F5034ECE` |

합본 zip 총 크기: 약 3,072,304,476 B (~2.93 GB).

## Reassemble

3개 파트를 같은 폴더에 다운로드한 뒤 합치세요.

### Windows (cmd)

```cmd
copy /b open-design-FULL.zip.001 + open-design-FULL.zip.002 + open-design-FULL.zip.003 open-design-FULL.zip
```

### Windows (PowerShell)

```powershell
cmd /c "copy /b open-design-FULL.zip.001 + open-design-FULL.zip.002 + open-design-FULL.zip.003 open-design-FULL.zip"
```

### macOS / Linux

```bash
cat open-design-FULL.zip.0* > open-design-FULL.zip
```

## Verify

합본 zip의 SHA-256을 비교해 보십시오. (값은 합본 후 1회 산출하여 여기에 갱신 권장.)

### Windows (PowerShell)

```powershell
Get-FileHash -Algorithm SHA256 open-design-FULL.zip
```

### macOS / Linux

```bash
shasum -a 256 open-design-FULL.zip
```

## Notes

- 본 번들은 외부 OSS 프로젝트 `open-codesign / open-design v0.6.0` 의 소스 트리 사본입니다. 재배포 조건은 원본 LICENSE를 따릅니다.
- Release 태그는 `DSQ-open-design-v0.6.0` 이며, 추후 갱신 시 새 태그를 사용합니다.
