bl_info = {
    "name": "LLM Mesh Assistant",
    "author": "Custom",
    "version": (1, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > LLM Mesh",
    "description": "사내 LLM API를 이용한 자연어 3D 메쉬 수정 도구",
    "category": "3D View",
}

import bpy
import json
import os
import threading
import traceback
import re
from urllib import request as urllib_request


# ============================================================
#  모듈 레벨 스레드 통신 (오퍼레이터 인스턴스 참조 금지)
# ============================================================
_thread_state = {
    "thread": None,
    "code": None,
    "error": None,
    "status": None,
    "auto_execute": True,
}


# ============================================================
#  TOKEN.TXT 자동 로딩
# ============================================================
def find_token_file():
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.expanduser("~"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        "D:/",
        "D:/TOKEN",
        "D:/하이닉스",
    ]
    for folder in search_paths:
        filepath = os.path.join(folder, "TOKEN.TXT")
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                if token:
                    print(f"[LLM Mesh] TOKEN.TXT 로드 성공: {filepath}")
                    return token, filepath
            except Exception as e:
                print(f"[LLM Mesh] TOKEN.TXT 읽기 실패 ({filepath}): {e}")
    return "", ""


_AUTO_TOKEN, _TOKEN_PATH = find_token_file()


# ============================================================
#  설정 프로퍼티
# ============================================================
class LLMMeshProperties(bpy.types.PropertyGroup):
    api_url: bpy.props.StringProperty(
        name="API URL",
        default="http://dev.hcp.llm.skhynix.com/v1/chat/completions",
    )
    api_key: bpy.props.StringProperty(
        name="API Key",
        default=_AUTO_TOKEN if _AUTO_TOKEN else "",
        subtype="PASSWORD",
    )
    model_name: bpy.props.StringProperty(
        name="Model",
        default="Qwen3-235B-A22B-Instruct-2507",
    )
    prompt: bpy.props.StringProperty(
        name="명령",
        description="자연어로 메쉬 수정 명령 입력",
        default="",
    )
    temperature: bpy.props.FloatProperty(
        name="Temperature", default=0.1, min=0.0, max=1.0,
    )
    max_retries: bpy.props.IntProperty(
        name="오류 재시도", default=3, min=1, max=5,
    )
    auto_execute: bpy.props.BoolProperty(
        name="자동 실행", default=True,
    )
    enable_thinking: bpy.props.BoolProperty(
        name="Thinking 모드",
        description="LLM thinking/추론 모드 활성화 (Qwen3 등 지원 모델용)",
        default=False,
    )
    last_code: bpy.props.StringProperty(default="")
    status: bpy.props.StringProperty(default="대기 중")
    token_path: bpy.props.StringProperty(
        default=_TOKEN_PATH if _TOKEN_PATH else "TOKEN.TXT 미발견",
    )
    is_running: bpy.props.BoolProperty(default=False)


# ============================================================
#  씬 컨텍스트 수집 (메인 스레드에서만 호출)
# ============================================================
def get_scene_context():
    ctx = {"objects": [], "active_object": None, "mode": bpy.context.mode}
    for obj in bpy.context.scene.objects:
        info = {
            "name": obj.name, "type": obj.type,
            "location": [round(v, 3) for v in obj.location],
            "scale": [round(v, 3) for v in obj.scale],
        }
        if obj.type == "MESH" and obj.data:
            info["verts"] = len(obj.data.vertices)
            info["edges"] = len(obj.data.edges)
            info["faces"] = len(obj.data.polygons)
            if obj.modifiers:
                info["modifiers"] = [{"name": m.name, "type": m.type} for m in obj.modifiers]
        ctx["objects"].append(info)
    if bpy.context.active_object:
        ctx["active_object"] = bpy.context.active_object.name
    ctx["selected"] = [o.name for o in bpy.context.selected_objects]
    return json.dumps(ctx, ensure_ascii=False, indent=2)


# ============================================================
#  시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = """당신은 Blender Python(bpy) 코드 생성기입니다.
유일한 역할: 사용자의 자연어 명령을 실행 가능한 bpy Python 코드로 변환.

## 엄격한 규칙
1. 반드시 ```python 코드 블록 하나만 출력하세요. 설명 금지.
2. 항상 `import bpy`로 시작하세요.
3. [씬 정보]를 읽고 정확한 오브젝트 이름을 사용하세요.
4. 사용자가 특정 오브젝트를 지정하지 않으면 active_object를 대상으로 하세요.
5. 작업 끝나면 반드시 오브젝트 모드로 복귀: `bpy.ops.object.mode_set(mode='OBJECT')`
6. 버텍스/엣지/면 수준 작업에는 bmesh를 사용하세요.
7. 모디파이어는 `bpy.ops.object.modifier_add(type='...')`를 사용하세요.
8. 사용자가 요청하지 않은 오브젝트나 작업을 추가하지 마세요.
9. 명령이 불분명하면 가장 합리적인 메쉬 작업으로 판단하세요.

## 예시

사용자: "Subdivision Surface 추가해줘"
```python
import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers[-1].levels = 2
    obj.modifiers[-1].render_levels = 2
```

사용자: "선택된 오브젝트를 스무딩해줘"
```python
import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.shade_smooth()
```

사용자: "버텍스를 부드럽게 해줘"
```python
import bpy
import bmesh
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=5)
    bpy.ops.object.mode_set(mode='OBJECT')
```

사용자: "Bevel 모디파이어 추가"
```python
import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.modifier_add(type='BEVEL')
    obj.modifiers[-1].width = 0.02
    obj.modifiers[-1].segments = 3
```

사용자: "면을 돌출시켜줘"
```python
import bpy
import bmesh
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, 1.0)}
    )
    bpy.ops.object.mode_set(mode='OBJECT')
```

사용자: "노멀 재계산"
```python
import bpy
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
```

## 반드시 지켜야 할 것
- ```python 블록만 출력. 그 외 텍스트 절대 금지.
- 씬 정보에서 정확한 오브젝트 이름을 사용.
- 사용자가 요청한 것만 정확히 수행. 그 이상도 이하도 안 됨.
"""


# ============================================================
#  LLM API 호출 (스레드 안전 - bpy 접근 없음)
# ============================================================
def call_llm_api(url, api_key, model, temperature, enable_thinking, messages):
    """순수 HTTP 호출 - bpy 프로퍼티 직접 참조하지 않음"""

    # Qwen 모델일 때만 /no_think 삽입
    if not enable_thinking and "qwen" in model.lower():
        if messages and messages[0]["role"] == "system":
            if "/no_think" not in messages[0]["content"]:
                messages[0]["content"] = "/no_think\n" + messages[0]["content"]

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }, ensure_ascii=False).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib_request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
    except urllib_request.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        error_msg = f"HTTP {e.code} 에러\n응답: {error_body[:500]}"
        print(f"[LLM Mesh] {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        print(f"[LLM Mesh] API 오류: {e}")
        raise RuntimeError(f"API 호출 실패: {e}")


# ============================================================
#  코드 추출 / 실행
# ============================================================
def extract_code(text):
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        code = "\n".join(matches)
        if "bpy" in code:
            return code
    return None


def execute_code(code):
    try:
        exec(code, {"__builtins__": __builtins__})
        return True, None
    except Exception:
        return False, traceback.format_exc()


# ============================================================
#  백그라운드 워커 (독립 함수 - self 참조 없음)
# ============================================================
def _llm_worker(url, api_key, model, temperature, enable_thinking,
                max_retries, scene_ctx, prompt):
    """백그라운드 스레드에서 실행 - bpy 접근 절대 금지"""
    try:
        user_msg = (
            f"[씬 정보]\n{scene_ctx}\n\n"
            f"[명령]\n{prompt}\n\n"
            f"```python 코드 블록만 출력하세요. 설명 금지."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(max_retries):
            _thread_state["status"] = f"응답 대기... ({attempt+1}/{max_retries})"

            response = call_llm_api(
                url, api_key, model, temperature, enable_thinking, messages
            )
            code = extract_code(response)

            if not code:
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "코드 블록이 없습니다. ```python 블록 안에 실행 가능한 bpy 코드를 작성해주세요."})
                    continue
                _thread_state["error"] = f"코드 추출 실패\n\n{response[:500]}"
                return

            _thread_state["code"] = code
            _thread_state["error"] = None
            return

    except Exception as e:
        _thread_state["error"] = str(e)


# ============================================================
#  타이머 콜백 (독립 함수 - 오퍼레이터 참조 없음)
# ============================================================
def _timer_poll():
    """메인 스레드 타이머 - bpy 접근 안전"""
    thread = _thread_state.get("thread")
    if thread is None:
        return None  # 타이머 해제

    # 아직 실행 중
    if thread.is_alive():
        try:
            props = bpy.context.scene.llm_mesh_props
            if _thread_state["status"]:
                props.status = _thread_state["status"]
        except Exception:
            pass
        return 0.3  # 300ms 후 재체크

    # 스레드 완료
    try:
        props = bpy.context.scene.llm_mesh_props
        props.is_running = False

        if _thread_state["error"]:
            props.status = f"오류: {_thread_state['error'][:80]}"
            props.last_code = _thread_state["error"] or ""
            print(f"[LLM Mesh] 오류: {_thread_state['error']}")
        elif _thread_state["code"]:
            code = _thread_state["code"]
            props.last_code = code
            print(f"[LLM Mesh] 코드 생성 완료 ({len(code)} chars)")

            if _thread_state["auto_execute"]:
                ok, err = execute_code(code)
                if ok:
                    props.status = "✅ 실행 완료"
                    print("[LLM Mesh] 코드 실행 성공")
                else:
                    props.status = "❌ 실행 오류"
                    props.last_code = f"# 오류:\n# {err}\n\n{code}"
                    print(f"[LLM Mesh] 실행 오류: {err}")
            else:
                props.status = "코드 생성 완료 (자동실행 OFF)"
        else:
            props.status = "완료 (결과 없음)"

    except Exception as e:
        print(f"[LLM Mesh] 타이머 오류: {e}")

    # 상태 리셋
    _thread_state["thread"] = None
    _thread_state["code"] = None
    _thread_state["error"] = None
    _thread_state["status"] = None

    # 화면 갱신
    try:
        for area in bpy.context.screen.areas:
            area.tag_redraw()
    except Exception:
        pass

    return None  # 타이머 해제


# ============================================================
#  오퍼레이터들
# ============================================================
class LLM_OT_SendPrompt(bpy.types.Operator):
    bl_idname = "llm_mesh.send_prompt"
    bl_label = "LLM 명령 전송"

    def execute(self, context):
        props = context.scene.llm_mesh_props

        if props.is_running:
            self.report({"WARNING"}, "이미 실행 중")
            return {"CANCELLED"}
        if not props.prompt.strip():
            self.report({"WARNING"}, "명령을 입력하세요")
            return {"CANCELLED"}
        if not props.api_key:
            self.report({"ERROR"}, "API Key 없음")
            return {"CANCELLED"}

        # 메인 스레드에서 씬 정보 미리 수집
        scene_ctx = get_scene_context()

        # 프로퍼티 값을 순수 Python 변수로 복사 (스레드에서 bpy 접근 방지)
        url = str(props.api_url)
        api_key = str(props.api_key)
        model = str(props.model_name)
        temperature = float(props.temperature)
        enable_thinking = bool(props.enable_thinking)
        max_retries = int(props.max_retries)
        prompt = str(props.prompt)

        # 상태 초기화
        _thread_state["code"] = None
        _thread_state["error"] = None
        _thread_state["status"] = "LLM 요청 중..."
        _thread_state["auto_execute"] = bool(props.auto_execute)

        props.is_running = True
        props.status = "LLM 요청 중..."

        # 백그라운드 스레드 시작
        thread = threading.Thread(
            target=_llm_worker,
            args=(url, api_key, model, temperature, enable_thinking,
                  max_retries, scene_ctx, prompt),
            daemon=True,
        )
        _thread_state["thread"] = thread
        thread.start()

        # 타이머 등록 (독립 함수, 오퍼레이터 참조 없음)
        bpy.app.timers.register(_timer_poll, first_interval=0.5)

        self.report({"INFO"}, "LLM 요청 전송됨")
        return {"FINISHED"}


class LLM_OT_ExecuteCode(bpy.types.Operator):
    bl_idname = "llm_mesh.execute_code"
    bl_label = "코드 실행"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        if not props.last_code:
            return {"CANCELLED"}
        ok, err = execute_code(props.last_code)
        props.status = "✅ 수동 실행 완료" if ok else "❌ 실행 오류"
        if not ok:
            self.report({"ERROR"}, f"오류: {err[:100]}")
        return {"FINISHED"}


class LLM_OT_CopyToEditor(bpy.types.Operator):
    bl_idname = "llm_mesh.copy_to_editor"
    bl_label = "에디터로 복사"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        if not props.last_code:
            return {"CANCELLED"}
        name = "LLM_Generated_Code"
        if name in bpy.data.texts:
            bpy.data.texts[name].clear()
        else:
            bpy.data.texts.new(name)
        bpy.data.texts[name].write(props.last_code)
        props.status = "📋 에디터에 복사됨"
        return {"FINISHED"}


class LLM_OT_TestConnection(bpy.types.Operator):
    bl_idname = "llm_mesh.test_connection"
    bl_label = "연결 테스트"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        try:
            resp = call_llm_api(
                str(props.api_url), str(props.api_key), str(props.model_name),
                0.1, False, [{"role": "user", "content": "Say OK"}]
            )
            props.status = f"✅ 연결 성공: {resp[:50]}"
            self.report({"INFO"}, "API 연결 성공!")
        except Exception as e:
            props.status = f"❌ 연결 실패: {str(e)[:60]}"
            self.report({"ERROR"}, f"연결 실패: {e}")
        return {"FINISHED"}


class LLM_OT_ReloadToken(bpy.types.Operator):
    bl_idname = "llm_mesh.reload_token"
    bl_label = "TOKEN 재로드"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        token, path = find_token_file()
        if token:
            props.api_key = token
            props.token_path = path
            props.status = f"✅ 토큰 로드: {path}"
        else:
            props.status = "❌ TOKEN.TXT 미발견"
        return {"FINISHED"}


class LLM_OT_PresetCommand(bpy.types.Operator):
    bl_idname = "llm_mesh.preset_command"
    bl_label = "프리셋"
    command: bpy.props.StringProperty()

    def execute(self, context):
        context.scene.llm_mesh_props.prompt = self.command
        bpy.ops.llm_mesh.send_prompt()
        return {"FINISHED"}


# ============================================================
#  UI 패널
# ============================================================
class LLM_PT_MeshPanel(bpy.types.Panel):
    bl_label = "LLM Mesh Assistant"
    bl_idname = "LLM_PT_mesh_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LLM Mesh"

    def draw(self, context):
        layout = self.layout
        props = context.scene.llm_mesh_props

        # API 설정
        box = layout.box()
        box.label(text="API 설정", icon="URL")
        box.prop(props, "api_url")
        box.prop(props, "api_key")
        box.prop(props, "model_name")
        row = box.row()
        row.prop(props, "temperature")
        row.prop(props, "max_retries")
        box.prop(props, "enable_thinking")
        row = box.row(align=True)
        row.operator("llm_mesh.test_connection", icon="VIEWZOOM")
        row.operator("llm_mesh.reload_token", icon="FILE_REFRESH")
        if props.token_path:
            box.label(text=f"Token: {props.token_path}", icon="KEYINGSET")

        layout.separator()

        # 명령 입력
        box = layout.box()
        box.label(text="메쉬 수정 명령", icon="MESH_DATA")
        box.prop(props, "prompt", text="")
        box.prop(props, "auto_execute")
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("llm_mesh.send_prompt", text="명령 전송", icon="PLAY")
        row.enabled = not props.is_running

        layout.separator()

        # 상태 + 코드
        box = layout.box()
        box.label(text=props.status, icon="INFO")
        if props.last_code:
            box2 = layout.box()
            box2.label(text="생성된 코드", icon="TEXT")
            for line in props.last_code.split("\n")[:8]:
                box2.label(text=line[:70])
            if len(props.last_code.split("\n")) > 8:
                box2.label(text="...")
            row = box2.row(align=True)
            row.operator("llm_mesh.execute_code", text="실행", icon="PLAY")
            row.operator("llm_mesh.copy_to_editor", text="복사", icon="COPYDOWN")


class LLM_PT_PresetsPanel(bpy.types.Panel):
    bl_label = "빠른 프리셋"
    bl_idname = "LLM_PT_presets_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LLM Mesh"
    bl_parent_id = "LLM_PT_mesh_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        presets = [
            ("Subdivision Surface 추가 (레벨 2)", "MOD_SUBSURF"),
            ("선택 면 Extrude 1.0", "MESH_DATA"),
            ("버텍스 스무딩 (0.5, 5회)", "MOD_SMOOTH"),
            ("노멀 재계산", "NORMALS_FACE"),
            ("Bevel 모디파이어 추가", "MOD_BEVEL"),
            ("삼각→사각면 변환", "MESH_GRID"),
            ("Mirror X축", "MOD_MIRROR"),
            ("Decimate (ratio 0.5)", "MOD_DECIM"),
            ("Shade Smooth", "MOD_SMOOTH"),
            ("모든 모디파이어 적용", "FILE_TICK"),
        ]
        for text, icon in presets:
            op = layout.operator("llm_mesh.preset_command", text=text, icon=icon)
            op.command = text


# ============================================================
#  등록
# ============================================================
classes = (
    LLMMeshProperties, LLM_OT_SendPrompt, LLM_OT_ExecuteCode,
    LLM_OT_CopyToEditor, LLM_OT_TestConnection, LLM_OT_ReloadToken,
    LLM_OT_PresetCommand, LLM_PT_MeshPanel, LLM_PT_PresetsPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.llm_mesh_props = bpy.props.PointerProperty(type=LLMMeshProperties)
    if _AUTO_TOKEN:
        print(f"[LLM Mesh] TOKEN 자동 로드: {_TOKEN_PATH}")
    else:
        print("[LLM Mesh] TOKEN.TXT 미발견 - 수동 입력 필요")

def unregister():
    # 실행 중인 타이머 정리
    try:
        if bpy.app.timers.is_registered(_timer_poll):
            bpy.app.timers.unregister(_timer_poll)
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.llm_mesh_props

if __name__ == "__main__":
    register()