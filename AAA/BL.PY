bl_info = {
    "name": "LLM Mesh Assistant",
    "author": "Custom",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
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
#  TOKEN.TXT 자동 로딩
# ============================================================
def find_token_file():
    """TOKEN.TXT 파일을 여러 경로에서 탐색하여 API 키를 읽어옴"""
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),       # 애드온 .py와 같은 폴더
        os.path.expanduser("~"),                           # 홈 디렉토리
        os.path.expanduser("~/Desktop"),                   # 바탕화면
        os.path.expanduser("~/Documents"),                 # 문서
        "D:/",
        "D:/TOKEN",
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
#  설정 프로퍼티 (SK Hynix 기본값 하드코딩)
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
        description="Qwen3 thinking 활성화 (정확↑ 속도↓)",
        default=False,
    )
    last_code: bpy.props.StringProperty(default="")
    last_response: bpy.props.StringProperty(default="")
    status: bpy.props.StringProperty(default="대기 중")
    token_path: bpy.props.StringProperty(
        default=_TOKEN_PATH if _TOKEN_PATH else "TOKEN.TXT 미발견",
    )
    is_running: bpy.props.BoolProperty(default=False)


# ============================================================
#  TOKEN.TXT 다시 불러오기
# ============================================================
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
            self.report({"WARNING"}, "TOKEN.TXT를 찾을 수 없습니다")
        return {"FINISHED"}


# ============================================================
#  씬 컨텍스트 수집
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
SYSTEM_PROMPT = """당신은 Blender Python(bpy) 전문가입니다. 사용자의 자연어 명령을 실행 가능한 bpy Python 코드로 변환합니다.

## 규칙
1. ```python 코드블록 안에 실행 가능한 bpy 코드만 작성
2. import bpy 항상 포함
3. 메쉬 수정 시 bmesh 적극 활용
4. 오브젝트/편집 모드 전환 정확히 처리
5. 씬 컨텍스트의 정확한 오브젝트 이름 사용
6. 작업 후 반드시 오브젝트 모드로 복귀
7. 에러 시 traceback 보고 수정 코드 제공
8. /no_think 지시 시 <think> 블록 없이 응답

## 코드 패턴
```python
import bpy
import bmesh

obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    # ... 메쉬 편집 ...
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
```
"""


# ============================================================
#  LLM API 호출
# ============================================================
def call_llm_api(props, messages):
    url = props.api_url.strip()

    # Qwen3 thinking 비활성화: /no_think 삽입
    if not props.enable_thinking:
        if messages and messages[0]["role"] == "system":
            if "/no_think" not in messages[0]["content"]:
                messages[0]["content"] = "/no_think\n" + messages[0]["content"]

    payload = json.dumps({
        "model": props.model_name,
        "messages": messages,
        "temperature": props.temperature,
        "max_tokens": 4096,
    }, ensure_ascii=False).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if props.api_key:
        headers["Authorization"] = f"Bearer {props.api_key}"

    req = urllib_request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            # <think>...</think> 블록 제거
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
    except Exception as e:
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
#  메인 오퍼레이터 (에러 재시도 루프 포함)
# ============================================================
class LLM_OT_SendPrompt(bpy.types.Operator):
    bl_idname = "llm_mesh.send_prompt"
    bl_label = "LLM 명령 전송"

    _result_code = None
    _result_error = None
    _result_status = None

    def execute(self, context):
        props = context.scene.llm_mesh_props
        if props.is_running:
            return {"CANCELLED"}
        if not props.prompt.strip():
            self.report({"WARNING"}, "명령을 입력하세요")
            return {"CANCELLED"}
        if not props.api_key:
            self.report({"ERROR"}, "API Key 없음. TOKEN.TXT 확인 또는 수동 입력")
            return {"CANCELLED"}

        props.is_running = True
        props.status = "LLM 요청 중..."

        thread = threading.Thread(target=self._run, args=(props,))
        thread.start()
        bpy.app.timers.register(lambda: self._poll(thread, props))
        return {"FINISHED"}

    def _run(self, props):
        try:
            scene_ctx = get_scene_context()
            user_msg = f"[씬 정보]\n{scene_ctx}\n\n[명령]\n{props.prompt}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            for attempt in range(props.max_retries):
                LLM_OT_SendPrompt._result_status = f"응답 대기... ({attempt+1}/{props.max_retries})"
                response = call_llm_api(props, messages)
                code = extract_code(response)

                if not code:
                    if attempt < props.max_retries - 1:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": "코드 블록이 없습니다. ```python 블록 안에 실행 가능한 bpy 코드를 작성해주세요."})
                        continue
                    LLM_OT_SendPrompt._result_error = f"코드 추출 실패\n\n{response[:500]}"
                    return

                LLM_OT_SendPrompt._result_code = code
                LLM_OT_SendPrompt._result_error = None
                return

        except Exception as e:
            LLM_OT_SendPrompt._result_error = str(e)

    @staticmethod
    def _poll(thread, props):
        if thread.is_alive():
            if LLM_OT_SendPrompt._result_status:
                props.status = LLM_OT_SendPrompt._result_status
            return 0.2

        props.is_running = False

        if LLM_OT_SendPrompt._result_error:
            props.status = f"오류: {LLM_OT_SendPrompt._result_error[:80]}"
            props.last_code = LLM_OT_SendPrompt._result_error or ""
        elif LLM_OT_SendPrompt._result_code:
            code = LLM_OT_SendPrompt._result_code
            props.last_code = code
            if props.auto_execute:
                ok, err = execute_code(code)
                props.status = "✅ 실행 완료" if ok else "❌ 실행 오류"
                if not ok:
                    props.last_code = f"# 오류: {err}\n\n{code}"
            else:
                props.status = "코드 생성 완료 (자동실행 OFF)"

        LLM_OT_SendPrompt._result_code = None
        LLM_OT_SendPrompt._result_error = None
        LLM_OT_SendPrompt._result_status = None

        for area in bpy.context.screen.areas:
            area.tag_redraw()
        return None


class LLM_OT_ExecuteCode(bpy.types.Operator):
    bl_idname = "llm_mesh.execute_code"
    bl_label = "코드 실행"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        if not props.last_code:
            return {"CANCELLED"}
        ok, err = execute_code(props.last_code)
        props.status = "✅ 수동 실행 완료" if ok else "❌ 실행 오류"
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
            resp = call_llm_api(props, [{"role": "user", "content": "Say OK"}])
            props.status = f"✅ 연결 성공: {resp[:50]}"
        except Exception as e:
            props.status = f"❌ 연결 실패: {str(e)[:60]}"
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


class LLM_OT_PresetCommand(bpy.types.Operator):
    bl_idname = "llm_mesh.preset_command"
    bl_label = "프리셋"
    command: bpy.props.StringProperty()

    def execute(self, context):
        context.scene.llm_mesh_props.prompt = self.command
        bpy.ops.llm_mesh.send_prompt()
        return {"FINISHED"}


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
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.llm_mesh_props

if __name__ == "__main__":
    register()