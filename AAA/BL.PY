bl_info = {
    "name": "LLM Mesh Assistant",
    "author": "Custom",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > LLM Mesh",
    "description": "사내 LLM API를 이용한 자연어 3D 메쉬 수정 도구",
    "category": "3D View",
}

import bpy
import json
import threading
import traceback
import re
from urllib import request as urllib_request


# ============================================================
#  설정 프로퍼티
# ============================================================
class LLMMeshProperties(bpy.types.PropertyGroup):
    api_url: bpy.props.StringProperty(
        name="API URL",
        description="사내 LLM API 엔드포인트 (예: http://10.0.0.1:8000/v1/chat/completions)",
        default="http://localhost:8000/v1/chat/completions",
    )
    api_key: bpy.props.StringProperty(
        name="API Key",
        description="API 인증 키",
        default="",
        subtype="PASSWORD",
    )
    model_name: bpy.props.StringProperty(
        name="Model",
        description="모델 이름 (예: gpt-4, codellama, qwen2.5-coder 등)",
        default="default",
    )
    prompt: bpy.props.StringProperty(
        name="명령",
        description="자연어로 메쉬 수정 명령을 입력하세요",
        default="",
    )
    temperature: bpy.props.FloatProperty(
        name="Temperature",
        default=0.1,
        min=0.0,
        max=1.0,
    )
    max_retries: bpy.props.IntProperty(
        name="오류 재시도",
        description="코드 실행 실패 시 LLM에게 재시도 요청 횟수",
        default=3,
        min=1,
        max=5,
    )
    auto_execute: bpy.props.BoolProperty(
        name="자동 실행",
        description="생성된 코드를 자동으로 실행",
        default=True,
    )
    last_code: bpy.props.StringProperty(
        name="마지막 코드",
        default="",
    )
    status: bpy.props.StringProperty(
        name="상태",
        default="대기 중",
    )
    is_running: bpy.props.BoolProperty(default=False)


# ============================================================
#  씬 컨텍스트 수집
# ============================================================
def get_scene_context():
    """현재 Blender 씬 정보를 LLM에 전달할 컨텍스트로 수집"""
    ctx = {"objects": [], "active_object": None, "mode": bpy.context.mode}

    for obj in bpy.context.scene.objects:
        info = {
            "name": obj.name,
            "type": obj.type,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
        }
        if obj.type == "MESH" and obj.data:
            info["vertices"] = len(obj.data.vertices)
            info["edges"] = len(obj.data.edges)
            info["polygons"] = len(obj.data.polygons)
            # 모디파이어
            if obj.modifiers:
                info["modifiers"] = [
                    {"name": m.name, "type": m.type} for m in obj.modifiers
                ]
        ctx["objects"].append(info)

    if bpy.context.active_object:
        ctx["active_object"] = bpy.context.active_object.name

    # 선택된 오브젝트
    ctx["selected_objects"] = [o.name for o in bpy.context.selected_objects]

    return json.dumps(ctx, ensure_ascii=False, indent=2)


# ============================================================
#  시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = """당신은 Blender Python(bpy) 전문가입니다. 사용자의 자연어 명령을 실행 가능한 bpy Python 코드로 변환합니다.

## 규칙
1. 반드시 ```python 코드블록 안에 실행 가능한 bpy 코드만 작성하세요.
2. import bpy는 항상 포함하세요.
3. 설명이나 주석 외의 일반 텍스트는 코드블록 밖에 작성하세요.
4. 메쉬 수정 시 bmesh 모듈을 적극 활용하세요.
5. 오브젝트 모드/편집 모드 전환을 정확히 처리하세요.
6. 현재 씬 컨텍스트를 참고하여 정확한 오브젝트 이름을 사용하세요.
7. 작업 후 씬을 업데이트(depsgraph)하세요.
8. 에러가 발생하면 traceback을 보고 수정된 코드를 제공하세요.

## 메쉬 수정 관련 주요 작업
- 버텍스 이동/삭제/추가
- 면(face) 세분화(subdivide), 돌출(extrude), 인셋(inset)
- 스무딩(smooth), 셰이드 플랫/스무스
- 모디파이어 추가 (Subdivision Surface, Boolean, Mirror, Bevel 등)
- 메쉬 병합, 분리
- 노멀 재계산
- 버텍스 그룹 편집
- UV 편집
- 선택 영역 기반 작업

## 코드 예시 패턴
```python
import bpy
import bmesh

obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    # ... 메쉬 편집 작업 ...
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
```
"""


# ============================================================
#  LLM API 호출 (urllib - 외부 패키지 불필요)
# ============================================================
def call_llm_api(props, messages):
    """사내 OpenAI 호환 API 호출 (requests 없이 urllib 사용)"""
    url = props.api_url.strip()
    payload = json.dumps({
        "model": props.model_name,
        "messages": messages,
        "temperature": props.temperature,
        "max_tokens": 4096,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if props.api_key:
        headers["Authorization"] = f"Bearer {props.api_key}"

    req = urllib_request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib_request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"API 호출 실패: {e}")


# ============================================================
#  코드 추출 및 실행
# ============================================================
def extract_code(response_text):
    """LLM 응답에서 Python 코드 블록 추출"""
    # ```python ... ``` 패턴
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return "\n".join(matches)

    # ``` ... ``` 패턴 (언어 표시 없음)
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        code = "\n".join(matches)
        if "bpy" in code or "import" in code:
            return code

    return None


def execute_code(code):
    """Blender에서 Python 코드 실행"""
    try:
        exec(code, {"__builtins__": __builtins__})
        return True, None
    except Exception:
        return False, traceback.format_exc()


# ============================================================
#  메인 실행 오퍼레이터
# ============================================================
class LLM_OT_SendPrompt(bpy.types.Operator):
    bl_idname = "llm_mesh.send_prompt"
    bl_label = "LLM 명령 전송"
    bl_description = "자연어 명령을 LLM에 보내고 메쉬 수정 코드를 받아 실행합니다"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        if props.is_running:
            self.report({"WARNING"}, "이미 실행 중입니다")
            return {"CANCELLED"}
        if not props.prompt.strip():
            self.report({"WARNING"}, "명령을 입력하세요")
            return {"CANCELLED"}

        props.is_running = True
        props.status = "LLM 요청 중..."

        # 백그라운드 스레드에서 API 호출
        thread = threading.Thread(
            target=self._run_in_thread, args=(props, context)
        )
        thread.start()

        # 타이머로 완료 체크
        bpy.app.timers.register(lambda: self._check_thread(thread, props, context))

        return {"FINISHED"}

    # ---------- 내부 변수 (스레드 간 공유) ----------
    _result_code = None
    _result_error = None
    _result_status = None

    def _run_in_thread(self, props, context):
        try:
            scene_ctx = get_scene_context()
            user_msg = (
                f"[현재 씬 정보]\n{scene_ctx}\n\n"
                f"[사용자 명령]\n{props.prompt}"
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            # LLM 호출 + 에러 재시도 루프
            for attempt in range(props.max_retries):
                LLM_OT_SendPrompt._result_status = (
                    f"LLM 응답 대기 중... (시도 {attempt + 1}/{props.max_retries})"
                )
                response = call_llm_api(props, messages)
                code = extract_code(response)

                if not code:
                    LLM_OT_SendPrompt._result_code = None
                    LLM_OT_SendPrompt._result_error = (
                        f"코드를 추출할 수 없습니다.\n\nLLM 응답:\n{response[:500]}"
                    )
                    return

                LLM_OT_SendPrompt._result_code = code

                if not props.auto_execute:
                    LLM_OT_SendPrompt._result_status = "코드 생성 완료 (자동실행 OFF)"
                    LLM_OT_SendPrompt._result_error = None
                    return

                # 코드 실행은 메인 스레드에서 해야 함 → 여기서는 코드만 저장
                LLM_OT_SendPrompt._result_status = "코드 실행 대기"
                LLM_OT_SendPrompt._result_error = None
                return

        except Exception as e:
            LLM_OT_SendPrompt._result_code = None
            LLM_OT_SendPrompt._result_error = str(e)
            LLM_OT_SendPrompt._result_status = "오류 발생"

    @staticmethod
    def _check_thread(thread, props, context):
        if thread.is_alive():
            if LLM_OT_SendPrompt._result_status:
                props.status = LLM_OT_SendPrompt._result_status
            return 0.2  # 200ms 후 다시 체크

        # 스레드 완료
        props.is_running = False

        if LLM_OT_SendPrompt._result_error:
            props.status = f"오류: {LLM_OT_SendPrompt._result_error[:80]}"
            props.last_code = LLM_OT_SendPrompt._result_error
        elif LLM_OT_SendPrompt._result_code:
            code = LLM_OT_SendPrompt._result_code
            props.last_code = code

            if props.auto_execute:
                success, err = execute_code(code)
                if success:
                    props.status = "✅ 실행 완료"
                else:
                    props.status = f"❌ 실행 오류 (수동 확인 필요)"
                    props.last_code = f"# === 실행 오류 ===\n# {err}\n\n{code}"
            else:
                props.status = "코드 생성 완료"

        # 리셋
        LLM_OT_SendPrompt._result_code = None
        LLM_OT_SendPrompt._result_error = None
        LLM_OT_SendPrompt._result_status = None

        # 화면 갱신
        for area in bpy.context.screen.areas:
            area.tag_redraw()

        return None  # 타이머 해제


# ============================================================
#  코드 수동 실행 오퍼레이터
# ============================================================
class LLM_OT_ExecuteCode(bpy.types.Operator):
    bl_idname = "llm_mesh.execute_code"
    bl_label = "코드 실행"
    bl_description = "생성된 코드를 수동으로 실행합니다"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        code = props.last_code
        if not code:
            self.report({"WARNING"}, "실행할 코드가 없습니다")
            return {"CANCELLED"}

        success, err = execute_code(code)
        if success:
            props.status = "✅ 수동 실행 완료"
            self.report({"INFO"}, "코드 실행 성공")
        else:
            props.status = f"❌ 실행 오류"
            self.report({"ERROR"}, f"실행 오류: {err[:100]}")

        return {"FINISHED"}


# ============================================================
#  코드 → 텍스트 에디터로 복사
# ============================================================
class LLM_OT_CopyToEditor(bpy.types.Operator):
    bl_idname = "llm_mesh.copy_to_editor"
    bl_label = "코드 에디터로 복사"
    bl_description = "생성된 코드를 Blender 텍스트 에디터에 복사합니다"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        code = props.last_code
        if not code:
            self.report({"WARNING"}, "복사할 코드가 없습니다")
            return {"CANCELLED"}

        text_name = "LLM_Generated_Code"
        if text_name in bpy.data.texts:
            text_block = bpy.data.texts[text_name]
            text_block.clear()
        else:
            text_block = bpy.data.texts.new(text_name)

        text_block.write(code)
        props.status = "📋 코드가 텍스트 에디터에 복사됨"
        self.report({"INFO"}, f"'{text_name}' 텍스트 블록에 복사 완료")

        return {"FINISHED"}


# ============================================================
#  API 연결 테스트
# ============================================================
class LLM_OT_TestConnection(bpy.types.Operator):
    bl_idname = "llm_mesh.test_connection"
    bl_label = "연결 테스트"
    bl_description = "사내 LLM API 연결을 테스트합니다"

    def execute(self, context):
        props = context.scene.llm_mesh_props
        try:
            messages = [{"role": "user", "content": "Say 'OK' if you can hear me."}]
            response = call_llm_api(props, messages)
            props.status = f"✅ 연결 성공: {response[:50]}"
            self.report({"INFO"}, "API 연결 성공!")
        except Exception as e:
            props.status = f"❌ 연결 실패: {str(e)[:60]}"
            self.report({"ERROR"}, f"연결 실패: {e}")

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

        # --- API 설정 ---
        box = layout.box()
        box.label(text="API 설정", icon="URL")
        box.prop(props, "api_url")
        box.prop(props, "api_key")
        box.prop(props, "model_name")
        row = box.row()
        row.prop(props, "temperature")
        row.prop(props, "max_retries")
        box.operator("llm_mesh.test_connection", icon="VIEWZOOM")

        layout.separator()

        # --- 명령 입력 ---
        box = layout.box()
        box.label(text="메쉬 수정 명령", icon="MESH_DATA")
        box.prop(props, "prompt", text="")
        box.prop(props, "auto_execute")

        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("llm_mesh.send_prompt", text="명령 전송", icon="PLAY")
        row.enabled = not props.is_running

        layout.separator()

        # --- 상태 ---
        box = layout.box()
        box.label(text="상태", icon="INFO")
        box.label(text=props.status)

        # --- 코드 미리보기 ---
        if props.last_code:
            box = layout.box()
            box.label(text="생성된 코드", icon="TEXT")
            # 코드 처음 5줄만 미리보기
            lines = props.last_code.split("\n")[:8]
            for line in lines:
                box.label(text=line[:60])
            if len(props.last_code.split("\n")) > 8:
                box.label(text="... (더 보려면 에디터로 복사)")

            row = box.row(align=True)
            row.operator("llm_mesh.execute_code", text="실행", icon="PLAY")
            row.operator("llm_mesh.copy_to_editor", text="에디터로 복사", icon="COPYDOWN")


# ============================================================
#  프리셋 메쉬 명령 패널
# ============================================================
class LLM_PT_PresetsPanel(bpy.types.Panel):
    bl_label = "빠른 명령 프리셋"
    bl_idname = "LLM_PT_presets_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "LLM Mesh"
    bl_parent_id = "LLM_PT_mesh_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        presets = [
            ("선택 오브젝트 Subdivision Surface 추가 (레벨 2)", "SURFACE_DATA"),
            ("선택된 면을 돌출(Extrude) 1.0만큼", "MESH_DATA"),
            ("모든 버텍스 스무딩 (factor 0.5, 반복 5회)", "MOD_SMOOTH"),
            ("노멀 바깥쪽으로 재계산", "NORMALS_FACE"),
            ("선택 오브젝트에 Bevel 모디파이어 추가", "MOD_BEVEL"),
            ("메쉬 삼각면을 사각면으로 변환", "MESH_GRID"),
            ("미러 모디파이어 추가 (X축)", "MOD_MIRROR"),
        ]
        for text, icon in presets:
            op = layout.operator(
                "llm_mesh.preset_command", text=text[:40], icon=icon
            )
            op.command = text


class LLM_OT_PresetCommand(bpy.types.Operator):
    bl_idname = "llm_mesh.preset_command"
    bl_label = "프리셋 명령"

    command: bpy.props.StringProperty()

    def execute(self, context):
        props = context.scene.llm_mesh_props
        props.prompt = self.command
        bpy.ops.llm_mesh.send_prompt()
        return {"FINISHED"}


# ============================================================
#  등록
# ============================================================
classes = (
    LLMMeshProperties,
    LLM_OT_SendPrompt,
    LLM_OT_ExecuteCode,
    LLM_OT_CopyToEditor,
    LLM_OT_TestConnection,
    LLM_OT_PresetCommand,
    LLM_PT_MeshPanel,
    LLM_PT_PresetsPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.llm_mesh_props = bpy.props.PointerProperty(
        type=LLMMeshProperties
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.llm_mesh_props


if __name__ == "__main__":
    register()