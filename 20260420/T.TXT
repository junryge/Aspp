"""
demos_v1/frontend.py - HTML_TEMPLATE (main frontend UI)
"""

# ============================================
# HTML 템플릿
# ============================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demos V1.0</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8f7f4;color:#1a1a1a;display:flex;height:100vh}
.sidebar{width:250px;background:#fff;border-right:1px solid #e5e3de;padding:20px 16px;display:flex;flex-direction:column;overflow-y:auto;transition:width .2s ease,padding .2s ease;flex-shrink:0;scrollbar-width:none}
.sidebar::-webkit-scrollbar{display:none}
.sidebar.collapsed{width:48px;padding:12px 6px;overflow:hidden}
.sidebar.collapsed .sidebar-inner{display:none}
.sidebar.collapsed .sidebar-file-panel{display:none!important}
.sidebar.collapsed .sidebar-logo{display:none}
.sidebar-toggle{position:absolute;top:12px;left:250px;width:24px;height:24px;border-radius:0 6px 6px 0;border:1px solid #e5e3de;border-left:none;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;color:#999;z-index:200;transition:left .2s ease}
.sidebar-toggle:hover{background:#eef2ff;color:#6366f1}
body.sb-collapsed .sidebar-toggle{left:48px}
body.sb-collapsed .chat-box-fixed{left:48px}
.sidebar-toggle-mini{width:100%;padding:6px 0;border:none;background:none;cursor:pointer;font-size:16px;display:none;color:#6366f1}
.sidebar.collapsed .sidebar-toggle-mini{display:block}
.sidebar-logo{font-size:22px;font-weight:700;margin-bottom:24px}.sidebar-logo span{color:#6366f1}
.sidebar-btn{width:100%;padding:10px 14px;border-radius:8px;border:none;cursor:pointer;font-size:14px;text-align:left;margin-bottom:4px;background:transparent;color:#555;transition:all .15s}
.sidebar-btn:hover{background:#f3f2ef}.sidebar-btn.active{background:#6366f1;color:#fff}
.session-item{display:flex;align-items:center;padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12px;color:#555;transition:all .12s;gap:4px;margin:1px 4px}
.session-item:hover{background:#f3f2ef}
.session-item.current{background:#eef2ff;color:#6366f1;font-weight:600}
.session-item .session-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.session-item .session-cb{width:14px;height:14px;cursor:pointer;accent-color:#6366f1;flex-shrink:0}
.session-item .session-time{font-size:9px;color:#bbb;flex-shrink:0}
.sidebar-collapse{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:4px 0;user-select:none}
.sidebar-collapse:hover{opacity:.8}
.sidebar-collapse .arrow{font-size:10px;color:#999;transition:transform .15s}
.sidebar-collapse .arrow.open{transform:rotate(90deg)}
.sidebar-collapsible{overflow:hidden;transition:max-height .2s ease}
.sidebar-collapsible.collapsed{max-height:0!important}
.session-actions{display:flex;gap:4px;padding:4px 8px}
.session-actions button{flex:1;padding:3px 0;border:1px solid #e5e3de;border-radius:4px;background:#fafaf8;font-size:10px;cursor:pointer;transition:all .12s}
.session-actions button:hover{background:#eef2ff;border-color:#6366f1}
.session-actions button.danger{color:#ef4444;border-color:#fca5a5}
.session-actions button.danger:hover{background:#fef2f2;border-color:#ef4444}
.sidebar-section{font-size:11px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.5px;margin:20px 0 8px 8px}
.skill-count{font-size:11px;color:#6366f1;background:#eef2ff;padding:2px 8px;border-radius:10px;margin-left:6px}
.sidebar-footer{margin-top:auto;padding-top:16px;border-top:1px solid #e5e3de}
.credits{font-size:12px;color:#999}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.header{padding:16px 32px;border-bottom:1px solid #e5e3de;background:#fff;display:flex;align-items:center;justify-content:space-between}
.project-title{font-size:20px;font-weight:600}
.content{flex:1;overflow-y:auto;padding:24px 32px}
.content-inner{max-width:800px;margin:0 auto}
.section-label{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#555;margin-bottom:10px}
/* Chat - Fixed Bottom */
.chat-box-fixed{position:fixed;bottom:0;left:250px;right:340px;background:#fff;border-top:2px solid #e5e3de;padding:6px 32px;z-index:100;box-shadow:0 -2px 10px rgba(0,0,0,.05);transition:left .2s ease,right .2s ease}
.chat-box-fixed:focus-within{border-top-color:#6366f1}
.chat-box-fixed-inner{max-width:800px;margin:0 auto}
.chat-input{width:100%;border:none;outline:none;font-size:13px;resize:none;min-height:20px;max-height:100px;font-family:inherit;line-height:1.4}
.chat-input::placeholder{color:#aaa;font-size:12px}
.chat-footer{display:flex;justify-content:space-between;align-items:center;margin-top:2px}
.send-btn{width:32px;height:32px;border-radius:50%;border:none;background:#6366f1;color:#fff;cursor:pointer;font-size:14px;transition:background .15s}
.send-btn:hover{background:#4f46e5}
.send-btn:disabled{background:#ccc;cursor:not-allowed}
/* 채팅 첨부파일 */
.attach-btn{width:32px;height:32px;border-radius:50%;border:none;background:transparent;color:#999;cursor:pointer;font-size:18px;transition:all .15s;display:flex;align-items:center;justify-content:center}
.attach-btn:hover{background:#f0f0f0;color:#6366f1}
.chat-dropdown{padding:3px 6px;border-radius:8px;border:1.5px solid #e5e3de;font-size:11px;cursor:pointer;background:#fff;color:#555;outline:none;transition:all .15s;appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3E%3Cpath d='M0 2l4 4 4-4z' fill='%23999'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 6px center;padding-right:18px}
.chat-dropdown:hover{border-color:#6366f1}.chat-dropdown:focus{border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.15)}
.chat-dropdown option{font-size:12px}
.chat-attach-preview{display:flex;flex-wrap:wrap;gap:6px;padding:4px 0}
.chat-attach-badge{display:inline-flex;align-items:center;gap:4px;background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:3px 10px;font-size:11px;color:#4f46e5;max-width:200px}
.chat-attach-badge .fname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chat-attach-badge .remove-attach{cursor:pointer;font-size:14px;color:#999;margin-left:2px;line-height:1}
.chat-attach-badge .remove-attach:hover{color:#ef4444}
.content{padding-bottom:80px!important}
/* Tags */
.tag-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.tag{padding:8px 18px;border-radius:20px;border:2px solid #e5e3de;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;background:#fff;user-select:none}
.tag:hover{border-color:#6366f1}.tag.selected{border-color:var(--c,#6366f1);background:var(--bg,#eef2ff);color:var(--fg,#4f46e5)}
/* Skills */
.skill-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin-bottom:24px;max-height:220px;overflow-y:auto;padding:4px}
.skill-card{padding:10px 14px;border-radius:10px;border:2px solid #e5e3de;background:#fff;cursor:pointer;transition:all .15s;font-size:13px;position:relative}
.skill-card:hover{border-color:#6366f1;transform:translateY(-1px)}
.skill-card.selected{border-color:#6366f1;background:#eef2ff}
.skill-card.auto-loaded{border-color:#f59e0b;background:#fffbeb;animation:autoPulse 1.5s ease-in-out}
@keyframes autoPulse{0%{box-shadow:0 0 0 0 rgba(245,158,11,.4)}70%{box-shadow:0 0 0 8px rgba(245,158,11,0)}100%{box-shadow:none}}
.skill-card.unavailable{opacity:.45;cursor:default}
.skill-card .sn{font-weight:600;margin-bottom:2px}.skill-card .sd{font-size:11px;color:#888}
.skill-card .badge{position:absolute;top:6px;right:8px;font-size:10px;background:#10b981;color:#fff;padding:1px 6px;border-radius:8px}
/* Effort */
.effort-section{margin-bottom:24px}
.effort-slider{width:100%;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:linear-gradient(to right,#6366f1 0%,#6366f1 var(--val,66%),#e5e3de var(--val,66%),#e5e3de 100%);outline:none}
.effort-slider::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:#6366f1;cursor:pointer;border:3px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.2)}
.effort-labels{display:flex;justify-content:space-between;font-size:12px;color:#999;margin-top:4px}
.effort-labels span.active{color:#6366f1;font-weight:600}
/* Style */
.style-row{display:flex;gap:16px;margin-bottom:24px}.style-group{flex:1}
.style-group label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#999;margin-bottom:8px;display:block}
.style-group textarea{width:100%;border:1px solid #e5e3de;border-radius:10px;padding:10px 12px;font-size:13px;resize:none;height:56px;font-family:inherit;outline:none}
.style-group textarea:focus{border-color:#6366f1}
.fmt-btns{display:flex;flex-wrap:wrap;gap:6px}
.fmt-btn{padding:6px 14px;border-radius:8px;border:2px solid #e5e3de;font-size:12px;cursor:pointer;transition:all .15s;background:#fff}
.fmt-btn:hover{border-color:#6366f1}.fmt-btn.selected{background:#6366f1;color:#fff;border-color:#6366f1}
/* Quick */
.quick-row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:20px}
.quick-btn{padding:8px 16px;border-radius:20px;border:1px solid #e5e3de;background:#fff;font-size:13px;cursor:pointer;transition:all .15s}
.quick-btn:hover{border-color:#6366f1;background:#eef2ff}
/* Auto Skill */
.auto-skill-toggle{display:flex;align-items:center;gap:8px;margin-bottom:12px;padding:8px 14px;background:#f8f7ff;border:2px solid #e5e3de;border-radius:12px;cursor:pointer;transition:all .15s;user-select:none}
.auto-skill-toggle:hover{border-color:#6366f1}
.auto-skill-toggle.on{border-color:#6366f1;background:#eef2ff}
.auto-skill-toggle .toggle-dot{width:36px;height:20px;border-radius:10px;background:#ccc;position:relative;transition:all .2s}
.auto-skill-toggle.on .toggle-dot{background:#6366f1}
.auto-skill-toggle .toggle-dot::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:#fff;transition:all .2s}
.auto-skill-toggle.on .toggle-dot::after{left:18px}
.auto-skill-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#eef2ff;color:#6366f1;border:1px solid #c7d2fe;margin:2px}
.auto-skill-preview{padding:6px 10px;background:#f0fdf4;border-bottom:1px solid #a7f3d0;font-size:11px;display:none}
.auto-skill-preview.show{display:block}
/* PPT Suggest Banner */
.ppt-suggest-banner{position:relative;margin:0 auto 8px;max-width:760px;padding:14px 18px 14px 50px;background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 50%,#fdf2f8 100%);border:1px solid #c7d2fe;border-radius:14px;box-shadow:0 2px 12px rgba(99,102,241,.12);overflow:hidden;animation:pptBannerSlide .5s cubic-bezier(.16,1,.3,1);cursor:default;transition:opacity .4s,transform .4s}
.ppt-suggest-banner.hiding{opacity:0;transform:translateY(-12px)}
.ppt-suggest-banner .ppt-banner-icon{position:absolute;left:14px;top:50%;transform:translateY(-50%);font-size:22px;animation:pptIconPulse 2s ease-in-out infinite}
.ppt-suggest-banner .ppt-banner-title{font-size:13px;font-weight:700;color:#4338ca;margin-bottom:4px}
.ppt-suggest-banner .ppt-banner-hint{font-size:12px;color:#6b7280;line-height:1.5}
.ppt-suggest-banner .ppt-banner-hint em{font-style:normal;color:#6366f1;font-weight:600;cursor:pointer;border-bottom:1px dashed #a5b4fc;transition:color .2s}
.ppt-suggest-banner .ppt-banner-hint em:hover{color:#4338ca}
.ppt-suggest-banner .ppt-banner-close{position:absolute;top:6px;right:10px;background:none;border:none;font-size:14px;color:#a5b4fc;cursor:pointer;padding:2px 6px;border-radius:4px;transition:all .2s}
.ppt-suggest-banner .ppt-banner-close:hover{background:#e0e7ff;color:#4338ca}
@keyframes pptBannerSlide{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pptIconPulse{0%,100%{transform:translateY(-50%) scale(1)}50%{transform:translateY(-50%) scale(1.15)}}
/* Messages */
.messages{margin-top:8px}
.msg{margin-bottom:8px;padding:8px 12px;border-radius:10px;line-height:1.5;font-size:13px;word-wrap:break-word;position:relative}
.msg.user{background:#6366f1;color:#fff;margin-left:120px;border-bottom-right-radius:4px;white-space:pre-wrap}
.msg.assistant{background:#fff;border:1px solid #e5e3de;margin-right:120px;border-bottom-left-radius:4px}
.msg pre{background:#f5f5f0;padding:8px;border-radius:6px;overflow-x:auto;margin:4px 0;font-size:12px;white-space:pre-wrap}
.msg code{font-family:'SF Mono','Fira Code',monospace;font-size:12px}
.msg p{margin:0 0 4px 0}.msg p:last-child{margin-bottom:0}
.msg h1,.msg h2,.msg h3,.msg h4{margin:8px 0 4px 0;font-weight:700}
.msg h1{font-size:1.2em;border-bottom:1px solid #e5e3de;padding-bottom:2px}
.msg h2{font-size:1.1em;border-bottom:1px solid #eee;padding-bottom:2px}
.msg h3{font-size:1em}.msg h4{font-size:.95em}
.msg ul,.msg ol{margin:4px 0 4px 16px;padding:0}.msg li{margin:1px 0}
.msg table{border-collapse:collapse;margin:4px 0;width:100%;font-size:12px}
.msg th,.msg td{border:1px solid #ddd;padding:4px 8px;text-align:left}
.msg th{background:#f5f5f0;font-weight:600}
.msg tr:nth-child(even){background:#fafaf8}
.msg hr{border:none;border-top:1px solid #e5e3de;margin:6px 0}
.msg blockquote{border-left:3px solid #6366f1;margin:4px 0;padding:2px 8px;color:#666;background:#fafaf8;border-radius:0 6px 6px 0}
.msg-label{font-size:10px;font-weight:600;color:#999;margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}
/* 피드백 버튼 */
.msg-del{position:absolute;top:6px;right:6px;background:rgba(255,255,255,.8);border:1px solid #ddd;color:#aaa;cursor:pointer;font-size:10px;padding:1px 6px;border-radius:4px;opacity:0;transition:opacity .15s;z-index:5}
.msg:hover .msg-del{opacity:1}
.msg-del:hover{color:#ef4444;border-color:#ef4444;background:#fef2f2}
.msg-feedback{display:flex;gap:4px;margin-top:6px;justify-content:flex-end}
.msg-feedback button{background:none;border:1px solid #e0e0e0;border-radius:6px;padding:3px 10px;font-size:14px;cursor:pointer;color:#999;transition:all .2s;line-height:1}
.msg-feedback button:hover{background:#f5f5f0;color:#333;border-color:#ccc}
.msg-feedback button.fb-selected-good{background:#e8f5e9;color:#2e7d32;border-color:#81c784}
.msg-feedback button.fb-selected-bad{background:#fce4ec;color:#c62828;border-color:#ef9a9a}
/* 피드백 모달 */
#feedbackModal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;background:rgba(0,0,0,.5);align-items:center;justify-content:center}
#feedbackModal.show{display:flex}
.fb-modal{background:#fff;border-radius:14px;padding:28px;width:420px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,.3);animation:fbSlideIn .25s ease}
@keyframes fbSlideIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.fb-modal h3{margin:0 0 6px;font-size:18px;color:#333}
.fb-modal .fb-rating-display{font-size:24px;margin-bottom:12px}
.fb-modal textarea{width:100%;height:100px;border:1.5px solid #ddd;border-radius:10px;padding:10px 12px;font-size:13px;resize:vertical;font-family:inherit;transition:border-color .2s}
.fb-modal textarea:focus{outline:none;border-color:#6366f1}
.fb-modal .fb-hint{font-size:11px;color:#999;margin-top:4px}
.fb-modal .fb-actions{display:flex;gap:8px;margin-top:16px;justify-content:flex-end}
.fb-modal .fb-btn{padding:8px 20px;border-radius:8px;border:none;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
.fb-modal .fb-btn-cancel{background:#f0f0f0;color:#666}.fb-modal .fb-btn-cancel:hover{background:#e0e0e0}
.fb-modal .fb-btn-submit{background:#6366f1;color:#fff}.fb-modal .fb-btn-submit:hover{background:#4f46e5}
.msg.user .msg-label{color:rgba(255,255,255,.7)}
.msg .skill-info{font-size:10px;color:#6366f1;margin-top:4px}
.typing{display:inline-flex;gap:4px;padding:8px 14px}
.typing span{width:8px;height:8px;border-radius:50%;background:#ccc;animation:blink 1.4s infinite both}
.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.3}40%{opacity:1}}
/* Modal */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:100;display:flex;align-items:center;justify-content:center}
.modal-bg.hidden{display:none}
.modal{background:#fff;border-radius:16px;padding:28px;width:500px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,.15)}
.modal h2{font-size:18px;margin-bottom:16px}
.modal label{font-size:13px;font-weight:600;color:#555;display:block;margin-bottom:4px;margin-top:12px}
.modal input{width:100%;padding:10px 12px;border:1px solid #e5e3de;border-radius:8px;font-size:14px;outline:none;font-family:inherit}
.modal input:focus{border-color:#6366f1}
.modal-btns{display:flex;justify-content:flex-end;gap:8px;margin-top:20px}
.modal-btns button{padding:10px 20px;border-radius:8px;border:none;font-size:14px;cursor:pointer}
.btn-ok{background:#6366f1;color:#fff}.btn-ok:hover{background:#4f46e5}
.btn-cancel{background:#f3f2ef;color:#555}
.settings-btn{background:none;border:none;cursor:pointer;font-size:20px;color:#999;padding:4px 8px;border-radius:6px}
.settings-btn:hover{background:#f3f2ef;color:#555}
.status{font-size:12px}
.status.on{color:#10b981}.status.off{color:#999}
/* Env selector */
.env-section{margin-bottom:6px}
.env-section-label{font-size:11px;font-weight:700;color:#6b7280;margin-bottom:3px;padding-left:4px}
.env-toggle{cursor:pointer;user-select:none;padding:6px 8px;border-radius:8px;transition:background .15s}
.env-toggle:hover{background:#f3f4f6}
.env-subsection-label{font-size:10px;color:#9ca3af;margin:3px 0 2px 4px}
.env-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
.env-btn{flex:1 1 90px;max-width:130px;padding:8px 6px;border-radius:10px;border:2px solid #e5e3de;background:#fff;cursor:pointer;text-align:center;transition:all .15s;font-size:11px}
.env-btn:hover{border-color:#6366f1;transform:translateY(-1px)}
.env-btn.selected{border-color:#6366f1;background:#eef2ff}
.env-btn .env-name{font-weight:700;font-size:12px;margin-bottom:1px}
.env-btn .env-model{font-size:10px;color:#888}
.env-btn.selected .env-model{color:#6366f1}
.uio-enter-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 12px;border-radius:8px;font-size:12px;font-weight:600;color:#fff;background:linear-gradient(135deg,#6366f1,#8b5cf6);text-decoration:none;cursor:pointer;transition:all .2s;border:none;box-shadow:0 2px 8px rgba(99,102,241,.3)}
.uio-enter-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(99,102,241,.5);background:linear-gradient(135deg,#818cf8,#a78bfa)}
.token-status{font-size:12px;padding:8px 12px;border-radius:8px;margin-bottom:16px}
.token-status.ok{background:#ecfdf5;color:#059669}
.token-status.missing{background:#fef2f2;color:#dc2626}
/* System Prompt */
.sysprompt-section{margin-bottom:24px}
.sysprompt-toggle{display:flex;align-items:center;gap:8px;cursor:pointer;margin-bottom:8px}
.sysprompt-toggle .arrow{font-size:12px;transition:transform .2s;color:#999}
.sysprompt-toggle .arrow.open{transform:rotate(90deg)}
.sysprompt-body{display:none}
.sysprompt-body.show{display:block}
.sysprompt-textarea{width:100%;border:1px solid #e5e3de;border-radius:10px;padding:12px;font-size:13px;resize:vertical;min-height:80px;max-height:300px;font-family:'SF Mono','Fira Code',monospace;line-height:1.5;outline:none;background:#fafaf8}
.sysprompt-textarea:focus{border-color:#6366f1;background:#fff}
.sysprompt-info{font-size:11px;color:#999;margin-top:4px}
.prompt-chip{display:inline-flex;align-items:center;gap:3px;padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;border:1px solid #e5e3de;background:#fafaf8;transition:all .15s;white-space:nowrap}
.prompt-chip:hover{background:#eef2ff;border-color:#6366f1}
.prompt-chip.active{background:#6366f1;color:#fff;border-color:#6366f1}
.prompt-chip.saved{border-color:#f59e0b;background:#fffbeb}
.prompt-chip.saved:hover{background:#fef3c7}
.prompt-chip.saved.active{background:#f59e0b;color:#fff}
.sysprompt-preview{font-size:11px;color:#6366f1;background:#eef2ff;padding:6px 10px;border-radius:6px;margin-top:6px;max-height:60px;overflow:hidden;cursor:pointer}
.sysprompt-preview:hover{max-height:200px;overflow-y:auto}
/* PPT 생성 블록 */
.pptx-gen-block{background:#f8f7ff;border:2px solid #a5b4fc;border-radius:12px;margin:12px 0;overflow:hidden}
.pptx-gen-header{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#eef2ff;border-bottom:1px solid #a5b4fc}
.pptx-gen-header span{font-weight:600;font-size:13px;color:#4338ca}
.pptx-gen-actions{display:flex;gap:6px;align-items:center}
.pptx-gen-btn{background:#6366f1;color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all .2s;font-weight:600}
.pptx-gen-btn:hover{background:#4f46e5;transform:translateY(-1px)}
.pptx-gen-btn:disabled{background:#d1d5db;cursor:not-allowed;transform:none}
.pptx-gen-btn.secondary{background:#e0e7ff;color:#4338ca}
.pptx-gen-btn.secondary:hover{background:#c7d2fe}
.pptx-gen-status{font-size:11px;color:#6b7280}
.pptx-remake-bar{display:flex;align-items:center;gap:6px;padding:8px 12px;background:#f8fafc;border-top:1px solid #e5e7eb;border-radius:0 0 10px 10px;flex-wrap:wrap}
.pptx-remake-label{font-size:11px;color:#6b7280;font-weight:600;white-space:nowrap}
.pptx-remake-btn{background:#fff;color:#4338ca;border:1px solid #c7d2fe;border-radius:16px;padding:4px 12px;font-size:11px;cursor:pointer;transition:all .2s;font-weight:500;white-space:nowrap}
.pptx-remake-btn:hover{background:#eef2ff;border-color:#818cf8;transform:translateY(-1px)}
/* PPT 스타일 드롭다운 */
.ppt-style-drop{display:none}
.ppt-ref-badge{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:10px;background:#fef3c7;color:#92400e;font-size:10px;font-weight:600;margin-left:4px}
/* CSV Upload */
.csv-section{margin-bottom:24px}
/* csv-upload-area 제거됨 → 채팅 📎 첨부로 대체 */
.csv-upload-area .icon{font-size:28px;margin-bottom:6px}
.csv-upload-area .label{font-size:13px;color:#888}
.csv-upload-area .sub{font-size:11px;color:#bbb;margin-top:2px}
.csv-info{background:#fff;border:2px solid #10b981;border-radius:12px;padding:14px 18px;position:relative}
.csv-info .fname{font-weight:700;font-size:14px;color:#059669}
.csv-info .fstats{font-size:12px;color:#666;margin-top:4px}
.csv-info .fremove{position:absolute;top:10px;right:12px;background:#fee2e2;color:#dc2626;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer}
.csv-info .fremove:hover{background:#fca5a5}
.csv-preview{margin-top:8px;max-height:160px;overflow:auto;font-size:11px;background:#f9fafb;border-radius:8px;padding:8px}
.csv-preview table{border-collapse:collapse;width:100%}
.csv-preview th,.csv-preview td{border:1px solid #e5e7eb;padding:3px 8px;text-align:left;white-space:nowrap}
.csv-preview th{background:#f3f4f6;font-weight:600;position:sticky;top:0}
/* Skill Detail Panel */
.skill-detail-overlay{position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:200;display:flex;align-items:center;justify-content:center}
.skill-detail-panel{background:#fff;border-radius:16px;width:720px;max-width:92vw;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.2)}
.sdp-header{padding:18px 24px;border-bottom:1px solid #e5e3de;display:flex;justify-content:space-between;align-items:center}
.sdp-header h2{font-size:18px;margin:0}.sdp-close{background:none;border:none;font-size:22px;cursor:pointer;color:#999;padding:4px 8px}
.sdp-close:hover{color:#333}
.sdp-tabs{display:flex;border-bottom:1px solid #e5e3de;padding:0 24px}
.sdp-tab{padding:10px 18px;font-size:13px;font-weight:600;cursor:pointer;border-bottom:3px solid transparent;color:#888;transition:all .15s}
.sdp-tab:hover{color:#333}.sdp-tab.active{color:#6366f1;border-bottom-color:#6366f1}
.sdp-body{flex:1;overflow-y:auto;padding:18px 24px}
.sdp-file-list{list-style:none;padding:0}
.sdp-file-item{display:flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid #e5e3de;border-radius:10px;margin-bottom:6px;transition:all .15s;cursor:pointer}
.sdp-file-item:hover{border-color:#6366f1;background:#f8f7ff}
.sdp-file-icon{font-size:18px}
.sdp-file-name{font-weight:600;font-size:13px;flex:1}
.sdp-file-size{font-size:11px;color:#999}
.sdp-file-actions{display:flex;gap:4px}
.sdp-btn{padding:4px 12px;border-radius:6px;border:1px solid #e5e3de;font-size:11px;cursor:pointer;background:#fff;transition:all .15s}
.sdp-btn:hover{border-color:#6366f1;background:#eef2ff}
.sdp-btn.run{background:#10b981;color:#fff;border-color:#10b981}.sdp-btn.run:hover{background:#059669}
.sdp-btn.inject{background:#6366f1;color:#fff;border-color:#6366f1}.sdp-btn.inject:hover{background:#4f46e5}
.sdp-code{background:#1e1e1e;color:#d4d4d4;padding:14px;border-radius:10px;font-family:'SF Mono','Fira Code',monospace;font-size:12px;line-height:1.5;overflow:auto;max-height:400px;white-space:pre-wrap;word-break:break-all}
.sdp-result{margin-top:12px;padding:12px;border-radius:10px;font-family:monospace;font-size:12px;line-height:1.4;overflow:auto;max-height:300px;white-space:pre-wrap}
.sdp-result.ok{background:#ecfdf5;border:1px solid #a7f3d0}.sdp-result.err{background:#fef2f2;border:1px solid #fca5a5}
.skill-card .extras{font-size:10px;color:#6366f1;margin-top:3px}
/* Copy button */
.copy-btn{position:absolute;top:4px;right:4px;background:#e5e3de;border:none;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;opacity:.7}
.copy-btn:hover{opacity:1}
.drawio-block{border:2px solid #6366f1;border-radius:10px;overflow:hidden;margin:10px 0;background:#fafaff}
.drawio-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#eef2ff;border-bottom:1px solid #d4d8f0;flex-wrap:wrap;gap:6px}
.drawio-header span{font-weight:600;font-size:13px;color:#4338ca}
.drawio-actions{display:flex;gap:6px;flex-wrap:wrap}
.drawio-btn{background:#6366f1;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.drawio-btn:hover{background:#4f46e5}
.chart-block{border:2px solid #f59e0b;border-radius:10px;overflow:hidden;margin:10px 0;background:#fffbf0}
.chart-header{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#fef3c7;border-bottom:1px solid #fcd34d;flex-wrap:wrap;gap:6px}
.chart-header span{font-weight:600;font-size:13px;color:#92400e}
.chart-actions{display:flex;gap:6px;flex-wrap:wrap}
.chart-btn{background:#f59e0b;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.chart-btn:hover{background:#d97706}
.chart-canvas-wrap{padding:12px;background:#fff;display:flex;justify-content:center;align-items:center;min-height:250px;max-height:450px}
.chart-canvas-wrap canvas{max-width:100%;max-height:420px}
.chart-raw-toggle{padding:0 12px 8px}
.chart-raw-toggle summary{font-size:11px;color:#999;cursor:pointer}
.chart-raw-toggle pre{font-size:10px;max-height:120px;overflow:auto;background:#f5f5f0;padding:8px;border-radius:6px;margin-top:4px}
.md-report-block{background:#f8faf8;border:2px solid #86efac;border-radius:12px;margin:12px 0;overflow:hidden}
.md-report-header{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#ecfdf5;border-bottom:1px solid #86efac}
.md-report-header span{font-weight:600;font-size:13px;color:#166534}
.md-report-actions{display:flex;gap:6px}
.md-report-btn{background:#22c55e;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .2s}
.md-report-btn:hover{background:#16a34a}
.md-report-btn.md-download{background:#6366f1}.md-report-btn.md-download:hover{background:#4f46e5}
.md-report-preview{padding:14px 18px;font-size:13px;line-height:1.7;max-height:500px;overflow-y:auto}
.md-report-preview h1{font-size:20px;margin:12px 0 8px;border-bottom:2px solid #e5e7eb;padding-bottom:4px}
.md-report-preview h2{font-size:17px;margin:10px 0 6px;color:#374151}
.md-report-preview h3{font-size:15px;margin:8px 0 4px;color:#4b5563}
.md-report-preview table{border-collapse:collapse;width:100%;margin:8px 0}
.md-report-preview th,.md-report-preview td{border:1px solid #d1d5db;padding:4px 8px;font-size:12px;text-align:left}
.md-report-preview th{background:#f3f4f6;font-weight:600}
.md-report-preview pre{background:#1e1e2e;color:#cdd6f4;padding:10px;border-radius:8px;overflow-x:auto;font-size:12px}
.md-report-preview blockquote{border-left:3px solid #6366f1;padding-left:12px;color:#6b7280;margin:8px 0}
.md-raw-toggle{margin:0;padding:4px 14px 8px}
.md-raw-toggle summary{font-size:11px;color:#9ca3af;cursor:pointer}
.md-raw-toggle pre{max-height:300px;overflow:auto;font-size:11px;margin-top:6px}
.drawio-preview{max-height:200px;overflow:auto;margin:0;border-radius:0;border:none;font-size:11px}
.drawio-preview code{font-size:11px}
.send-btn.stop-mode{background:#e74c3c;animation:pulse-stop 1.2s infinite}
@keyframes pulse-stop{0%,100%{opacity:1}50%{opacity:.7}}
.sidebar-file-panel{margin-top:auto;padding:8px 8px 0;border-top:1px solid #e5e3de;max-height:220px;overflow-y:auto}
.uploaded-files-header{font-size:11px;font-weight:600;color:#555;padding:4px 0;border-bottom:1px solid #eee;margin-bottom:2px;display:flex;align-items:center;justify-content:space-between}
.uploaded-file-item{display:flex;align-items:center;gap:4px;padding:3px 2px;border-bottom:1px solid #f0f0f0;font-size:11px}
.ufi-icon{font-size:14px;flex-shrink:0}
.ufi-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#333;cursor:help}
.ufi-size{color:#999;font-size:10px;flex-shrink:0}
.ufi-remove{background:none;border:none;color:#ccc;cursor:pointer;font-size:12px;padding:0 2px}
.ufi-remove:hover{color:#e74c3c}
.ufi-pending{opacity:.6}
.ufi-drm{opacity:.7;border-left:2px solid #e67e22;padding-left:4px}
.ufi-drm .ufi-name{color:#e67e22}
.think-box{margin:8px 0 12px;border:1px solid #d4c8f0;border-radius:8px;background:#f8f5ff;overflow:hidden}
.think-box summary{cursor:pointer;padding:8px 12px;font-size:12px;font-weight:600;color:#7c5cbf;background:#f0ebfa;user-select:none}
.think-box summary:hover{background:#e8e0f6}
.think-box[open] summary{border-bottom:1px solid #d4c8f0}
.think-content{padding:10px 14px;font-size:12px;color:#555;line-height:1.6;max-height:400px;overflow-y:auto}
pre{position:relative}
/* ===== 오른쪽 코드 어시스턴트 패널 ===== */
.right-panel{width:340px;background:#fff;border-left:1px solid #e5e3de;display:flex;flex-direction:column;overflow:hidden;transition:width .2s ease;flex-shrink:0}
.right-panel.collapsed{width:0;border-left:none;overflow:hidden}
.right-panel-toggle{position:fixed;top:12px;right:340px;width:28px;height:28px;border-radius:6px 0 0 6px;border:1px solid #e5e3de;border-right:none;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;color:#999;z-index:200;transition:right .2s ease}
.right-panel-toggle:hover{background:#eef2ff;color:#6366f1}
body.rp-collapsed .right-panel-toggle{right:0}
body.rp-collapsed .chat-box-fixed{right:0}
.rp-header{padding:14px 16px;border-bottom:1px solid #e5e3de;display:flex;align-items:center;justify-content:space-between}
.rp-header h3{margin:0;font-size:15px;font-weight:700;color:#1a1a1a}
.rp-body{flex:1;overflow-y:auto;padding:12px 14px}
.rp-section{margin-bottom:16px}
.rp-section-title{font-size:11px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.rp-file-drop{border:2px dashed #d1d5db;border-radius:10px;padding:24px 16px;text-align:center;cursor:pointer;transition:all .15s;background:#fafaf9;margin-bottom:12px}
.rp-file-drop:hover,.rp-file-drop.dragover{border-color:#6366f1;background:#eef2ff}
.rp-file-drop-icon{font-size:28px;margin-bottom:6px}
.rp-file-drop-text{font-size:12px;color:#888}
.rp-file-list{max-height:120px;overflow-y:auto;margin-bottom:8px}
.rp-file-item{display:flex;align-items:center;gap:6px;padding:5px 8px;background:#f8f7f4;border-radius:6px;margin-bottom:4px;font-size:12px}
.rp-file-item .fname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#333}
.rp-file-item .fsize{color:#aaa;font-size:11px;flex-shrink:0}
.rp-file-item .fremove{background:none;border:none;cursor:pointer;color:#ef4444;font-size:12px;padding:0 2px}
.rp-action-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.rp-action-btn{padding:10px 8px;border-radius:8px;border:1px solid #e5e3de;background:#fff;cursor:pointer;text-align:center;transition:all .15s;font-size:12px}
.rp-action-btn:hover{border-color:#6366f1;background:#eef2ff}
.rp-action-btn.active{border-color:#6366f1;background:#6366f1;color:#fff}
.rp-action-btn .rp-btn-icon{font-size:18px;display:block;margin-bottom:3px}
.rp-action-btn .rp-btn-label{font-weight:600}
.rp-run-btn{width:100%;padding:10px;border-radius:8px;border:none;background:#6366f1;color:#fff;font-size:14px;font-weight:700;cursor:pointer;margin-top:10px;transition:background .15s}
.rp-run-btn:hover{background:#4f46e5}
.rp-run-btn:disabled{background:#d1d5db;cursor:not-allowed}
.rp-result{margin-top:12px;padding:16px;background:#1e1e2e;color:#cdd6f4;border-radius:10px;font-size:13px;line-height:1.7;overflow-y:auto;display:none}
.rp-result.expanded{max-height:none;position:fixed;top:8px;right:8px;bottom:8px;width:calc(var(--rp-width, 340px) - 16px);z-index:1100;box-shadow:0 8px 32px rgba(0,0,0,.4)}
.rp-result h1,.rp-result h2,.rp-result h3{color:#cba6f7;margin:12px 0 6px}
.rp-result h1{font-size:16px} .rp-result h2{font-size:14px} .rp-result h3{font-size:13px}
.rp-result pre{background:#181825;border-radius:6px;padding:10px;overflow-x:auto;margin:8px 0}
.rp-result code{font-family:'Fira Code',monospace;font-size:12px}
.rp-result p{margin:6px 0}
.rp-result ul,.rp-result ol{padding-left:20px;margin:6px 0}
.rp-result table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}
.rp-result th,.rp-result td{border:1px solid #45475a;padding:6px 8px;text-align:left}
.rp-result th{background:#313244;color:#cba6f7}
.rp-result-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #45475a}
.rp-result-header .rp-result-title{font-weight:700;color:#a6e3a1;font-size:14px}
.rp-result-header button{background:none;border:1px solid #45475a;color:#cdd6f4;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px}
.rp-result-header button:hover{background:#313244}
.rp-skill-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:#eef2ff;color:#6366f1;margin-left:6px}
.rp-tree{font-size:12px;max-height:250px;overflow-y:auto;border:1px solid #e5e3de;border-radius:8px;padding:8px;background:#fafaf9;margin-bottom:8px}
.rp-tree-node{padding:2px 0;cursor:pointer;user-select:none;display:flex;align-items:center;gap:2px}
.rp-tree-cb{width:14px;height:14px;cursor:pointer;accent-color:#6366f1;flex-shrink:0}
.rp-select-bar{display:flex;gap:6px;margin-bottom:6px;align-items:center}
.rp-select-btn{background:none;border:1px solid #d1d5db;border-radius:4px;padding:2px 8px;font-size:10px;cursor:pointer;color:#666}
.rp-select-btn:hover{border-color:#6366f1;color:#6366f1}
.rp-checked-count{font-size:11px;color:#6366f1;font-weight:600}
.rp-tree-node:hover{background:#eef2ff;border-radius:4px}
.rp-tree-node-content{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rp-tree-folder{font-weight:600;color:#555}
.rp-tree-file{color:#333}
.rp-tree-file.active{background:#6366f1;color:#fff;border-radius:4px;padding:1px 4px}
.rp-tree-children{padding-left:16px;border-left:1px solid #e5e3de;margin-left:6px}
.rp-tree-toggle{display:inline-block;width:14px;font-size:10px;color:#999;text-align:center}
.rp-tree-icon{margin-right:4px}
.rp-tree-stats{font-size:11px;color:#888;padding:6px 8px;background:#f0f0ed;border-radius:6px;margin-bottom:8px;display:flex;justify-content:space-between}
.rp-tab-row{display:flex;gap:4px;margin-bottom:8px}
.rp-tab{padding:4px 10px;border-radius:6px;border:1px solid #e5e3de;background:#fff;cursor:pointer;font-size:11px;font-weight:600;color:#888;transition:all .15s}
.rp-tab:hover{border-color:#6366f1;color:#6366f1}
.rp-tab.active{background:#6366f1;color:#fff;border-color:#6366f1}
@media(max-width:768px){.sidebar{display:none}.sidebar-toggle{display:none}.right-panel{display:none}.right-panel-toggle{display:none}.chat-box-fixed{left:0;right:0}.style-row{flex-direction:column}.msg.user{margin-left:16px}.msg.assistant{margin-right:16px}.msg{font-size:12px}}
/* 인트로 비디오 오버레이 */
/* 인트로 제거됨 */
#introVideo{width:100%;height:100%;object-fit:cover}
#introSkip{position:absolute;bottom:40px;right:40px;background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);color:#fff;padding:10px 24px;border-radius:24px;font-size:14px;cursor:pointer;transition:all .2s;z-index:10000}
#introSkip:hover{background:rgba(255,255,255,.3)}
#introProgress{position:absolute;bottom:0;left:0;height:3px;background:linear-gradient(90deg,#6366f1,#a855f7);width:0;transition:width .1s linear}
/* UIO 컨테이너 */
#uioContainer{position:fixed;top:0;left:0;width:100%;height:100%;z-index:9000;display:none;background:#0f151e}
#uioContainer iframe{width:100%;height:100%;border:none}
#uioBackBtn{position:fixed;top:16px;left:16px;z-index:9001;padding:10px 20px;border-radius:10px;border:2px solid rgba(255,255,255,.25);background:rgba(18,27,40,.85);backdrop-filter:blur(8px);color:#fff;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
#uioBackBtn:hover{background:rgba(99,102,241,.3);border-color:#6366f1}
</style>
</head>
<body>
<!-- 피드백 모달 -->
<div id="feedbackModal">
  <div class="fb-modal">
    <h3 id="fbModalTitle">피드백</h3>
    <div class="fb-rating-display" id="fbRatingDisplay"></div>
    <textarea id="fbComment" placeholder="어떤 점이 좋았나요? 또는 어떤 점을 개선하면 좋을까요? (선택사항)"></textarea>
    <div class="fb-hint">피드백은 서비스 개선에 활용됩니다</div>
    <div class="fb-actions">
      <button class="fb-btn fb-btn-cancel" onclick="closeFeedbackModal()">취소</button>
      <button class="fb-btn fb-btn-submit" onclick="submitFeedback()">보내기</button>
    </div>
  </div>
</div>

<!-- 인트로 비디오 오버레이 -->
<!-- 인트로 제거됨 - 바로 로그인 -->

<!-- 소개 팝업: /beno/about.html로 분리됨 (openAboutPopup() 함수로 열기) -->
<div id="aboutPopup_removed" style="display:none;">
  <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);border:1px solid rgba(99,102,241,0.3);border-radius:24px;padding:0;width:90%;max-width:800px;max-height:90vh;overflow-y:auto;box-shadow:0 25px 60px rgba(0,0,0,0.6);scrollbar-width:none;">
    <style>#aboutPopup *::-webkit-scrollbar{display:none}</style>
    <!-- 헤더 -->
    <div style="position:sticky;top:0;background:linear-gradient(135deg,#0f172a,#1e1b4b);padding:24px 32px 16px;border-bottom:1px solid rgba(99,102,241,0.15);z-index:1;display:flex;justify-content:space-between;align-items:center;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:48px;height:48px;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);border-radius:14px;display:flex;align-items:center;justify-content:center;">
          <span style="font-size:22px;color:#fff;font-weight:900;">D</span>
        </div>
        <div>
          <h2 style="margin:0;color:#fff;font-size:22px;font-weight:800;">Demos V1.0</h2>
          <p style="margin:0;color:#64748b;font-size:12px;">AI Scientific Assistant Platform</p>
        </div>
      </div>
      <button onclick="document.getElementById('aboutPopup').style.display='none'" style="background:rgba(255,255,255,0.1);border:none;color:#fff;width:36px;height:36px;border-radius:10px;font-size:18px;cursor:pointer;">✕</button>
    </div>
    <!-- 본문 -->
    <div style="padding:24px 32px 32px;">
      <!-- 소개 배너 -->
      <div style="text-align:center;padding:32px 0;margin-bottom:24px;">
        <h1 style="color:#fff;font-size:32px;font-weight:900;margin:0 0 8px;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Next-Gen AI Platform</h1>
        <p style="color:#94a3b8;font-size:15px;margin:0;">355+ 전문 스킬 / 하네스 파이프라인 / 멀티에이전트 시스템</p>
      </div>
      <!-- 핵심 기능 그리드 -->
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:28px;">
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">🧬</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">355+ 전문 스킬</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">생물정보학, 화학, 데이터/ML, 금융, 임상의학, 반도체 등 10개+ 도메인을 커버하는 전문 스킬</p>
        </div>
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">🤖</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">하네스 파이프라인</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">Expert Pool 동적 에이전트 선택 + 피드백 루프로 사용할수록 품질 향상</p>
        </div>
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">👔</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">CEO 멀티에이전트</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">병렬 에이전트 실행 + 자체검토 + 크로스리뷰 + CEO 자동 검토/반려/재작업</p>
        </div>
        <div style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">📚</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">개인 지식 도메인</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">사용자별 지식 등록/검색. 프론트매터 자동 생성, 카테고리/태그 기반 정확한 검색</p>
        </div>
        <div style="background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">📝</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">보고서 자동 생성</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">PPT / Draw.io / 차트 / MD / HTML 문서 자동 생성 및 다운로드</p>
        </div>
        <div style="background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);border-radius:16px;padding:20px;">
          <div style="font-size:32px;margin-bottom:10px;">🔍</div>
          <h3 style="color:#fff;font-size:15px;margin:0 0 6px;">로그프레소 연동</h3>
          <p style="color:#94a3b8;font-size:12px;margin:0;line-height:1.5;">LPQL 쿼리 자동 생성 + 서버 직접 조회. 자연어로 데이터 검색</p>
        </div>
      </div>
      <!-- 스킬 도메인 -->
      <div style="margin-bottom:28px;">
        <h3 style="color:#fff;font-size:16px;font-weight:700;margin:0 0 14px;">스킬 도메인</h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
          <span style="padding:6px 14px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);border-radius:20px;color:#fca5a5;font-size:12px;">🧬 생물정보학</span>
          <span style="padding:6px 14px;background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.2);border-radius:20px;color:#67e8f9;font-size:12px;">⚗️ 화학/신약</span>
          <span style="padding:6px 14px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);border-radius:20px;color:#fcd34d;font-size:12px;">📊 데이터/ML</span>
          <span style="padding:6px 14px;background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);border-radius:20px;color:#c4b5fd;font-size:12px;">⚛️ 재료/물리</span>
          <span style="padding:6px 14px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);border-radius:20px;color:#6ee7b7;font-size:12px;">💰 금융/경제</span>
          <span style="padding:6px 14px;background:rgba(236,72,153,0.1);border:1px solid rgba(236,72,153,0.2);border-radius:20px;color:#f9a8d4;font-size:12px;">🏥 임상/의학</span>
          <span style="padding:6px 14px;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:20px;color:#93c5fd;font-size:12px;">📝 논문/연구</span>
          <span style="padding:6px 14px;background:rgba(20,184,166,0.1);border:1px solid rgba(20,184,166,0.2);border-radius:20px;color:#5eead4;font-size:12px;">🤖 랩 자동화</span>
          <span style="padding:6px 14px;background:rgba(107,114,128,0.1);border:1px solid rgba(107,114,128,0.2);border-radius:20px;color:#d1d5db;font-size:12px;">🔧 유틸리티</span>
        </div>
      </div>
      <!-- 기술 스택 -->
      <div style="margin-bottom:28px;">
        <h3 style="color:#fff;font-size:16px;font-weight:700;margin:0 0 14px;">기술 스택</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
          <div style="text-align:center;padding:14px;background:rgba(255,255,255,0.03);border-radius:12px;">
            <div style="font-size:24px;margin-bottom:4px;">🐍</div>
            <div style="color:#e2e8f0;font-size:12px;font-weight:600;">Python Flask</div>
          </div>
          <div style="text-align:center;padding:14px;background:rgba(255,255,255,0.03);border-radius:12px;">
            <div style="font-size:24px;margin-bottom:4px;">💻</div>
            <div style="color:#e2e8f0;font-size:12px;font-weight:600;">GGUF Local LLM</div>
          </div>
          <div style="text-align:center;padding:14px;background:rgba(255,255,255,0.03);border-radius:12px;">
            <div style="font-size:24px;margin-bottom:4px;">🔗</div>
            <div style="color:#e2e8f0;font-size:12px;font-weight:600;">Multi-API</div>
          </div>
        </div>
      </div>
      <!-- 하단 -->
      <div style="text-align:center;padding-top:16px;border-top:1px solid rgba(255,255,255,0.05);">
        <p style="color:#475569;font-size:12px;margin:0;">&copy; 2026 Demos Platform &middot; Harness Integrated &middot; Built with Claude Opus 4.6</p>
      </div>
    </div>
  </div>
</div>

<!-- 로그인 오버레이 -->
<div id="loginOverlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:9998;background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);align-items:center;justify-content:center;overflow:hidden;">
  <!-- 배경 애니메이션 -->
  <div style="position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;pointer-events:none;">
    <div style="position:absolute;top:20%;left:10%;width:300px;height:300px;background:radial-gradient(circle,rgba(99,102,241,0.15) 0%,transparent 70%);border-radius:50%;animation:loginFloat 8s ease-in-out infinite;"></div>
    <div style="position:absolute;bottom:20%;right:10%;width:400px;height:400px;background:radial-gradient(circle,rgba(168,85,247,0.1) 0%,transparent 70%);border-radius:50%;animation:loginFloat 10s ease-in-out infinite reverse;"></div>
    <div style="position:absolute;top:50%;left:50%;width:200px;height:200px;background:radial-gradient(circle,rgba(34,197,94,0.08) 0%,transparent 70%);border-radius:50%;animation:loginFloat 6s ease-in-out infinite 2s;"></div>
  </div>
  <style>
    @keyframes loginFloat{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-30px) scale(1.05)}}
    @keyframes loginPulse{0%,100%{box-shadow:0 0 20px rgba(99,102,241,0.3)}50%{box-shadow:0 0 40px rgba(99,102,241,0.5)}}
    @keyframes loginOrbit{0%{transform:rotate(0deg) translateX(180px) rotate(0deg)}100%{transform:rotate(360deg) translateX(180px) rotate(-360deg)}}
    @keyframes loginOrbit2{0%{transform:rotate(0deg) translateX(250px) rotate(0deg)}100%{transform:rotate(-360deg) translateX(250px) rotate(360deg)}}
    @keyframes loginSpin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
    @keyframes loginFadeSlide{0%{opacity:0;transform:translateY(20px)}100%{opacity:1;transform:translateY(0)}}
    @keyframes loginTyping{0%,100%{opacity:.3}50%{opacity:1}}
    @keyframes loginLine{0%{transform:translateX(-100%)}100%{transform:translateX(200%)}}
    @keyframes starTwinkle{0%,100%{opacity:.2;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
    #loginOverlay input:focus{border-color:#6366f1!important;box-shadow:0 0 0 3px rgba(99,102,241,0.15)}
    .login-card{animation:loginFadeSlide .8s ease-out}
    .login-particle{position:absolute;border-radius:50%;pointer-events:none}
    .login-star{position:absolute;width:3px;height:3px;background:#fff;border-radius:50%;animation:starTwinkle 2s ease-in-out infinite;pointer-events:none}
    .login-ring{position:absolute;border:1px solid rgba(99,102,241,0.15);border-radius:50%;animation:loginSpin 20s linear infinite;pointer-events:none}
    .login-feature{display:flex;align-items:center;gap:10px;padding:10px 14px;background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.1);border-radius:10px;margin-bottom:6px;animation:loginFadeSlide .6s ease-out backwards}
    .login-feature:nth-child(1){animation-delay:.2s}.login-feature:nth-child(2){animation-delay:.4s}.login-feature:nth-child(3){animation-delay:.6s}.login-feature:nth-child(4){animation-delay:.8s}
  </style>
  <!-- 별 파티클 -->
  <script>(function(){var o=document.getElementById('loginOverlay');if(!o)return;for(var i=0;i<40;i++){var s=document.createElement('div');s.className='login-star';s.style.left=Math.random()*100+'%';s.style.top=Math.random()*100+'%';s.style.animationDelay=Math.random()*3+'s';s.style.animationDuration=(2+Math.random()*3)+'s';o.querySelector('div').appendChild(s);}})();</script>
  <!-- 궤도 링 -->
  <div class="login-ring" style="width:400px;height:400px;top:50%;left:50%;margin:-200px 0 0 -200px;"></div>
  <div class="login-ring" style="width:600px;height:600px;top:50%;left:50%;margin:-300px 0 0 -300px;animation-direction:reverse;animation-duration:30s;border-color:rgba(168,85,247,0.1);"></div>
  <!-- 궤도 도는 작은 구슬들 -->
  <div style="position:absolute;top:50%;left:50%;width:0;height:0;pointer-events:none;">
    <div style="position:absolute;width:8px;height:8px;background:linear-gradient(135deg,#6366f1,#a855f7);border-radius:50%;box-shadow:0 0 12px #6366f1;animation:loginOrbit 12s linear infinite;"></div>
    <div style="position:absolute;width:6px;height:6px;background:linear-gradient(135deg,#22c55e,#10b981);border-radius:50%;box-shadow:0 0 10px #22c55e;animation:loginOrbit2 18s linear infinite;"></div>
    <div style="position:absolute;width:5px;height:5px;background:linear-gradient(135deg,#f59e0b,#ef4444);border-radius:50%;box-shadow:0 0 8px #f59e0b;animation:loginOrbit 15s linear infinite reverse;"></div>
  </div>
  <div style="position:relative;display:flex;align-items:center;justify-content:center;gap:48px;max-width:900px;width:100%;padding:0 24px;">
  <!-- 왼쪽: 피처 소개 -->
  <div style="flex:1;max-width:360px;display:none;" id="loginFeatures">
    <h2 style="color:#fff;font-size:24px;font-weight:800;margin:0 0 6px;">Welcome</h2>
    <p style="color:#64748b;font-size:13px;margin:0 0 16px;">AI Scientific Assistant Platform</p>
    <button onclick="openAboutPopup()" style="display:inline-flex;align-items:center;gap:6px;padding:10px 24px;background:#fff;border:none;color:#1e1b4b;border-radius:24px;font-size:13px;font-weight:800;cursor:pointer;transition:all .2s;margin-bottom:20px;box-shadow:0 4px 15px rgba(255,255,255,0.2);" onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 20px rgba(255,255,255,0.3)'" onmouseout="this.style.transform='';this.style.boxShadow='0 4px 15px rgba(255,255,255,0.2)'">✨ 데모스 소개 보기</button>
    <div class="login-feature">
      <span style="font-size:24px;">🧬</span>
      <div><div style="color:#e2e8f0;font-size:13px;font-weight:700;">355+ 전문 스킬</div><div style="color:#64748b;font-size:11px;">과학/개발/AI/인프라/비즈니스</div></div>
    </div>
    <div class="login-feature">
      <span style="font-size:24px;">🤖</span>
      <div><div style="color:#e2e8f0;font-size:13px;font-weight:700;">하네스 파이프라인</div><div style="color:#64748b;font-size:11px;">Expert Pool + 피드백 루프</div></div>
    </div>
    <div class="login-feature">
      <span style="font-size:24px;">📚</span>
      <div><div style="color:#e2e8f0;font-size:13px;font-weight:700;">개인 지식 도메인</div><div style="color:#64748b;font-size:11px;">사용자별 지식 등록/검색</div></div>
    </div>
    <div class="login-feature">
      <span style="font-size:24px;">🔀</span>
      <div><div style="color:#e2e8f0;font-size:13px;font-weight:700;">멀티에이전트 CEO</div><div style="color:#64748b;font-size:11px;">병렬 실행 + LGTM 자동 리뷰</div></div>
    </div>
  </div>
  <!-- 오른쪽: 로그인 카드 -->
  <div style="width:100%;max-width:420px;">
  <script>if(window.innerWidth>768)document.getElementById('loginFeatures').style.display='block';</script>
    <!-- 카드 -->
    <div style="background:rgba(15,23,42,0.8);backdrop-filter:blur(24px);border:1px solid rgba(99,102,241,0.2);border-radius:24px;padding:48px 36px 36px;box-shadow:0 25px 50px rgba(0,0,0,0.5);">
      <!-- 로고 -->
      <div style="text-align:center;margin-bottom:36px;">
        <div style="display:inline-flex;align-items:center;justify-content:center;width:72px;height:72px;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);border-radius:20px;margin-bottom:20px;animation:loginPulse 3s ease-in-out infinite;">
          <span style="font-size:32px;color:#fff;font-weight:900;text-shadow:0 2px 8px rgba(0,0,0,0.3);">D</span>
        </div>
        <h1 style="color:#fff;font-size:26px;font-weight:800;margin:0 0 4px;letter-spacing:-0.5px;">Demos V1.0</h1>
        <p style="color:#64748b;font-size:13px;margin:0;">AI Scientific Assistant Platform</p>
        <div style="width:40px;height:3px;background:linear-gradient(90deg,#6366f1,#a855f7);margin:16px auto 0;border-radius:2px;"></div>
      </div>
      <!-- 에러 -->
      <div id="loginError" style="display:none;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);color:#f87171;padding:10px 14px;border-radius:10px;margin-bottom:16px;font-size:13px;text-align:center;"></div>
      <!-- 입력 -->
      <div style="margin-bottom:14px;">
        <label style="display:block;color:#94a3b8;font-size:11px;font-weight:700;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;">ID</label>
        <div style="position:relative;">
          <span style="position:absolute;left:14px;top:50%;transform:translateY(-50%);color:#475569;font-size:16px;">👤</span>
          <input id="loginId" type="text" placeholder="아이디를 입력하세요" style="width:100%;padding:13px 16px 13px 42px;border:1.5px solid #1e293b;border-radius:12px;background:rgba(30,41,59,0.5);color:#f1f5f9;font-size:14px;box-sizing:border-box;outline:none;transition:all .2s;" onkeydown="if(event.key==='Enter')document.getElementById('loginPw').focus()">
        </div>
      </div>
      <div style="margin-bottom:24px;">
        <label style="display:block;color:#94a3b8;font-size:11px;font-weight:700;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;">PASSWORD</label>
        <div style="position:relative;">
          <span style="position:absolute;left:14px;top:50%;transform:translateY(-50%);color:#475569;font-size:16px;">🔒</span>
          <input id="loginPw" type="password" placeholder="비밀번호를 입력하세요" style="width:100%;padding:13px 16px 13px 42px;border:1.5px solid #1e293b;border-radius:12px;background:rgba(30,41,59,0.5);color:#f1f5f9;font-size:14px;box-sizing:border-box;outline:none;transition:all .2s;" onkeydown="if(event.key==='Enter')doLogin()">
        </div>
      </div>
      <!-- 버튼 -->
      <button onclick="doLogin()" style="width:100%;padding:14px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;box-shadow:0 4px 15px rgba(99,102,241,0.3);" onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 6px 20px rgba(99,102,241,0.4)'" onmouseout="this.style.transform='';this.style.boxShadow='0 4px 15px rgba(99,102,241,0.3)'">로그인</button>
    </div>
    <!-- 하단 -->
    <p style="text-align:center;margin-top:24px;">
      <button onclick="openAboutPopup()" style="background:none;border:none;color:#6366f1;font-size:12px;cursor:pointer;text-decoration:underline;">소개 페이지</button>
      <span style="color:#334155;font-size:11px;">&middot; &copy; 2026 Demos Platform &middot; Harness Integrated</span>
    </p>
  </div>
  </div>
  </div>
</div>

<!-- ADMIN 사용자 관리 팝업 -->
<div id="adminPopup" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;background:rgba(0,0,0,0.7);align-items:center;justify-content:center;">
  <div style="background:#1e1e2e;border:1px solid #333;border-radius:16px;padding:24px;width:480px;max-height:80vh;overflow-y:auto;color:#e2e8f0;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <h3 style="margin:0;color:#f59e0b;">👑 사용자 관리 (ADMIN)</h3>
      <button onclick="closeAdminPopup()" style="background:none;border:none;color:#888;font-size:20px;cursor:pointer;">✕</button>
    </div>
    <div style="background:rgba(99,102,241,0.1);border:1px solid #6366f1;border-radius:10px;padding:14px;margin-bottom:16px;">
      <div style="font-weight:700;margin-bottom:10px;color:#a5b4fc;">➕ 새 사용자 등록</div>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <input id="regName" placeholder="이름" style="padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
        <input id="regId" placeholder="아이디" style="padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
        <input id="regPw" type="password" placeholder="비밀번호" style="padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
        <select id="regRole" style="padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
          <option value="user">일반 사용자</option>
          <option value="admin">관리자</option>
        </select>
        <button onclick="registerUser()" style="padding:10px;background:#6366f1;color:#fff;border:none;border-radius:6px;font-weight:700;cursor:pointer;">등록</button>
      </div>
      <div id="regMsg" style="margin-top:6px;font-size:12px;"></div>
    </div>
    <div style="font-weight:700;margin-bottom:8px;color:#a5b4fc;">📋 등록된 사용자</div>
    <div id="adminUserList"></div>
  </div>
</div>


<!-- 지식 도메인 관리 팝업 -->
<div id="knowledgePopup" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;background:rgba(0,0,0,0.7);align-items:center;justify-content:center;">
  <div style="background:#1e1e2e;border:1px solid #333;border-radius:16px;padding:24px;width:560px;max-height:85vh;overflow-y:auto;color:#e2e8f0;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <h3 style="margin:0;color:#22c55e;">📚 내 지식 도메인</h3>
      <button onclick="closeKnowledgePopup()" style="background:none;border:none;color:#888;font-size:20px;cursor:pointer;">✕</button>
    </div>

    <!-- 탭: 파일 업로드 / 수동 입력 -->
    <div style="display:flex;gap:4px;margin-bottom:16px;">
      <button id="kdTabUpload" onclick="switchKdTab('upload')" style="flex:1;padding:8px;background:#6366f1;color:#fff;border:none;border-radius:6px;cursor:pointer;font-weight:600;">📁 파일 업로드</button>
      <button id="kdTabManual" onclick="switchKdTab('manual')" style="flex:1;padding:8px;background:#2a2a3e;color:#888;border:none;border-radius:6px;cursor:pointer;font-weight:600;">✏️ 수동 입력</button>
    </div>

    <!-- 파일 업로드 탭 -->
    <div id="kdUploadTab" style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:14px;margin-bottom:16px;">
      <div style="text-align:center;padding:20px;border:2px dashed #444;border-radius:8px;cursor:pointer;margin-bottom:10px;" onclick="document.getElementById('kdFileInput').click()" id="kdDropZone">
        <div style="font-size:32px;margin-bottom:8px;">📄</div>
        <div style="color:#94a3b8;font-size:13px;">MD 또는 TXT 파일을 클릭하여 선택</div>
        <div id="kdFileName" style="color:#22c55e;font-size:13px;margin-top:6px;font-weight:600;"></div>
      </div>
      <input type="file" id="kdFileInput" accept=".md,.txt" style="display:none;" onchange="kdFileSelected(this)">
      <button onclick="uploadKdFile()" style="width:100%;padding:10px;background:#22c55e;color:#fff;border:none;border-radius:6px;font-weight:700;cursor:pointer;">📤 업로드 + 프론트매터 자동 생성</button>
      <div id="kdUploadMsg" style="margin-top:6px;font-size:12px;"></div>
    </div>

    <!-- 수동 입력 탭 -->
    <div id="kdManualTab" style="display:none;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.3);border-radius:10px;padding:14px;margin-bottom:16px;">
      <input id="kdTitle" placeholder="제목" style="width:100%;padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;margin-bottom:8px;box-sizing:border-box;">
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <select id="kdCategory" onchange="if(this.value==='_custom')document.getElementById('kdCategoryCustom').style.display='block'" style="flex:1;padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
          <option value="guide">가이드</option>
          <option value="column_definition">컬럼 사전</option>
          <option value="architecture">아키텍처</option>
          <option value="specification">기술 명세</option>
          <option value="changelog">변경 이력</option>
          <option value="plan">계획 문서</option>
          <option value="etc">기타</option>
          <option value="_custom">+ 직접 입력</option>
        </select>
        <input id="kdCategoryCustom" placeholder="새 카테고리명" style="display:none;flex:1;padding:8px;border:1px solid #6366f1;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;">
      </div>
      <input id="kdTags" placeholder="태그 (쉼표 구분: OHT, 반송, M14)" style="width:100%;padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;margin-bottom:8px;box-sizing:border-box;">
      <textarea id="kdContent" placeholder="내용을 입력하세요..." style="width:100%;height:150px;padding:8px;border:1px solid #444;border-radius:6px;background:#2a2a3e;color:#fff;font-size:13px;margin-bottom:8px;box-sizing:border-box;resize:vertical;"></textarea>
      <button onclick="saveKdManual()" style="width:100%;padding:10px;background:#6366f1;color:#fff;border:none;border-radius:6px;font-weight:700;cursor:pointer;">💾 저장 + 프론트매터 자동 생성</button>
      <div id="kdManualMsg" style="margin-top:6px;font-size:12px;"></div>
    </div>

    <!-- 내 파일 목록 -->
    <div style="font-weight:700;margin-bottom:8px;color:#a5b4fc;">📋 등록된 지식 파일</div>
    <div id="kdFileList" style="font-size:12px;color:#888;">로딩 중...</div>
  </div>
</div>

<!-- UIO 2D 픽셀 컨테이너 -->
<div id="uioContainer">
  <iframe id="uioFrame" src="about:blank"></iframe>
  <button id="uioBackBtn" onclick="toggleUioMode()">← 접기</button>
</div>

<div class="sidebar" id="sidebar">
  <button class="sidebar-toggle-mini" onclick="toggleSidebar()" title="펼치기">☰</button>
  <div class="sidebar-inner">
    <div class="sidebar-logo"><span>D</span>emos <span style="font-size:12px;color:#999">V1.0</span></div>
    <div style="display:flex;gap:4px;margin-bottom:4px;">
      <button class="sidebar-btn active" onclick="createNewSession()" style="flex:1">✨ 새 세션</button>
      <button class="sidebar-btn" onclick="selectAllSessions()" style="flex:0;padding:6px 10px;font-size:11px;" title="전체 선택">☑</button>
      <button class="sidebar-btn" onclick="deleteAllSessions()" style="flex:0;padding:6px 10px;font-size:11px;color:#ef4444;" title="전체 삭제">🗑️</button>
    </div>
    <div class="sidebar-section" ondblclick="openSessionPopup()" style="cursor:pointer;" title="더블클릭으로 전체 보기">세션 목록 <span class="skill-count" id="sessionCount">0</span></div>
    <div id="sessionList" style="max-height:250px;overflow-y:auto;margin-bottom:4px;"></div>
    <div class="session-actions" id="sessionActions" style="display:none">
      <button onclick="archiveSelectedSessions()">📦 보관</button>
      <button class="danger" onclick="deleteSelectedSessions()">🗑️ 삭제</button>
      <button onclick="clearSessionSelection()">취소</button>
    </div>
  </div>
  <div id="fileListPanel" class="sidebar-file-panel" style="display:none"></div>
  <div class="sidebar-footer">
    <div class="credits">🔬 Demos V1.0</div>
  </div>
  </div>
</div>
<!-- 세션 목록 팝업 -->
<div id="sessionPopupOverlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;justify-content:center;align-items:center;" onclick="if(event.target===this)closeSessionPopup()">
  <div style="background:#fff;border-radius:16px;width:90%;max-width:700px;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
    <div style="padding:16px 20px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center;">
      <div style="font-size:16px;font-weight:700;">세션 목록 <span id="popupSessionCount" style="font-size:13px;color:#6366f1;"></span></div>
      <div style="display:flex;gap:6px;">
        <button onclick="selectAllSessionsPopup()" style="padding:6px 12px;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;font-size:12px;">☑ 전체선택</button>
        <button onclick="deleteSelectedSessionsPopup()" style="padding:6px 12px;border:1px solid #fca5a5;border-radius:8px;background:#fff;color:#ef4444;cursor:pointer;font-size:12px;">🗑️ 선택삭제</button>
        <button onclick="deleteAllSessionsPopup()" style="padding:6px 12px;border:1px solid #fca5a5;border-radius:8px;background:#fef2f2;color:#dc2626;cursor:pointer;font-size:12px;">전체삭제</button>
        <button onclick="exportSessionsToMd()" style="padding:6px 12px;border:1px solid #86efac;border-radius:8px;background:#f0fdf4;color:#16a34a;cursor:pointer;font-size:12px;">📄 MD 저장</button>
        <button onclick="openHarnessSessionTab()" style="padding:6px 12px;border:1px solid #c7d2fe;border-radius:8px;background:#eef2ff;color:#6366f1;cursor:pointer;font-size:12px;font-weight:600">💾 하네스 저장</button>
        <button onclick="closeSessionPopup()" style="padding:6px 12px;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;font-size:16px;">✕</button>
      </div>
    </div>
    <div style="padding:8px 20px;border-bottom:1px solid #f3f4f6;">
      <input id="sessionSearchInput" type="text" placeholder="세션 내용 검색... (제목, 대화 내용)" oninput="searchSessionsPopup(this.value)" style="width:100%;padding:8px 12px;border:1px solid #d1d5db;border-radius:8px;font-size:13px;outline:none;box-sizing:border-box;" />
    </div>
    <div id="sessionPopupBody" style="flex:1;overflow-y:auto;padding:12px 16px;"></div>
  </div>
</div>

<button class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">◀</button>

<!-- 인트로 스플래시 (영상 우선, 없으면 이미지 3초) -->
<div id="splashScreen" style="position:fixed;inset:0;z-index:99999;background:#0a0e17;display:flex;align-items:center;justify-content:center;transition:opacity .6s;">
  <div onclick="skipSplash()" style="position:absolute;bottom:30px;right:40px;background:rgba(255,255,255,.15);backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,.3);color:#fff;padding:8px 20px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;z-index:1;transition:background .2s;" onmouseenter="this.style.background='rgba(255,255,255,.3)'" onmouseleave="this.style.background='rgba(255,255,255,.15)'">SKIP ▶</div>
  <video id="splashVideo" autoplay muted playsinline style="max-width:85%;max-height:85vh;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.6);display:none;">
    <source src="/beno/intro.mp4" type="video/mp4">
    <source src="/beno/intro.webm" type="video/webm">
  </video>
  <img id="splashImg" src="data:image/jpeg;base64,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" style="max-width:80%;max-height:80vh;border-radius:16px;box-shadow:0 20px 60px rgba(0,0,0,.6);animation:splashIn .8s ease-out;" alt="Demos">
</div>
<style>
@keyframes splashIn{0%{opacity:0;transform:scale(.9)}100%{opacity:1;transform:scale(1)}}
</style>
<script>
(function(){
  const splash=document.getElementById('splashScreen');
  const video=document.getElementById('splashVideo');
  const img=document.getElementById('splashImg');
  let phase='video'; // 'video' → 'image' → 'done'
  function closeSplash(){
    if(phase==='done') return;
    phase='done';
    splash.style.opacity='0';
    setTimeout(()=>splash.remove(),600);
  }
  function showImage(){
    if(phase==='done') return;
    phase='image';
    video.style.display='none';
    img.style.display='block';
    img.style.animation='splashIn .8s ease-out';
    setTimeout(closeSplash, 3000);
  }
  // 영상 로드 성공 → 재생 → 끝나면 이미지 3초
  video.addEventListener('canplay',()=>{
    if(phase!=='video') return;
    img.style.display='none';
    video.style.display='block';
    video.play();
  });
  video.addEventListener('ended', showImage);
  // 영상 없으면 바로 이미지 3초
  video.addEventListener('error', showImage);
  // 30초 안전장치
  setTimeout(()=>{ if(splash.parentNode) closeSplash(); }, 30000);
  // SKIP 버튼 → 영상: 이미지로 넘어감, 이미지: 메인으로 넘어감
  window.skipSplash = function(){
    if(phase==='video'){ video.pause(); showImage(); }
    else if(phase==='image') closeSplash();
  };
})();
</script>

<div class="main">
  <div class="header">
    <div class="project-title">📁 Demos V1.0 <span style="font-size:12px;color:#6366f1;background:#eef2ff;padding:2px 10px;border-radius:10px;margin-left:8px;font-weight:500;">Opus SKILL 4.6 하네스 사용중</span><button onclick="toggleMainTokenSettings()" style="font-size:12px;color:#6366f1;background:#eef2ff;border:1px solid #c7d2fe;padding:2px 10px;border-radius:10px;margin-left:6px;cursor:pointer;font-weight:500;" title="토큰/컨텍스트 설정">⚙️ 토큰 설정</button><span style="font-size:11px;color:#9ca3af;margin-left:6px;">2달에 한번 스킬 업데이트 | 사용을 많이 해줄수록 기능이 업데이트 됩니다</span></div>
    <div style="display:flex;align-items:center;gap:6px;">
      <button id="hdKdBtn" onclick="openKnowledgePopup()" style="font-size:12px;color:#fff;background:#22c55e;border:none;padding:5px 12px;border-radius:8px;cursor:pointer;font-weight:600;">📚 내 지식</button>
      <button id="hdAdminBtn" onclick="openAdminPopup()" style="display:none;font-size:12px;color:#000;background:#f59e0b;border:none;padding:5px 12px;border-radius:8px;cursor:pointer;font-weight:600;">👑 사용자 관리</button>
      <button id="hdLogoutBtn" onclick="doLogout()" style="font-size:12px;color:#888;background:none;border:1px solid #ddd;padding:5px 10px;border-radius:8px;cursor:pointer;">🚪</button>
      <span style="color:#ccc;font-size:10px;">|</span>
      <span id="tokenBadge" class="status off">⏳ 로딩중...</span>
      <button onclick="openHarnessSessionTab()" style="font-size:12px;color:#6366f1;background:#eef2ff;border:1px solid #c7d2fe;padding:4px 12px;border-radius:8px;cursor:pointer;font-weight:500;">💾 저장 세션</button>
      <a href="/uio" target="_blank" class="uio-enter-btn">🎮 2D 오피스</a>
      <span id="status" class="status off">⚪ 환경 미선택</span>
    </div>
  </div>
  <!-- 토큰 설정 패널 (프로젝트 타이틀 아래 드롭다운) -->
  <div id="mainTokenPanel" style="display:none;padding:12px 16px;background:#f8fafc;border-bottom:1px solid #e2e8f0;">
    <div style="display:flex;gap:24px;flex-wrap:wrap;">
      <div>
        <div style="font-size:11px;font-weight:700;color:#6366f1;margin-bottom:6px;">🌐 API 설정</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end;">
          <div style="min-width:140px;">
            <label style="font-size:11px;color:#64748b;display:block;margin-bottom:3px;">API max_tokens (응답 길이)</label>
            <select id="mainAgentMaxTokens" onchange="updateMainTokenSetting()" style="width:100%;padding:6px;border-radius:6px;border:1px solid #e2e8f0;font-size:12px;">
              <option value="2048">2048 (짧게)</option><option value="4096">4096 (보통)</option><option value="8192">8192 (길게)</option><option value="16384">16384 (매우 길게)</option>
            </select>
          </div>
          <div style="min-width:140px;">
            <label style="font-size:11px;color:#64748b;display:block;margin-bottom:3px;">CEO 합성 max_tokens (보고서)</label>
            <select id="mainSynthMaxTokens" onchange="updateMainTokenSetting()" style="width:100%;padding:6px;border-radius:6px;border:1px solid #e2e8f0;font-size:12px;">
              <option value="4096">4096 (짧게)</option><option value="8192">8192 (보통)</option><option value="16384">16384 (길게)</option><option value="32768">32768 (매우 길게)</option>
            </select>
          </div>
        </div>
      </div>
      <div style="border-left:1px solid #e2e8f0;padding-left:24px;">
        <div style="font-size:11px;font-weight:700;color:#f59e0b;margin-bottom:6px;">💻 GGUF 설정</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end;">
          <div style="min-width:140px;">
            <label style="font-size:11px;color:#64748b;display:block;margin-bottom:3px;">n_ctx 컨텍스트 윈도우</label>
            <select id="mainNCtx" onchange="updateMainTokenSetting()" style="width:100%;padding:6px;border-radius:6px;border:1px solid #e2e8f0;font-size:12px;">
              <option value="4096">4096 (빠름)</option><option value="8192">8192</option><option value="16384">16384</option><option value="32768">32768 (대용량)</option>
            </select>
          </div>
          <div style="min-width:140px;">
            <label style="font-size:11px;color:#64748b;display:block;margin-bottom:3px;">GGUF 응답 상한 (reply cap)</label>
            <select id="mainReplyCap" onchange="updateMainTokenSetting()" style="width:100%;padding:6px;border-radius:6px;border:1px solid #e2e8f0;font-size:12px;">
              <option value="4096">4096</option><option value="8192">8192</option><option value="16384">16384</option><option value="32768">32768</option>
            </select>
          </div>
        </div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;margin-top:8px;">
      <button onclick="saveTokenSettings()" style="padding:6px 20px;background:#6366f1;color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;">💾 저장 및 적용</button>
      <span id="tokenSaveStatus" style="font-size:11px;color:#22c55e;display:none;">✅ 저장 완료</span>
      <span style="font-size:10px;color:#94a3b8;">API: max_tokens↑ = 긴 응답 | GGUF: n_ctx↑ = 긴 대화 가능, VRAM 사용↑</span>
    </div>
  </div>
  <div class="content">
    <div class="content-inner">
      <!-- 환경 선택 -->
      <div class="section-label">LLM 환경 선택 <span id="envStatusBadge" style="font-weight:400;font-size:11px;color:#6366f1"></span></div>
      <div id="tokenStatus" class="token-status missing">⏳ 토큰 상태 확인중...</div>
      <div id="envContainer">
        <div class="env-section">
          <div class="env-row" id="envRowAuto"></div>
        </div>
        <div class="env-section" id="envSectionApi">
          <div class="env-section-label env-toggle" onclick="toggleEnvSection('Api')">
            <span id="envArrowApi">▶</span> API 모델
          </div>
          <div id="envApiInner" style="display:none">
            <div class="env-subsection-label" id="envHighLabel" style="display:none">🚀 고성능 모델 <span style="color:#b0b0b0">- 복잡한 분석, 코드 생성, 논문 작성</span></div>
            <div class="env-row" id="envRowHigh"></div>
            <div class="env-subsection-label" id="envMidLabel" style="display:none">⚡ 범용 모델 <span style="color:#b0b0b0">- 일반 질문, 요약, 간단한 코드</span></div>
            <div class="env-row" id="envRowMid"></div>
            <div class="env-subsection-label" id="envLowLabel" style="display:none">💨 경량 모델 <span style="color:#b0b0b0">- 빠른 응답, 단순 Q&A, 번역</span></div>
            <div class="env-row" id="envRowLow"></div>
            <div class="env-subsection-label" id="envVlLabel" style="display:none">👁️ Vision 모델 <span style="color:#b0b0b0">- 이미지 분석, 스크린샷 해석, 도면 인식</span></div>
            <div class="env-row" id="envRowVl"></div>
          </div>
        </div>
        <div class="env-section" id="envSectionGguf">
          <div class="env-section-label env-toggle" onclick="toggleEnvSection('Gguf')">
            <span id="envArrowGguf">▶</span> GGUF 로컬 모델
          </div>
          <div class="env-row env-collapsible" id="envRowGguf" style="display:none"></div>
        </div>
      </div>

      <!-- 업로드된 파일 목록은 사이드바 하단으로 이동 -->
      <div id="csvInfoPanel" style="display:none"></div>

      <div class="section-label" style="cursor:pointer;user-select:none;" onclick="toggleSection('tagSection','tagArrow')">
        <span class="arrow" id="tagArrow">▶</span> 분야 선택
      </div>
      <div id="tagSection" style="display:none">
        <div class="tag-row" id="tagRow"></div>
      </div>

      <div class="section-label" style="cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px;" onclick="toggleSection('skillSection','skillArrow')">
        <span class="arrow" id="skillArrow">▶</span> 스킬 선택 <span style="font-weight:400;color:#bbb">( ✅ = SKILL.md 로드됨 )</span>
        <button onclick="event.stopPropagation();openSkillCreate()" style="background:#6366f1;color:#fff;border:none;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer;margin-left:auto;">➕ 새 스킬</button>
      </div>
      <div id="skillSection" style="display:none">
        <div class="skill-grid" id="skillGrid"></div>
      </div>

      <div class="section-label" style="cursor:pointer;user-select:none;" onclick="toggleSection('comboSection','comboArrow')">
        <span class="arrow" id="comboArrow">▶</span> 스킬 조합 <span style="font-weight:400;color:#bbb">( 여러 조합을 동시에 선택할 수 있습니다 )</span>
      </div>
      <div id="comboSection" style="display:none;padding:4px 0;">
        <div id="comboActiveMsg" style="display:none;margin-bottom:8px;padding:8px 12px;border-radius:8px;background:#6366f110;border:1px solid #6366f130;font-size:12px;color:#6366f1;"></div>
        <div id="comboGrid" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
        <div id="comboClearWrap" style="display:none;margin-top:8px;text-align:right;">
          <button onclick="clearAllCombos()" style="background:#ef4444;color:#fff;border:none;border-radius:6px;padding:6px 16px;font-size:12px;font-weight:700;cursor:pointer;">🗑️ 전체 해제</button>
        </div>
      </div>

      <div class="sysprompt-section">
        <div class="sysprompt-toggle" onclick="toggleSysPrompt()">
          <span class="arrow" id="spArrow">▶</span>
          <div class="section-label" style="margin-bottom:0;cursor:pointer">시스템 프롬프트</div>
          <span id="spPreviewBadge" style="font-size:11px;color:#6366f1;background:#eef2ff;padding:2px 8px;border-radius:10px">기본값</span>
        </div>
        <div class="sysprompt-body" id="spBody">
          <!-- 프리셋/저장 프롬프트 선택 바 -->
          <div id="promptPresets" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px"></div>
          <textarea class="sysprompt-textarea" id="systemPromptInput" placeholder="시스템 프롬프트를 입력하세요. 비워두면 기본값 사용됩니다.&#10;&#10;아래 프리셋 버튼을 클릭하면 자동으로 채워집니다.&#10;수정 후 '저장' 버튼으로 나만의 프롬프트를 저장하세요."></textarea>
          <div class="sysprompt-info" style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
            <span>💡 스킬 프롬프트 <strong>앞에</strong> 추가됩니다.</span>
            <button style="background:#6366f1;color:#fff;border:none;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer" onclick="saveCurrentPrompt()">💾 저장</button>
            <button style="background:#ef4444;color:#fff;border:none;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer" onclick="deleteCurrentPrompt()" id="btnDeletePrompt" title="현재 선택된 저장 프롬프트 삭제">🗑️ 삭제</button>
            <button style="background:#f3f2ef;border:1px solid #e5e3de;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer" onclick="resetSysPrompt()">초기화</button>
            <span id="promptSaveStatus" style="font-size:11px;color:#16a34a;display:none">✅ 저장됨!</span>
          </div>
        </div>
      </div>

      <div class="effort-section">
        <div class="section-label">응답 수준</div>
        <input type="range" class="effort-slider" id="effortSlider" min="0" max="3" value="2" oninput="updateEffort()">
        <div class="effort-labels">
          <span id="e0">즉시</span><span id="e1">빠름</span><span id="e2" class="active">표준</span><span id="e3">프로</span>
        </div>
      </div>

      <!-- 작성스타일/출력형식: 채팅창 드롭다운으로 이동, 내부 상태 유지용 hidden -->
      <div style="display:none">
        <div id="styleChips"></div>
        <textarea id="writingStyle"></textarea>
        <div class="fmt-btns">
          <div class="fmt-btn selected" data-f="auto" onclick="selFmt(this)">🤖 자동</div>
          <div class="fmt-btn" data-f="code" onclick="selFmt(this)">코드</div>
          <div class="fmt-btn" data-f="code-fix" onclick="selFmt(this)">코드수정</div>
          <div class="fmt-btn" data-f="analysis" onclick="selFmt(this)">분석</div>
          <div class="fmt-btn" data-f="report" onclick="selFmt(this)">보고서</div>
          <div class="fmt-btn" data-f="step-by-step" onclick="selFmt(this)">단계별</div>
        </div>
      </div>

      <!-- 자동 스킬 선택 -->
      <div class="auto-skill-toggle" id="autoSkillToggle" onclick="toggleAutoSkill()">
        <div class="toggle-dot"></div>
        <div>
          <div style="font-weight:700;font-size:13px">🧠 자동 스킬 선택</div>
          <div style="font-size:11px;color:#888">질문을 분석하여 관련 스킬을 자동으로 로드합니다</div>
        </div>
      </div>
      <div class="quick-row">
        <button class="quick-btn" onclick="qp('이 데이터를 분석해줘')">📊 데이터 분석</button>
        <button class="quick-btn" onclick="qp('업로드된 CSV 데이터의 통계 요약을 보여줘')">📋 CSV 요약</button>
        <button class="quick-btn" onclick="qp('이 데이터로 시각화 코드를 작성해줘')">📈 시각화</button>
        <button class="quick-btn" onclick="qp('Python 코드를 작성해줘')">🐍 코드 작성</button>
        <button class="quick-btn" onclick="qp('이 데이터에서 이상치를 찾아줘')">🔍 이상치 탐색</button>
        <button class="quick-btn" onclick="qp('논문 스타일로 정리해줘')">📝 논문 정리</button>
      </div>

      <div id="pptSuggestArea"></div>
      <div class="messages" id="msgs"></div>
    </div>
  </div>

  <!-- 질문 입력 - 하단 고정 -->
  <div class="chat-box-fixed">
    <div class="chat-box-fixed-inner">
      <div class="auto-skill-preview" id="autoSkillPreview"></div>
      <div class="chat-attach-preview" id="chatAttachPreview" style="display:none"></div>
      <textarea class="chat-input" id="input" placeholder="질문을 하거나 수행하려는 분석을 설명하세요..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();handleSendStop()}" oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px';checkPptStyleVisibility()"></textarea>
      <input type="file" id="chatFileInput" multiple style="display:none" onchange="handleChatFileSelect(this.files)">
      <div class="chat-footer">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          <button class="attach-btn" onclick="document.getElementById('chatFileInput').click()" title="파일 첨부">📎</button>
          <select class="chat-dropdown" id="chatStyleDrop" onchange="applyChatStyle(this.value)" title="작성 스타일">
            <option value="">🤖 자동</option>
            <option value="간결하고 핵심만. 불필요한 설명 생략.">⚡ 간결</option>
            <option value="상세하고 친절하게. 원리, 배경, 예시 포함.">📖 상세</option>
            <option value="바로 복붙해서 쓸 수 있게. 실전 위주, 이론 최소화.">🔨 실용적</option>
            <option value="학술적 톤. 정확한 용어, 인용, 근거 제시.">🎓 학술</option>
            <option value="코드 주석을 상세한 한국어로. 변수명은 영어 유지.">💬 한글주석</option>
            <option value="시니어 개발자 관점. 왜 이렇게 하는지, 대안은 뭔지, 주의점 포함.">👨‍💻 시니어</option>
            <option value="에러 원인 분석 중심. traceback 해석, 재현 조건, 해결책 순서.">🐛 디버그</option>
            <option value="데이터 스토리텔링. 숫자→의미→액션 순서로 해석.">📊 데이터</option>
          </select>
          <select class="chat-dropdown" id="chatFormatDrop" onchange="applyChatFormat(this.value)" title="출력 형식">
            <option value="auto">🤖 자동</option>
            <option value="code">💻 코드</option>
            <option value="code-fix">🔧 수정</option>
            <option value="analysis">📊 분석</option>
            <option value="report">📄 보고서</option>
            <option value="step-by-step">📝 단계별</option>
          </select>
          <select class="chat-dropdown ppt-style-drop" id="pptStyleDrop" onchange="applyPptStyle(this.value)" title="PPT 디자인 스타일">
            <option value="">📽️ PPT:자동</option>
            <option value="minimal">⬜ 미니멀</option>
            <option value="corporate">🏢 기업용</option>
            <option value="colorful">🎨 컬러풀</option>
            <option value="dark">🌙 다크</option>
            <option value="academic">🎓 학술</option>
            <option value="startup">🚀 스타트업</option>
            <option value="ref">📎 참고PPT</option>
          </select><span id="pptRefBadge" class="ppt-ref-badge" style="display:none">📎 참고스타일 적용중</span>
          <label style="display:flex;align-items:center;gap:3px;font-size:11px;color:#888;cursor:pointer;user-select:none" title="체크하면 모델이 답변 전에 깊이 사고합니다 (응답 느림)">
            <input type="checkbox" id="thinkToggle" style="margin:0;accent-color:#7c5cbf">💭사고
          </label>
          <span style="font-size:11px;color:#bbb">Enter 전송</span>
        </div>
        <button class="send-btn" onclick="handleSendStop()" id="sendBtn">▶</button>
      </div>
    </div>
  </div>
</div>

<!-- 오른쪽 코드 어시스턴트 패널 -->
<div class="right-panel" id="rightPanel">
  <div class="rp-header">
    <h3>🖥️ Code Assistant<span class="rp-skill-badge">code-assistant</span></h3>
  </div>
  <div class="rp-body">
    <div class="rp-section">
      <div class="rp-section-title">프로젝트 / 코드 파일</div>
      <div class="rp-tab-row">
        <button class="rp-tab active" onclick="switchUploadMode('folder',this)">📁 폴더 업로드</button>
        <button class="rp-tab" onclick="switchUploadMode('files',this)">📄 파일 업로드</button>
      </div>
      <div id="rpFolderMode">
        <div class="rp-file-drop" id="rpFolderDrop" onclick="document.getElementById('rpFolderInput').click()">
          <div class="rp-file-drop-icon">📁</div>
          <div class="rp-file-drop-text">클릭하여 <strong>프로젝트 폴더</strong> 통째로 업로드<br><span style="font-size:10px;color:#bbb">폴더 선택 시 하위 파일 전체가 트리로 표시됩니다</span></div>
        </div>
        <input type="file" id="rpFolderInput" webkitdirectory directory multiple style="display:none" onchange="handleRpFolderSelect(this.files)">
      </div>
      <div id="rpFileMode" style="display:none">
        <div class="rp-file-drop" id="rpFileDrop" onclick="document.getElementById('rpFileInput').click()">
          <div class="rp-file-drop-icon">📄</div>
          <div class="rp-file-drop-text">클릭 또는 드래그하여 코드 파일 업로드<br><span style="font-size:10px;color:#bbb">.py .js .html .css .java .c .cpp .go .rs .sh .json .yaml .md</span></div>
        </div>
        <input type="file" id="rpFileInput" multiple accept=".py,.js,.html,.css,.json,.xml,.yaml,.yml,.c,.cpp,.h,.java,.go,.rs,.sh,.bat,.md,.txt" style="display:none" onchange="handleRpFileSelect(this.files)">
      </div>
      <div id="rpTreeStats" class="rp-tree-stats" style="display:none">
        <span id="rpStatsInfo">0 파일 / 0 폴더</span>
        <span id="rpStatsSize">0 KB</span>
        <button onclick="clearRpProject()" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:11px;font-weight:600">전체 삭제</button>
      </div>
      <div id="rpLangDetect" style="display:none;padding:6px 10px;background:#f0f0ff;border-radius:6px;margin-bottom:6px;font-size:11px;">
        <div style="font-weight:600;color:#4f46e5;margin-bottom:3px;">🔍 감지된 언어/도구</div>
        <div id="rpLangTags" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
        <div id="rpAutoSkills" style="margin-top:4px;color:#6b7280;font-size:10px;"></div>
      </div>
      <div class="rp-select-bar" id="rpSelectBar" style="display:none">
        <button class="rp-select-btn" onclick="rpCheckAll(true)">전체 선택</button>
        <button class="rp-select-btn" onclick="rpCheckAll(false)">전체 해제</button>
        <span class="rp-checked-count" id="rpCheckedCount">0개 선택</span>
      </div>
      <div class="rp-tree" id="rpTree" style="display:none"></div>
      <div class="rp-file-list" id="rpFileList"></div>
    </div>

    <div class="rp-section">
      <div class="rp-section-title">분석 모드 선택 <span style="font-size:10px;color:#6366f1;background:#eef2ff;padding:1px 6px;border-radius:6px;">🚀 Coder-480B 전용</span></div>
      <div class="rp-action-grid">
        <button class="rp-action-btn" data-mode="explain" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">📖</span>
          <span class="rp-btn-label">코드 설명</span>
        </button>
        <button class="rp-action-btn" data-mode="find_bugs" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🐛</span>
          <span class="rp-btn-label">버그 탐지</span>
        </button>
        <button class="rp-action-btn" data-mode="improve" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">✨</span>
          <span class="rp-btn-label">개선 제안</span>
        </button>
        <button class="rp-action-btn" data-mode="tests" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🧪</span>
          <span class="rp-btn-label">테스트 생성</span>
        </button>
        <button class="rp-action-btn" data-mode="docstring" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">📝</span>
          <span class="rp-btn-label">문서화</span>
        </button>
        <button class="rp-action-btn" data-mode="refactor" onclick="selectRpMode(this)">
          <span class="rp-btn-icon">🔧</span>
          <span class="rp-btn-label">리팩토링</span>
        </button>
      </div>
    </div>

    <button class="rp-run-btn" id="rpRunBtn" onclick="runCodeAssistant()" disabled>▶ 분석 실행</button>
    <div class="rp-result" id="rpResult" style="display:none"></div>
  </div>
</div>
<button class="right-panel-toggle" id="rightPanelToggle" onclick="toggleRightPanel()">▶</button>

<!-- 스킬 상세 패널 -->
<div id="skillDetailOverlay" class="skill-detail-overlay" style="display:none" onclick="if(event.target===this)closeSkillDetail()">
  <div class="skill-detail-panel">
    <div class="sdp-header">
      <h2 id="sdpTitle">스킬 상세</h2>
      <button class="sdp-close" onclick="closeSkillDetail()">✕</button>
    </div>
    <div class="sdp-tabs" id="sdpTabs"></div>
    <div class="sdp-body" id="sdpBody">로딩중...</div>
  </div>
</div>

<!-- 스킬 생성 모달 -->
<div id="skillCreateOverlay" class="skill-detail-overlay" style="display:none" onclick="if(event.target===this)closeSkillCreate()">
  <div class="skill-detail-panel" style="max-width:800px;max-height:90vh;">
    <div class="sdp-header">
      <h2>➕ 새 스킬 만들기</h2>
      <button class="sdp-close" onclick="closeSkillCreate()">✕</button>
    </div>
    <div class="sdp-tabs" id="scTabs">
      <div class="sdp-tab active" onclick="switchScTab('manual')">✏️ 직접 작성</div>
      <div class="sdp-tab" onclick="switchScTab('llm')">🤖 LLM 자동 생성</div>
    </div>
    <div class="sdp-body" id="scBody" style="overflow-y:auto;max-height:calc(90vh - 120px);">
      <!-- 직접 작성 탭 -->
      <div id="scManual">
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">스킬 이름 <span style="color:#ef4444">*</span></label>
          <input id="scName" type="text" placeholder="my-skill-name (소문자+하이픈)" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;box-sizing:border-box;" oninput="validateScName()">
          <div id="scNameStatus" style="font-size:11px;margin-top:2px;"></div>
        </div>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">Description <span style="color:#ef4444">*</span></label>
          <textarea id="scDesc" rows="3" placeholder="스킬의 용도와 트리거 조건을 명확히 기술 (최소 20자)" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;box-sizing:border-box;"></textarea>
        </div>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">본문 (마크다운 지시문)</label>
          <textarea id="scBody_manual" rows="15" placeholder="# 스킬 제목&#10;&#10;## Overview&#10;스킬이 하는 일 설명&#10;&#10;## When to Use&#10;- 이 스킬을 사용할 상황 1&#10;- 상황 2&#10;&#10;## Instructions&#10;Claude가 따를 지시문..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;resize:vertical;box-sizing:border-box;"></textarea>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button onclick="validateSkillPreview()" style="background:#6366f1;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">🔍 검증</button>
          <button onclick="applySkill()" style="background:#f59e0b;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">🚀 스킬 적용</button>
          <button onclick="downloadSkill()" style="background:#16a34a;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">📥 다운로드 (.zip)</button>
        </div>
        <div id="scValidateResult" style="margin-top:10px;"></div>
      </div>
      <!-- LLM 자동 생성 탭 -->
      <div id="scLlm" style="display:none;">
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">스킬 주제 <span style="color:#ef4444">*</span></label>
          <input id="scTopic" type="text" placeholder="예: 반도체 식각 공정 분석, 코드 리뷰어, 논문 검색" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;box-sizing:border-box;">
        </div>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">스킬 유형</label>
          <select id="scType" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;">
            <option value="분석/도구">🔧 분석/도구형 (데이터 분석, 자동화)</option>
            <option value="에이전트">🤖 역할/에이전트형 (전문가 페르소나)</option>
            <option value="가이드">📖 가이드/지식형 (기법, 지식 제공)</option>
          </select>
        </div>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">상세 요구사항 (선택)</label>
          <textarea id="scDetails" rows="4" placeholder="스킬에 포함할 기능, 특수 요구사항 등..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;box-sizing:border-box;"></textarea>
        </div>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">생성 언어</label>
          <div style="display:flex;gap:6px;">
            <button id="scLangKo" onclick="setScLang('ko')" style="padding:6px 16px;border-radius:6px;font-size:13px;cursor:pointer;border:2px solid #6366f1;background:#6366f1;color:#fff;font-weight:700;">🇰🇷 한글</button>
            <button id="scLangEn" onclick="setScLang('en')" style="padding:6px 16px;border-radius:6px;font-size:13px;cursor:pointer;border:2px solid #e2e8f0;background:#fff;color:#333;font-weight:700;">🇺🇸 English</button>
          </div>
          <input type="hidden" id="scLang" value="ko">
        </div>
        <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
          <button onclick="generateSkillLLM()" id="scGenBtn" style="background:#8b5cf6;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">🤖 LLM으로 생성</button>
          <button onclick="validateGenerated()" style="background:#6366f1;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">🔍 검증</button>
          <button onclick="applyGeneratedSkill()" style="background:#f59e0b;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">🚀 스킬 적용</button>
          <button onclick="downloadGeneratedSkill()" style="background:#16a34a;color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer;">📥 다운로드 (.zip)</button>
        </div>
        <div id="scGenStatus" style="font-size:12px;color:#6366f1;display:none;margin-bottom:8px;">⏳ 생성 중... (최대 2분)</div>
        <div id="scGenResult" style="display:none;">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px;">SKILL.md 내용 (LLM 생성 후 편집 가능)</label>
          <textarea id="scGenContent" rows="20" placeholder="🤖 LLM으로 생성 버튼을 클릭하면 여기에 SKILL.md 초안이 생성됩니다.&#10;직접 입력해도 됩니다." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;resize:vertical;box-sizing:border-box;"></textarea>
          <div id="scGenValidateResult" style="margin-top:10px;"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let envs = {};
let hasToken = false;
let selEnvs = ['auto'];  // 배열: ['auto'] 또는 ['gguf-0','gguf-1'] 또는 ['coder-480b','dev']
let catalog = {};
let selDomains = [];
let selSkills = [];
let autoSkillMode = true;  // 기본 ON
let selFormat = 'auto';
let formatManualOverride = false;  // 사용자가 수동으로 출력형식을 변경했는지
let styleManualOverride = false;   // 사용자가 수동으로 스타일을 변경했는지
let effort = 2;
let history = [];
let maxTokens = 0;  // 0 = 서버 자동 결정 (모델에 맞게)
// ===== 스킬 조합 프리셋 =====
const SKILL_COMBOS = [
  {
    id:'data-analysis', label:'📊 데이터 분석', color:'#6366f1',
    desc:'CSV/엑셀 데이터 탐색, 통계, 시각화 풀세트',
    skills:['exploratory-data-analysis','statistical-analysis','matplotlib','seaborn','plotly','polars','agent-data-scientist','xlsx']
  },
  {
    id:'coding', label:'💻 코딩 개발', color:'#10b981',
    desc:'코드 작성, 리뷰, 디버깅 올인원',
    skills:['agent-python-pro','agent-fullstack-developer','debugging','agent-debugger','code-review','agent-backend-developer','agent-frontend-developer']
  },
  {
    id:'ml-ai', label:'🤖 ML/AI 제작', color:'#f59e0b',
    desc:'머신러닝 모델 학습, 평가, 해석',
    skills:['scikit-learn','pytorch-lightning','shap','umap-learn','statsmodels','agent-data-scientist','statistical-analysis','exploratory-data-analysis']
  },
  {
    id:'report-docs', label:'📝 보고서/문서', color:'#ef4444',
    desc:'Word, PPT, PDF, 마크다운 보고서 생성',
    skills:['docx','pptx','xlsx','pdf','scientific-writing','markdown-mermaid-writing','drawio-diagram']
  },
  {
    id:'literature', label:'📚 논문/문헌', color:'#8b5cf6',
    desc:'논문 검색, 문헌 리뷰, 인용 관리',
    skills:['literature-review','citation-management','scientific-writing','pubmed-database','openalex-database','biorxiv-database','peer-review']
  },
  {
    id:'visualization', label:'📈 시각화 전문', color:'#14b8a6',
    desc:'다양한 차트/다이어그램/인포그래픽 생성',
    skills:['matplotlib','seaborn','plotly','scientific-visualization','scientific-schematics','drawio-diagram','infographics','markdown-mermaid-writing']
  },
  {
    id:'web-fullstack', label:'🌐 웹 풀스택', color:'#0ea5e9',
    desc:'프론트+백엔드+DB+배포 풀세트',
    skills:['agent-fullstack-developer','agent-frontend-developer','agent-backend-developer','agent-react-specialist','agent-nextjs-developer','databases','frontend-development','backend-development']
  },
  {
    id:'devops-cloud', label:'☁️ DevOps/클라우드', color:'#64748b',
    desc:'CI/CD, 도커, 쿠버네티스, 클라우드 아키텍처',
    skills:['devops','agent-cloud-architect','agent-devops-engineer','agent-docker-expert','agent-kubernetes-specialist','agent-terraform-engineer']
  },
  {
    id:'skill-creation', label:'🛠️ 스킬 제작', color:'#a855f7',
    desc:'Claude 스킬 생성 (3개 메타스킬 조합)',
    skills:['skill-creator','anthropic-prompt-engineer','engineer-skill-creator','anthropic-architect']
  },
  {
    id:'timeseries', label:'⏱️ 시계열 분석', color:'#f97316',
    desc:'시계열 데이터 예측, 이상 탐지, 패턴 분석',
    skills:['aeon','timesfm-forecasting','statistical-analysis','statsmodels','matplotlib','exploratory-data-analysis']
  },
  {
    id:'prompt-eng', label:'✍️ 프롬프트 엔지니어링', color:'#7c3aed',
    desc:'Claude/GPT 프롬프트 최적화',
    skills:['anthropic-prompt-engineer','openai-prompt-engineer','anthropic-architect','sequential-thinking']
  },
  {
    id:'fab-semiconductor', label:'🏭 반도체/FAB', color:'#475569',
    desc:'FAB 공정 데이터 분석, 통계, 시각화',
    skills:['exploratory-data-analysis','statistical-analysis','matplotlib','seaborn','polars','agent-data-scientist','scikit-learn']
  },
  {
    id:'knowledge-domain', label:'📖 지식 도메인', color:'#0d9488',
    desc:'도메인 지식/FAB 컬럼/아키텍처 검색',
    skills:['knowledge-search']
  },
  {
    id:'logpresso-query', label:'🔍 로그프레소 쿼리', color:'#2563eb',
    desc:'LPQL 쿼리 생성 전문가',
    skills:['logpresso-query']
  },
  {
    id:'logpresso-search', label:'🔎 로그프레소 조회', color:'#1d4ed8',
    desc:'로그프레소 서버 직접 조회/검색',
    skills:['logpresso-search']
  },
  {
    id:'superpowers', label:'⚡ 슈퍼파워', color:'#eab308',
    desc:'8개 행동 메타 스킬 — 브레인스토밍/TDD/디버깅/검증/리뷰 세트',
    skills:['using-superpowers','brainstorming','writing-plans','test-driven-development','systematic-debugging','verification-before-completion','writing-skills','requesting-code-review']
  },
];

function renderCombos(){
  const grid = document.getElementById('comboGrid');
  if(!grid) return;
  grid.innerHTML = '';
  SKILL_COMBOS.forEach(combo=>{
    const allSelected = combo.skills.every(sid=>selSkills.includes(sid));
    const someSelected = !allSelected && combo.skills.some(sid=>selSkills.includes(sid));
    const borderColor = allSelected ? '#ef4444' : someSelected ? '#f59e0b' : combo.color+'22';
    const bgColor = allSelected ? '#ef444410' : someSelected ? '#f59e0b08' : combo.color+'06';
    const card = document.createElement('div');
    card.style.cssText = `width:calc(50% - 4px);padding:10px 12px;border-radius:8px;border:2px solid ${borderColor};background:${bgColor};cursor:pointer;transition:all .15s;user-select:none;`;
    card.title = allSelected ? '클릭하면 이 조합을 해제합니다' : '클릭하면 이 스킬들이 추가됩니다 (기존 선택 유지)';
    card.onmouseenter = ()=>{ card.style.background=combo.color+'14'; };
    card.onmouseleave = ()=>{ card.style.background=bgColor; };
    card.onclick = ()=>{ applyCombo(combo); };
    const statusBadge = allSelected ? '<span style="font-size:10px;background:#ef4444;color:#fff;padding:1px 6px;border-radius:4px;margin-left:6px;">적용중</span>' : someSelected ? '<span style="font-size:10px;background:#f59e0b;color:#fff;padding:1px 6px;border-radius:4px;margin-left:6px;">일부</span>' : '';
    const skillTags = combo.skills.map(sid=>{
      const koName = _comboSkillNames[sid] || sid;
      const isOn = selSkills.includes(sid);
      const tagBg = isOn ? '#ef444420' : combo.color+'12';
      const tagColor = isOn ? '#ef4444' : combo.color;
      const check = isOn ? '✓ ' : '';
      return `<span style="display:inline-block;background:${tagBg};color:${tagColor};padding:1px 6px;border-radius:4px;font-size:10px;margin:1px;font-weight:${isOn?700:400}">${check}${koName}</span>`;
    }).join('');
    card.innerHTML = `<div style="font-size:13px;font-weight:700;color:${combo.color};margin-bottom:3px;">${combo.label}${statusBadge}</div>
      <div style="font-size:11px;color:#64748b;margin-bottom:5px;">${combo.desc}</div>
      <div style="line-height:1.8;">${skillTags}</div>`;
    grid.appendChild(card);
  });
  updateComboActiveMsg();
}
// 스킬 ID → 짧은 한국어 이름 매핑
const _comboSkillNames = {
  'exploratory-data-analysis':'탐색적분석','statistical-analysis':'통계분석','matplotlib':'Matplotlib','seaborn':'Seaborn',
  'plotly':'Plotly','polars':'Polars','agent-data-scientist':'데이터과학자','xlsx':'엑셀','agent-python-pro':'Python전문가',
  'agent-fullstack-developer':'풀스택','debugging':'디버깅','agent-debugger':'디버거','code-review':'코드리뷰',
  'agent-backend-developer':'백엔드','agent-frontend-developer':'프론트엔드','scikit-learn':'Scikit-learn',
  'pytorch-lightning':'PyTorch','shap':'SHAP해석','umap-learn':'UMAP','statsmodels':'통계모델',
  'docx':'Word','pptx':'PPT','pdf':'PDF','scientific-writing':'논문작성','markdown-mermaid-writing':'마크다운',
  'drawio-diagram':'Draw.io','literature-review':'문헌리뷰','citation-management':'인용관리',
  'pubmed-database':'PubMed','openalex-database':'OpenAlex','biorxiv-database':'bioRxiv','peer-review':'논문심사',
  'scientific-visualization':'과학시각화','scientific-schematics':'과학다이어그램','infographics':'인포그래픽',
  'agent-react-specialist':'React','agent-nextjs-developer':'Next.js','databases':'데이터베이스',
  'frontend-development':'프론트개발','backend-development':'백엔드개발',
  'devops':'DevOps','agent-cloud-architect':'클라우드설계','agent-devops-engineer':'DevOps엔지니어',
  'agent-docker-expert':'Docker','agent-kubernetes-specialist':'K8s','agent-terraform-engineer':'Terraform',
  'skill-creator':'스킬제작','anthropic-prompt-engineer':'Anthropic프롬프트','engineer-skill-creator':'엔지니어스킬',
  'anthropic-architect':'Anthropic설계','aeon':'시계열ML','timesfm-forecasting':'시계열예측',
  'openai-prompt-engineer':'OpenAI프롬프트','sequential-thinking':'순차사고',
  'knowledge-search':'지식검색','logpresso-query':'LPQL쿼리','logpresso-search':'로그조회',
};

function applyCombo(combo){
  const allSelected = combo.skills.every(sid=>selSkills.includes(sid));
  if(allSelected){
    // 이 조합의 스킬만 제거 (다른 조합 스킬은 유지)
    selSkills = selSkills.filter(sid=>!combo.skills.includes(sid));
  } else {
    // 이 조합의 스킬을 기존에 추가 (중복 제거)
    combo.skills.forEach(sid=>{
      if(!selSkills.includes(sid)) selSkills.push(sid);
    });
  }
  // 자동 스킬 선택 OFF + 자동 스킬 비우기
  autoLoadedSkills = [];
  if(autoSkillMode){
    autoSkillMode = false;
    const toggle = document.getElementById('autoSkillToggle');
    if(toggle) toggle.classList.remove('on');
  }
  // 관련 도메인 자동 활성화
  if(catalog){
    Object.keys(catalog).forEach(did=>{
      const domainSkills = (catalog[did].skills||[]).map(s=>s.id);
      if(combo.skills.some(sid=>domainSkills.includes(sid)) && !selDomains.includes(did)){
        selDomains.push(did);
      }
    });
    renderTags();
  }
  renderSkills();
  updateLoaded();
  renderCombos();
  updateComboActiveMsg();
}
function updateComboActiveMsg(){
  const msgEl = document.getElementById('comboActiveMsg');
  const clearWrap = document.getElementById('comboClearWrap');
  if(!msgEl) return;
  const active = SKILL_COMBOS.filter(c=>c.skills.every(sid=>selSkills.includes(sid)));
  const hasAny = selSkills.length > 0;
  if(clearWrap) clearWrap.style.display = hasAny ? 'block' : 'none';
  if(active.length === 0){
    msgEl.style.display = hasAny ? 'block' : 'none';
    if(hasAny) msgEl.innerHTML = `🔗 총 <b>${selSkills.length}개</b> 스킬 선택됨`;
    return;
  }
  const labels = active.map(c=>c.label).join(' + ');
  msgEl.innerHTML = `🔗 <b>${active.length}개 조합 활성:</b> ${labels} — 총 <b>${selSkills.length}개</b> 스킬 선택됨`;
  msgEl.style.display = 'block';
}
function clearAllCombos(){
  selSkills = [];
  renderSkills();
  updateLoaded();
  renderCombos();
}

// 페이지 로드 시 콤보 렌더링
setTimeout(renderCombos, 500);

// 스킬 카드 자동 사용법 툴팁 생성
function _autoTip(id, desc){
  if(!desc) return '클릭하여 선택';
  // 에이전트 스킬: "agent-xxx"
  if(id.startsWith('agent-')){
    const role = id.replace('agent-','').replace(/-/g,' ');
    return `${desc}\n📌 이럴 때: 전문가 수준의 ${role} 작업이 필요할 때\n💡 사용법: "${role} 관점에서 분석해줘"`;
  }
  // 데이터베이스 스킬: "xxx-database"
  if(id.endsWith('-database')){
    const db = id.replace('-database','').replace(/-/g,' ');
    return `${desc}\n📌 이럴 때: ${db} 데이터를 검색하거나 조회할 때\n💡 사용법: "${db}에서 [키워드] 검색해줘"`;
  }
  // 시각화 스킬
  if(['matplotlib','seaborn','plotly','scientific-visualization'].includes(id)){
    return `${desc}\n📌 이럴 때: 데이터를 그래프/차트로 시각화할 때\n💡 사용법: "데이터를 [차트 종류]로 그려줘"`;
  }
  // 문서 생성 스킬
  if(['pptx','docx','xlsx','pdf','markdown-mermaid-writing','scientific-writing'].includes(id)){
    const fmt = {pptx:'PPT',docx:'Word',xlsx:'엑셀',pdf:'PDF'}[id] || '문서';
    const when = {pptx:'발표 자료가 필요할 때',docx:'보고서/문서를 작성할 때',xlsx:'스프레드시트를 만들거나 편집할 때',pdf:'PDF를 읽거나 생성할 때'}[id] || '문서를 작성할 때';
    return `${desc}\n📌 이럴 때: ${when}\n💡 사용법: "[내용]을 ${fmt}로 만들어줘"`;
  }
  // ML/통계 스킬
  if(['scikit-learn','pytorch-lightning','statistical-analysis','shap','umap-learn','statsmodels'].includes(id)){
    const when = {'scikit-learn':'분류/회귀/클러스터링 모델을 만들 때','pytorch-lightning':'딥러닝 모델을 학습할 때','statistical-analysis':'통계 검정/분석이 필요할 때','shap':'모델의 예측 이유를 설명할 때','umap-learn':'고차원 데이터를 2D로 축소할 때','statsmodels':'회귀/시계열 통계 모델링할 때'}[id] || '데이터 분석/모델링할 때';
    return `${desc}\n📌 이럴 때: ${when}\n💡 사용법: "이 데이터로 [분석/모델링] 해줘"`;
  }
  // 데이터 처리 스킬
  if(['polars','dask','vaex','exploratory-data-analysis','networkx'].includes(id)){
    const when = {polars:'대용량 CSV/데이터를 빠르게 처리할 때',dask:'메모리보다 큰 데이터를 처리할 때',vaex:'수억 건 데이터를 탐색할 때','exploratory-data-analysis':'데이터를 처음 받아서 전체 파악할 때',networkx:'관계/네트워크 데이터를 분석할 때'}[id];
    return `${desc}\n📌 이럴 때: ${when}\n💡 사용법: "이 데이터 분석해줘"`;
  }
  // 화학/분자 스킬
  if(['rdkit','datamol','deepchem','molfeat','matchms','medchem','diffdock','molecular-dynamics','torchdrug'].includes(id)){
    return `${desc}\n📌 이럴 때: 분자 구조 분석/약물 설계/화학 데이터 처리할 때\n💡 사용법: "[SMILES/분자명]을 분석해줘"`;
  }
  // 생물정보학 스킬
  if(['biopython','scanpy','pydeseq2','bioservices','anndata','scvelo','scvi-tools','pysam'].includes(id)){
    return `${desc}\n📌 이럴 때: 유전체/단백질/단일세포 데이터를 분석할 때\n💡 사용법: "[유전자명/서열]을 분석해줘"`;
  }
  // 디버깅 스킬
  if(['debugging','agent-debugger'].includes(id)){
    return `${desc}\n📌 이럴 때: 코드 에러를 찾거나 수정할 때\n💡 사용법: "이 에러 원인 찾아줘" + 에러 메시지 붙여넣기`;
  }
  // 가이드/메타 스킬
  if(['skill-creator','anthropic-prompt-engineer','openai-prompt-engineer','anthropic-architect','engineer-skill-creator'].includes(id)){
    const when = {'skill-creator':'새로운 Claude 스킬을 만들 때','anthropic-prompt-engineer':'Claude용 프롬프트를 최적화할 때','openai-prompt-engineer':'GPT용 프롬프트를 작성할 때','anthropic-architect':'AI 시스템 아키텍처를 설계할 때','engineer-skill-creator':'전문가 지식을 스킬로 변환할 때'}[id];
    return `${desc}\n📌 이럴 때: ${when}\n💡 사용법: 관련 요청을 하면 자동 활성화`;
  }
  // 양자 컴퓨팅
  if(['qiskit','cirq','pennylane','qutip'].includes(id)){
    return `${desc}\n📌 이럴 때: 양자 회로/시뮬레이션 작업할 때\n💡 사용법: "양자 회로 만들어줘"`;
  }
  // 기본: description 기반
  return `${desc}\n📌 이럴 때: ${desc} 작업이 필요할 때\n💡 사용법: 관련 질문을 입력하면 자동으로 활성화됩니다`;
}
const _skillTips = {
  'matplotlib':'그래프/차트를 그려줍니다\n예: "데이터를 막대 차트로 그려줘"',
  'seaborn':'예쁜 통계 차트를 만들어줍니다\n예: "상관관계 히트맵 그려줘"',
  'plotly':'인터랙티브 차트를 만듭니다\n예: "줌 가능한 꺾은선 차트 만들어줘"',
  'scikit-learn':'머신러닝 모델을 만들어줍니다\n예: "이 데이터로 예측 모델 만들어줘"',
  'statistical-analysis':'통계 분석을 해줍니다\n예: "평균, 분산, 상관관계 분석해줘"',
  'exploratory-data-analysis':'데이터를 탐색하고 요약합니다\n예: "CSV 데이터 전체 분석해줘"',
  'polars':'대용량 데이터를 빠르게 처리합니다\n예: "CSV 파일 읽고 필터링해줘"',
  'agent-data-scientist':'데이터 분석 전문가처럼 답합니다\n예: "이 데이터의 패턴을 찾아줘"',
  'agent-python-pro':'파이썬 코드를 작성해줍니다\n예: "파일 읽는 함수 만들어줘"',
  'scientific-writing':'논문/보고서를 작성합니다\n예: "분석 결과를 보고서로 정리해줘"',
  'pptx':'PPT 파일을 생성합니다\n예: "분석 결과를 PPT로 만들어줘"',
  'docx':'Word 문서를 만들어줍니다\n예: "보고서를 Word로 만들어줘"',
  'xlsx':'엑셀 파일을 처리합니다\n예: "엑셀 데이터 정리해줘"',
  'drawio-diagram':'다이어그램을 그려줍니다\n예: "시스템 구조도 그려줘"',
  'biopython':'유전자/단백질 분석을 합니다\n예: "DNA 서열 분석해줘"',
  'rdkit':'분자 구조를 분석합니다\n예: "화학 분자 그려줘"',
  'debugging':'에러를 찾아 고쳐줍니다\n예: "이 에러 원인 찾아줘"',
  'agent-debugger':'버그를 추적하고 수정합니다\n예: "traceback 분석해줘"',
  'logpresso-search':'로그프레소 데이터를 조회합니다\n예: "오늘 설비 로그 조회해줘"',
  'knowledge-search':'도메인 지식을 검색합니다\n예: "FAB 컬럼 정보 알려줘"',
  'agent-cloud-architect':'클라우드 아키텍처를 설계합니다\n예: "AWS 배포 구조 설계해줘"',
  'agent-llm-architect':'LLM 시스템을 설계합니다\n예: "RAG 파이프라인 설계해줘"',
  'markdown-mermaid-writing':'마크다운/다이어그램을 작성합니다\n예: "플로우차트 그려줘"',
  'scientific-visualization':'과학 데이터를 시각화합니다\n예: "실험 결과를 그래프로 보여줘"',
  'agent-sql-pro':'SQL 쿼리를 작성해줍니다\n예: "테이블 조인 쿼리 만들어줘"',
  'agent-fullstack-developer':'웹앱을 개발합니다\n예: "로그인 페이지 만들어줘"',
  'pdf':'PDF 파일을 처리합니다\n예: "PDF에서 텍스트 추출해줘"',
  'market-research-reports':'시장조사 보고서를 작성합니다\n예: "시장 트렌드 보고서 만들어줘"',
};
let currentSessionId = null;
let sessions = {};

// ===== Init =====
loadSessions();
// Restore last session or create new
const lastSessions = Object.values(sessions).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
if(lastSessions.length > 0){
  currentSessionId = lastSessions[0].id;
  const s = lastSessions[0];
  history = s.history || [];
  setTimeout(()=>{
    document.getElementById('msgs').innerHTML = s.msgsHtml || '';
    injectDeleteButtons();
    if(s.writingStyle){ document.getElementById('writingStyle').value = s.writingStyle; syncStyleDropToSidebar(); }
    if(s.systemPrompt) document.getElementById('systemPromptInput').value = s.systemPrompt;
    if(s.systemPromptId){ currentPromptId=s.systemPromptId; renderPromptChips(); }
    if(s.thinkMode!==undefined) document.getElementById('thinkToggle').checked=s.thinkMode;
    if(s.selFormat){ selFormat=s.selFormat; formatManualOverride=(selFormat!=='auto'); document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); syncFormatDropToChat(); }
    if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; }
    if(s.selEnvs){ selEnvs=s.selEnvs; renderEnvs(); updateStatus(); }
    else if(s.selEnv){ selEnvs=[s.selEnv]; renderEnvs(); updateStatus(); }  // 하위호환
    if(s.selDomains && s.selDomains.length>0){ selDomains=s.selDomains; renderTags(); renderSkills(); }
    if(s.selSkills && s.selSkills.length>0){ selSkills=s.selSkills; renderSkills(); updateLoaded(); }
    renderSessionList();
    // 페이지 로드 시 차트 재렌더링 (Chart.js 로드 대기)
    const _initCharts=()=>{
      document.querySelectorAll('#msgs canvas[data-chart-json]').forEach(cv=>{
        try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
        catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
      });
    };
    if(typeof Chart!=='undefined') _initCharts();
    else setTimeout(_initCharts, 500);
  }, 100);
} else {
  currentSessionId = 'sess_'+Date.now();
  sessions[currentSessionId] = { id:currentSessionId, name:'새 세션', history:[], msgsHtml:'', updatedAt:Date.now() };
  saveSessions();
}

Promise.all([
  fetch('/api/config').then(r=>r.json()),
  fetch('/api/skills').then(r=>r.json()),
]).then(([cfgData, skillData])=>{
  envs = cfgData.envs || {};
  hasToken = cfgData.has_token;
  catalog = skillData || {};
  try{ renderTokenStatus(); }catch(e){ console.error('renderTokenStatus:',e); }
  try{ renderEnvs(); }catch(e){ console.error('renderEnvs:',e); }
  try{ renderTags(); }catch(e){ console.error('renderTags:',e); }
  try{ renderSkills(); }catch(e){ console.error('renderSkills:',e); }
  try{ renderSessionList(); }catch(e){ console.error('renderSessionList:',e); }
}).catch(err=>{
  console.error('설정 로드 실패:', err);
  document.getElementById('tokenStatus').className='token-status missing';
  document.getElementById('tokenStatus').textContent='❌ 서버 연결 실패 - 새로고침 해주세요';
  document.getElementById('tokenBadge').className='status off';
  document.getElementById('tokenBadge').textContent='❌ 연결 실패';
});

// ===== 환경 선택 =====
function toggleEnvSection(type){
  const target = type==='Api' ? document.getElementById('envApiInner') : document.getElementById('envRowGguf');
  const arrow = document.getElementById('envArrow'+type);
  if(target.style.display==='none'){
    target.style.display='';
    arrow.textContent='▼';
  } else {
    target.style.display='none';
    arrow.textContent='▶';
  }
}
function _envType(id){ return id.startsWith('gguf-') ? 'gguf' : 'api'; }
function _makeEnvBtn(id, env, icons){
  const btn = document.createElement('div');
  const isSel = selEnvs.includes(id);
  btn.className = 'env-btn' + (isSel?' selected':'');
  const idx = isSel ? selEnvs.indexOf(id)+1 : 0;
  const badge = selEnvs.length > 1 && isSel ? `<span style="position:absolute;top:4px;right:6px;background:#6366f1;color:#fff;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:bold">${idx}</span>` : '';
  btn.style.position='relative';
  btn.innerHTML = `${badge}<div class="env-name">${icons[id]||'🔗'} ${env.name}</div><div class="env-model">${env.model}</div>`;
  btn.onclick = ()=>{
    // 단일 선택만 허용 (병렬 제거)
    if(selEnvs.includes(id)){
      selEnvs = ['auto'];
    } else {
      selEnvs = [id];
    }
    // 모델 타입에 따라 토큰 자동 설정
    autoApplyTokenSettings(id);
    renderEnvs(); updateStatus();
  };
  return btn;
}
function renderEnvs(){
  const rowAuto = document.getElementById('envRowAuto');
  const rowHigh = document.getElementById('envRowHigh');
  const rowMid = document.getElementById('envRowMid');
  const rowLow = document.getElementById('envRowLow');
  const rowVl = document.getElementById('envRowVl');
  const rowGguf = document.getElementById('envRowGguf');
  rowAuto.innerHTML=''; rowHigh.innerHTML=''; rowMid.innerHTML=''; rowLow.innerHTML=''; rowVl.innerHTML=''; rowGguf.innerHTML='';
  const icons = {'dev':'🧪','coder-480b':'🚀','coder-next':'⚡','coder-next-common':'⚡','common':'🌐','summary':'📊','vl-medium':'👁️','vl-fast':'👁️','vl-common':'👁️','vl-common-30b':'👁️','vl-small':'👁️'};
  for(const id of Object.keys(envs)){ if(id.startsWith('gguf-')) icons[id]='💻'; }
  // AUTO 버튼
  const autoBtn = document.createElement('div');
  autoBtn.className = 'env-btn' + (selEnvs.includes('auto')?' selected':'');
  autoBtn.style.cssText = 'flex:1 1 100%;max-width:100%';
  autoBtn.innerHTML = '<div class="env-name">🤖 AUTO</div><div class="env-model">자동 모델 선택</div>';
  autoBtn.onclick = ()=>{ selEnvs=['auto']; renderEnvs(); updateStatus(); };
  rowAuto.appendChild(autoBtn);
  // API / GGUF 분류
  // API 모델 tier 분류
  const highIds = new Set(['coder-480b','summary']);
  const lowIds = new Set(['common']);
  let hasApi=false, hasGguf=false, hasHigh=false, hasMid=false, hasLow=false, hasVl=false;
  for(const [id, env] of Object.entries(envs)){
    if(id.startsWith('gguf-')){
      rowGguf.appendChild(_makeEnvBtn(id, env, icons));
      hasGguf = true;
    } else if(id.startsWith('vl-')){
      rowVl.appendChild(_makeEnvBtn(id, env, icons));
      hasVl = true; hasApi = true;
    } else if(highIds.has(id)){
      rowHigh.appendChild(_makeEnvBtn(id, env, icons));
      hasHigh = true; hasApi = true;
    } else if(lowIds.has(id)){
      rowLow.appendChild(_makeEnvBtn(id, env, icons));
      hasLow = true; hasApi = true;
    } else {
      rowMid.appendChild(_makeEnvBtn(id, env, icons));
      hasMid = true; hasApi = true;
    }
  }
  document.getElementById('envHighLabel').style.display = hasHigh ? '' : 'none';
  document.getElementById('envMidLabel').style.display = hasMid ? '' : 'none';
  document.getElementById('envLowLabel').style.display = hasLow ? '' : 'none';
  document.getElementById('envVlLabel').style.display = hasVl ? '' : 'none';
  document.getElementById('envSectionApi').style.display = hasApi ? '' : 'none';
  document.getElementById('envSectionGguf').style.display = hasGguf ? '' : 'none';
}
function renderTokenStatus(){
  const el = document.getElementById('tokenStatus');
  const badge = document.getElementById('tokenBadge');
  if(hasToken){
    el.className='token-status ok';
    el.textContent='🔑 TOKEN.TXT 로드됨 - API 키 자동 적용';
    badge.className='status on';
    badge.textContent='🔑 토큰 OK';
  } else {
    el.className='token-status ok';
    el.textContent='ℹ️ TOKEN.TXT 미설정 - 폐쇄망 API는 토큰 없이 사용 가능합니다';
    badge.className='status on';
    badge.textContent='🔗 토큰 선택';
  }
}
function updateStatus(){
  const st = document.getElementById('status');
  const eb = document.getElementById('envStatusBadge');
  if(selEnvs.includes('auto')){
    st.className='status on';
    st.textContent='🤖 AUTO (자동 선택)';
    if(eb) eb.innerHTML='( 🤖 AUTO - 단일/병렬 자동 결정 )';
  } else if(selEnvs.length >= 2){
    const type = selEnvs[0].startsWith('gguf-') ? 'GGUF' : 'API';
    const names = selEnvs.map(e=> envs[e] ? envs[e].name : e).join(', ');
    st.className='status on';
    st.textContent='🔀 병렬 ' + selEnvs.length + '개 모델';
    if(eb) eb.innerHTML=`( 🔀 병렬 ${selEnvs.length}개 ${type}: ${names} )`;
  } else if(selEnvs.length === 1 && envs[selEnvs[0]]){
    const type = selEnvs[0].startsWith('gguf-') ? 'GGUF' : 'API';
    st.className='status on';
    st.textContent='🟢 ' + envs[selEnvs[0]].name;
    if(eb) eb.innerHTML=`( 🟢 단일 ${type}: ${envs[selEnvs[0]].name} )`;
  } else {
    st.className='status off';
    st.textContent='⚪ 환경 미선택';
    if(eb) eb.innerHTML='';
  }
}

// ===== Tags =====
function renderTags(){
  const row = document.getElementById('tagRow');
  row.innerHTML = '';
  for(const [id, d] of Object.entries(catalog)){
    const t = document.createElement('div');
    t.className = 'tag' + (selDomains.includes(id)?' selected':'');
    t.style.setProperty('--c', d.color);
    t.style.setProperty('--bg', d.color+'18');
    t.style.setProperty('--fg', d.color);
    t.textContent = d.icon + ' ' + d.label;
    t.onclick = ()=>{ toggleDomain(id); t.classList.toggle('selected'); renderSkills(); };
    row.appendChild(t);
  }
}
function toggleDomain(id){
  if(selDomains.includes(id)) selDomains=selDomains.filter(x=>x!==id);
  else selDomains.push(id);
}

// ===== Skills =====
let autoLoadedSkills = [];  // 자동 로드된 스킬 목록 (시각적 표시용)

function renderSkills(){
  const grid = document.getElementById('skillGrid');
  grid.innerHTML = '';
  selDomains.forEach(did=>{
    const d = catalog[did];
    if(!d) return;
    d.skills.forEach(s=>{
      const c = document.createElement('div');
      const isAutoLoaded = autoLoadedSkills.includes(s.id);
      c.className = 'skill-card' + (selSkills.includes(s.id)?' selected':'') + (isAutoLoaded?' auto-loaded':'') + (!s.available?' unavailable':'');
      const desc = s.desc ? s.desc : '';
      const avail = s.available ? '✅' : '❌';
      // extras: 스크립트/레퍼런스 개수
      let extras = [];
      if(s.scripts > 0) extras.push(`🐍${s.scripts}`);
      if(s.references > 0) extras.push(`📄${s.references}`);
      if(s.assets > 0) extras.push(`📦${s.assets}`);
      const extrasHtml = extras.length ? `<div class="extras">${extras.join(' ')}</div>` : '';
      const autoBadge = isAutoLoaded ? '<span class="badge" style="background:#f59e0b">🧠자동</span>' : '';
      const tipText = _skillTips[s.id] || _autoTip(s.id, desc);
      c.title = `${s.name}\n${tipText}\n\n[클릭=선택, Shift+클릭=상세]`;
      c.innerHTML = `<div class="sn">${avail} ${s.name}</div><div class="sd">${desc}</div>${extrasHtml}${autoBadge}`;
      if(s.available){
        c.onclick = (e)=>{
          // Shift+클릭 = 상세 패널
          if(e.shiftKey || (extras.length > 0 && e.detail === 2)){
            openSkillDetail(s.id);
            return;
          }
          if(selSkills.includes(s.id)) selSkills=selSkills.filter(x=>x!==s.id);
          else selSkills.push(s.id);
          // 수동 선택 시 자동 스킬 전부 해제
          if(selSkills.length > 0){ autoLoadedSkills = []; }
          renderSkills();
          updateLoaded();
        };
        // 우클릭 = 상세 패널
        c.oncontextmenu = (e)=>{
          e.preventDefault();
          openSkillDetail(s.id);
        };
      }
      grid.appendChild(c);
    });
  });
  updateLoaded();
}
function updateLoaded(){
  const totalCount = selSkills.length + autoLoadedSkills.length;
  const countEl = document.getElementById('loadedCount');
  if(countEl) countEl.textContent = totalCount;
  const el = document.getElementById('loadedSkillsList');
  if(!el) return;  // 사이드바에서 제거된 경우
  if(totalCount === 0 && dismissedAutoSkills.size === 0){
    el.innerHTML = '<div style="color:#bbb">선택된 스킬 없음</div>';
    return;
  }
  // 수동 선택 스킬 (체크박스로 해제 가능)
  let html = selSkills.map(s =>
    `<label style="display:flex;align-items:center;gap:4px;padding:2px 0;cursor:pointer;">
      <input type="checkbox" checked onchange="unloadSkill('${s}')" style="accent-color:#4f46e5;cursor:pointer;">
      <span style="font-size:12px">✅ ${s}</span>
    </label>`
  ).join('');
  // 자동 추천 스킬 (1회성, 📌으로 수동 고정 / ✕로 해제)
  if(autoLoadedSkills.length > 0){
    html += autoLoadedSkills.map(s =>
      `<div style="display:flex;align-items:center;justify-content:space-between;padding:2px 0;opacity:.8;">
        <span style="font-size:12px">🧠 ${s}</span>
        <span style="display:flex;gap:2px;">
          <button onclick="pinAutoSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#4f46e5;padding:0 2px;" title="수동 고정">📌</button>
          <button onclick="dismissAutoSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#ef4444;padding:0 2px;" title="해제 (다시 추천 안함)">✕</button>
        </span>
      </div>`
    ).join('');
  }
  // 해제된 자동 스킬 (블랙리스트) 표시
  if(dismissedAutoSkills.size > 0){
    html += `<div style="margin-top:6px;padding-top:6px;border-top:1px dashed #e5e3de;">
      <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">🚫 해제됨 (자동추천 제외)</div>`;
    html += [...dismissedAutoSkills].map(s =>
      `<div style="display:flex;align-items:center;justify-content:space-between;padding:1px 0;opacity:.5;">
        <span style="font-size:11px;text-decoration:line-through;color:#9ca3af;">${s}</span>
        <button onclick="restoreDismissedSkill('${s}')" style="background:none;border:none;cursor:pointer;font-size:10px;color:#22c55e;padding:0 2px;" title="복구 (자동추천 허용)">♻️</button>
      </div>`
    ).join('');
    if(dismissedAutoSkills.size >= 2){
      html += `<button onclick="clearDismissed()" style="width:100%;margin-top:3px;padding:2px 0;font-size:10px;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;cursor:pointer;color:#ef4444;">전체 복구</button>`;
    }
    html += `</div>`;
  }
  if(selSkills.length + autoLoadedSkills.length >= 2){
    html += `<button onclick="unloadAllSkills()" style="width:100%;margin-top:4px;padding:3px 0;font-size:11px;background:#f5f5f0;border:1px solid #e5e3de;border-radius:4px;cursor:pointer;color:#888;">전체 해제</button>`;
  }
  el.innerHTML = html;
}
let dismissedAutoSkills = new Set();  // 사용자가 해제한 자동 스킬 (세션 내 재로드 방지)

function unloadSkill(id){
  selSkills = selSkills.filter(x => x !== id);
  // 자동 스킬이면 해제 블랙리스트에 추가
  if(autoLoadedSkills.includes(id)){
    autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
    dismissedAutoSkills.add(id);
  }
  renderSkills();
  updateLoaded();
}
function unloadAllSkills(){
  // 자동 스킬 모두 블랙리스트에 추가
  autoLoadedSkills.forEach(id => dismissedAutoSkills.add(id));
  selSkills = [];
  autoLoadedSkills = [];
  renderSkills();
  updateLoaded();
}
function pinAutoSkill(id){
  // 자동 스킬을 수동 고정으로 전환 → 블랙리스트에서 제거
  autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
  dismissedAutoSkills.delete(id);
  if(!selSkills.includes(id)) selSkills.push(id);
  renderSkills();
  updateLoaded();
}
function dismissAutoSkill(id){
  // 자동 스킬을 블랙리스트로 이동 (다시 추천 안함)
  autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
  dismissedAutoSkills.add(id);
  renderSkills();
  updateLoaded();
}
function dismissPreviewSkill(id, btn){
  // 미리보기에서 자동 스킬 제외 (전송 시 사용 안함)
  dismissedAutoSkills.add(id);
  autoLoadedSkills = autoLoadedSkills.filter(x => x !== id);
  const badge = btn.closest('.auto-skill-badge');
  if(badge) badge.remove();
  const prev = document.getElementById('autoSkillPreview');
  if(prev && prev.querySelectorAll('.auto-skill-badge').length === 0){
    prev.classList.remove('show');
  }
  updateLoaded();
}
function restoreDismissedSkill(id){
  // 블랙리스트에서 복구 (다음 자동 추천 허용)
  dismissedAutoSkills.delete(id);
  updateLoaded();
}
function clearDismissed(){
  // 블랙리스트 전체 초기화
  dismissedAutoSkills.clear();
  updateLoaded();
}

// ===== Effort =====
function updateEffort(){
  const s = document.getElementById('effortSlider');
  effort = parseInt(s.value);
  s.style.setProperty('--val', (effort/3*100)+'%');
  for(let i=0;i<=3;i++) document.getElementById('e'+i).className = i===effort?'active':'';
}
updateEffort();

// ===== Format =====
function selFmt(el){
  document.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('selected'));
  el.classList.add('selected');
  selFormat = el.dataset.f;
  formatManualOverride = (selFormat !== 'auto');  // auto 선택 시 자동 모드로 복귀
  syncFormatDropToChat();
}

// ===== Quick prompt =====
function qp(t){ document.getElementById('input').value=t; }

// ===== Auto Skill =====
function toggleAutoSkill(){
  autoSkillMode = !autoSkillMode;
  const el = document.getElementById('autoSkillToggle');
  el.classList.toggle('on', autoSkillMode);
  if(!autoSkillMode){
    document.getElementById('autoSkillPreview').classList.remove('show');
    document.getElementById('autoSkillPreview').innerHTML = '';
  }
}
// 초기 상태 반영
setTimeout(()=>{
  document.getElementById('autoSkillToggle').classList.toggle('on', autoSkillMode);
}, 200);

// ── 토큰 설정 (메인 페이지) ──
function toggleMainTokenSettings(){
  const panel = document.getElementById('mainTokenPanel');
  if(!panel) return;
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
  if(panel.style.display === 'block') loadMainTokenSettings();
}

async function loadMainTokenSettings(){
  try{
    const r = await fetch('/api/config/tokens');
    const ts = await r.json();
    const map = {
      mainAgentMaxTokens: 'agent_max_tokens',
      mainSynthMaxTokens: 'synth_max_tokens',
      mainNCtx: 'default_n_ctx',
      mainReplyCap: 'gguf_reply_cap',
    };
    for(const [elId, key] of Object.entries(map)){
      const sel = document.getElementById(elId);
      if(sel && ts[key]) sel.value = String(ts[key]);
    }
  }catch(e){}
}

function updateMainTokenSetting(){
  // 드롭다운 변경 시 UI만 업데이트 (저장 버튼 눌러야 서버 반영)
  const status = document.getElementById('tokenSaveStatus');
  if(status){ status.style.display='none'; }
}

async function saveTokenSettings(){
  const data = {
    agent_max_tokens: parseInt(document.getElementById('mainAgentMaxTokens')?.value || '8192'),
    synth_max_tokens: parseInt(document.getElementById('mainSynthMaxTokens')?.value || '16384'),
    default_n_ctx: parseInt(document.getElementById('mainNCtx')?.value || '4096'),
    gguf_reply_cap: parseInt(document.getElementById('mainReplyCap')?.value || '4096'),
  };
  try{
    await fetch('/api/config/tokens',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const status = document.getElementById('tokenSaveStatus');
    if(status){ status.style.display='inline'; setTimeout(()=>{ status.style.display='none'; }, 3000); }
  }catch(e){ alert('저장 실패: ' + e); }
}


// 모델 타입에 따라 토큰 자동 설정
async function autoApplyTokenSettings(envId){
  const isGguf = envId.startsWith('gguf-');
  const isAuto = (envId === 'auto');
  let data;
  if(isGguf){
    // GGUF: 빠른 설정 (RTX 3060 12GB 최적화)
    data = { default_n_ctx: 4096, gguf_reply_cap: 4096 };
    const nctx = document.getElementById('mainNCtx');
    const cap = document.getElementById('mainReplyCap');
    if(nctx) nctx.value = '4096';
    if(cap) cap.value = '4096';
  } else if(!isAuto){
    // API: 기본 설정
    data = { agent_max_tokens: 8192, synth_max_tokens: 16384 };
    const agent = document.getElementById('mainAgentMaxTokens');
    const synth = document.getElementById('mainSynthMaxTokens');
    if(agent) agent.value = '8192';
    if(synth) synth.value = '16384';
  } else {
    return; // auto는 변경 안 함
  }
  try{
    await fetch('/api/config/tokens',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  }catch(e){}
}

// 입력 중 자동 스킬 미리보기 (디바운스)
let autoSkillTimer = null;
document.addEventListener('DOMContentLoaded', ()=>{
  const inp = document.getElementById('input');
  if(inp) inp.addEventListener('input', ()=>{
    if(!autoSkillMode || selSkills.length > 0) return;
    clearTimeout(autoSkillTimer);
    autoSkillTimer = setTimeout(async ()=>{
      const q = inp.value.trim();
      if(q.length < 4) { document.getElementById('autoSkillPreview').classList.remove('show'); return; }
      try{
        const r = await fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,max_skills:3,history:history})});
        const d = await r.json();
        const prev = document.getElementById('autoSkillPreview');
        if(d.skills && d.skills.length > 0){
          prev.innerHTML = '🧠 자동 추천: ' + d.skills.filter(s=>!dismissedAutoSkills.has(s.id)).map(s=>{
            const tip = _skillTips[s.id] || s.desc || s.id;
            return `<span class="auto-skill-badge" title="${tip.replace(/"/g,'&quot;').replace(/\n/g,'&#10;')}">${s.id} (${s.score})<button onclick="dismissPreviewSkill('${s.id}',this)" style="background:none;border:none;cursor:pointer;font-size:10px;color:#ef4444;padding:0 2px;margin-left:4px;" title="제외">✕</button></span>`;
          }).join('');
          prev.classList.add('show');
        } else {
          prev.classList.remove('show');
        }
      }catch(e){}
    }, 500);
  });
});

// ===== Chat =====
let chatAbort = null;  // AbortController for current request
let isSending = false;

function handleSendStop(){
  if(isSending){
    stopGeneration();
  } else {
    send();
  }
}
function stopGeneration(){
  if(chatAbort){
    chatAbort.abort();
    chatAbort = null;
  }
  // 백엔드 GGUF 중단 요청
  fetch('/api/chat/stop',{method:'POST'}).catch(()=>{});
  isSending = false;
  const btn = document.getElementById('sendBtn');
  btn.textContent = '▶';
  btn.classList.remove('stop-mode');
  btn.disabled = false;
  // typing indicator 제거
  document.querySelectorAll('.typing-wrap').forEach(el=>el.remove());
  addMsg('assistant','⏹️ 응답이 중지되었습니다.');
  saveCurrentSession();
}

/* ── PPT 제안 배너 ── */
let _pptBannerShownCount = 0;
const _pptSuggestItems = [
  {icon:'📊', title:'차트/그래프를 포함할까요?', hint:'"매출 데이터를 막대 차트로 넣어줘"', example:'차트/그래프도 포함해서 PPT 만들어줘'},
  {icon:'📝', title:'그래프 없이 깔끔하게?', hint:'"그래프 없이 텍스트와 표로만 PPT 만들어줘"', example:'그래프 없이 텍스트와 표 위주로 PPT 만들어줘'},
  {icon:'📋', title:'표(Table)를 넣어볼까요?', hint:'"비교 데이터를 표로 정리해서 넣어줘"', example:'데이터를 표로 정리해서 PPT에 넣어줘'},
  {icon:'🖼️', title:'이미지/다이어그램도 가능해요!', hint:'"구조도를 슬라이드에 추가해줘"', example:'다이어그램도 포함해서 PPT 만들어줘'},
  {icon:'📈', title:'데이터 시각화를 추가할까요?', hint:'"트렌드를 꺾은선 그래프로 보여줘"', example:'데이터를 시각화해서 PPT에 넣어줘'},
  {icon:'🎨', title:'심플한 디자인으로?', hint:'"심플하게 핵심 내용만 PPT 만들어줘"', example:'디자인 심플하게 핵심만 PPT 만들어줘'},
  {icon:'🍩', title:'원형/도넛 차트는 어때요?', hint:'"비율을 도넛 차트로 만들어줘"', example:'비율 데이터를 원형 차트로 PPT에 넣어줘'},
  {icon:'📉', title:'비교 차트를 넣어볼까요?', hint:'"전년 대비 성장률을 비교 차트로"', example:'비교 차트를 포함해서 PPT 만들어줘'},
];

const _pptNoVisualItems = [
  {icon:'📝', title:'그래프 없이도 만들 수 있어요!', hint:'"그래프 없이 텍스트와 표로만 PPT 만들어줘"', example:'그래프 없이 텍스트와 표 위주로 PPT 다시 만들어줘'},
  {icon:'🎨', title:'심플한 PPT는 어때요?', hint:'"심플하게 핵심 내용만 PPT로"', example:'디자인 심플하게 핵심만 PPT 만들어줘'},
];
const _pptVisualItems = [
  {icon:'📊', title:'그래프를 추가해볼까요?', hint:'"차트/그래프도 포함해서 PPT 만들어줘"', example:'차트/그래프도 포함해서 PPT 다시 만들어줘'},
  {icon:'📈', title:'데이터 시각화는 어때요?', hint:'"데이터를 시각화해서 PPT에 넣어줘"', example:'데이터를 시각화해서 PPT에 넣어줘'},
];

function _showPptSuggestBanner(mode){
  if(_pptBannerShownCount >= 3) return;
  _pptBannerShownCount++;
  const area = document.getElementById('pptSuggestArea');
  if(!area) return;
  let pool;
  if(mode === 'no-visual') pool = _pptNoVisualItems;
  else if(mode === 'visual') pool = _pptVisualItems;
  else pool = _pptSuggestItems;
  const item = pool[Math.floor(Math.random()*pool.length)];
  const banner = document.createElement('div');
  banner.className = 'ppt-suggest-banner';
  banner.innerHTML = `<span class="ppt-banner-icon">${item.icon}</span>`
    + `<div class="ppt-banner-title">${item.title}</div>`
    + `<div class="ppt-banner-hint"><em onclick="_usePptSuggestion(this,'${item.example.replace(/'/g,"\\'")}')">${item.hint}</em> 처럼 말해보세요!</div>`
    + `<button class="ppt-banner-close" onclick="_closePptBanner(this)" title="닫기">✕</button>`;
  area.innerHTML = '';
  area.appendChild(banner);
  setTimeout(()=>{
    if(banner.parentNode){
      banner.classList.add('hiding');
      setTimeout(()=>{ if(banner.parentNode) banner.remove(); }, 400);
    }
  }, 8000);
}

function _closePptBanner(btn){
  const banner = btn.closest('.ppt-suggest-banner');
  if(banner){ banner.classList.add('hiding'); setTimeout(()=>banner.remove(), 400); }
}

function _usePptSuggestion(el, text){
  const input = document.getElementById('input');
  if(input){ input.value = text; input.focus(); input.style.height='auto'; input.style.height=input.scrollHeight+'px'; }
  const banner = el.closest('.ppt-suggest-banner');
  if(banner){ banner.classList.add('hiding'); setTimeout(()=>banner.remove(), 400); }
}

const _pptKeywords = /PPT|ppt|파워포인트|프레젠테이션|발표자료|슬라이드\s*만들|피피티/i;

async function send(){
  if(isSending) return;  // 응답 중이면 중복 전송 차단

  const el=document.getElementById('input');
  const text=el.value.trim();
  if(!text && chatPendingFiles.length === 0) return;

  if(text && _pptKeywords.test(text)){
    const hasVisual = /차트|그래프|표|table|chart|graph|시각화|도넛|원형/i.test(text);
    const hasNoVisual = /그래프\s*없|차트\s*없|텍스트\s*위주|텍스트만|심플하게/i.test(text);
    _showPptSuggestBanner(hasVisual && !hasNoVisual ? 'visual' : hasNoVisual && !hasVisual ? 'no-visual' : null);
  }
  if(!selEnvs||selEnvs.length===0){alert('먼저 위에서 LLM 환경을 선택해주세요.');return;}

  // 매 질문마다 이전 자동 스킬 초기화 → 새 질문/파일에 맞게 재감지
  // 전송 전 사용자가 dismiss한 스킬을 보존 (✕ 클릭한 것 반영)
  const currentDismissed = new Set(dismissedAutoSkills);
  autoLoadedSkills = [];
  dismissedAutoSkills.clear();
  updateLoaded();  // UI도 즉시 초기화

  // 첨부파일이 있으면 먼저 업로드 (파일 확장자 기반 스킬이 autoLoadedSkills에 추가됨)
  let attachedNames = [];
  if(chatPendingFiles.length > 0){
    attachedNames = await uploadChatPendingFiles();
  }

  // 스킬 구성: 수동 선택 시 해당 스킬만 사용, 미선택 시 자동 추천
  let skillsToUse = [...selSkills];
  let autoLoaded = [];

  if(selSkills.length === 0){
    // 수동 선택 없음 → 파일 기반 자동 + 질문 기반 자동 추천
    autoLoaded = [...autoLoadedSkills];
    const manualSet = new Set(skillsToUse);
    autoLoaded = autoLoaded.filter(id => !manualSet.has(id) && !currentDismissed.has(id));
    skillsToUse = [...skillsToUse, ...autoLoaded];

    // 자동 스킬 모드: 질문 분석으로 추가 보충
    if(autoSkillMode){
      try{
        const currentCount = skillsToUse.length;
        const maxAuto = currentCount === 0 ? 7 : Math.max(0, 10 - currentCount);
        if(maxAuto > 0){
          const ar = await fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:text,max_skills:maxAuto,history:history})});
          const ad = await ar.json();
          if(ad.skills && ad.skills.length > 0){
            const usedSet = new Set(skillsToUse);
            const newAuto = ad.skills.map(s=>s.id).filter(id=>!usedSet.has(id) && !currentDismissed.has(id));
            autoLoaded = [...autoLoaded, ...newAuto];
            skillsToUse = [...skillsToUse, ...newAuto];
          }
        }
      }catch(e){}
    }
  }
  // 수동 1개 선택 → 단일 에이전트, 수동 2개+ → 병렬 에이전트 (자동 추가 없음)

  // 첨부파일 표시 + 메시지
  let displayText = text || '';
  if(attachedNames.length > 0){
    const attachInfo = '📎 ' + attachedNames.join(', ');
    displayText = displayText ? attachInfo + '\n' + displayText : attachInfo;
  }
  addMsg('user', displayText);
  history.push({role:'user',content: displayText});
  el.value=''; el.style.height='auto';
  document.getElementById('autoSkillPreview').classList.remove('show');

  // 자동 로드 안내 (selSkills에는 추가하지 않음 - 1회성 사용)
  autoLoadedSkills = autoLoaded;
  renderSkills();
  updateLoaded();
  if(autoLoaded.length > 0){
    const manualCount = selSkills.length;
    const info = manualCount > 0 ? ` (수동 ${manualCount}개 + 자동 ${autoLoaded.length}개)` : '';
    addMsg('assistant', '🧠 자동 스킬: ' + autoLoaded.join(', ') + info);
  }

  const typing=addTyping();
  isSending = true;
  const btn = document.getElementById('sendBtn');
  btn.textContent = '⏹';
  btn.classList.add('stop-mode');
  btn.disabled = false;

  chatAbort = new AbortController();

  try{
    const resp=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal: chatAbort.signal,
      body:JSON.stringify(Object.assign({
        env: selEnvs,
        messages:history, skills:skillsToUse, effort,
        format: formatManualOverride ? selFormat : 'auto',
        writing_style: styleManualOverride ? document.getElementById('writingStyle').value.trim() : 'auto',
        system_prompt:document.getElementById('systemPromptInput').value.trim(),
        think_mode: document.getElementById('thinkToggle').checked,
        ppt_style: activePptStyle ? (PPT_STYLE_PROMPTS[activePptStyle]||'') : '',
        ppt_ref_design: pptRefDesign,
        user_id: currentUser ? currentUser.id : null,
      }, maxTokens > 0 ? {max_tokens: maxTokens} : {}))
    });
    const data=await resp.json();
    typing.remove();

    if(data.error){
      addMsg('assistant','❌ '+data.error);
    } else {
      let info = '';
      // 모델 이름 표시 (auto/수동)
      let _se = selEnvs[0] || 'auto';
      let modelName = data.model_used || (envs[_se] ? envs[_se].name : (selEnvs.length>=2 ? selEnvs.length+'개 병렬' : _se));
      if(data.loaded_skills && data.loaded_skills.length > 0){
        let extra = data.tokens_budget ? ` [${data.tokens_budget}]` : '';
        let mode = autoLoaded.length > 0 ? '🧠자동' : '✅수동';
        info = `\n[${mode} 스킬: ${data.loaded_skills.join(', ')}] [${modelName}] (${data.system_prompt_length ?? 0}자)${extra}`;
      }
      // 자동 라우팅 표시
      if(data.auto_routed){
        info += ` [🤖 자동: ${data.model_used} (${data.route_reason})]`;
      }
      // 폴백 표시
      if(data.fallback_used){
        info += ` [⚠️ 대체: ${data.fallback_from} → ${data.model_used}]`;
      }
      // 병렬 에이전트 표시
      if(data.parallel_agents && data.parallel_agents >= 2){
        let pGroups = (data.parallel_groups || []).join(', ');
        let pModels = (data.parallel_models || []).join(', ');
        let pSynth = data.parallel_synthesis === 'fallback_concat' ? '(단순연결)' : '(합성)';
        info += ` [🔀 병렬 ${data.parallel_agents}에이전트: ${pGroups}] [모델: ${pModels}] ${pSynth}`;
        if(data.parallel_failed > 0) info += ` [⚠️ ${data.parallel_failed}개 실패]`;
      }
      // 자동 형식/스타일 표시
      if(data.auto_format || data.auto_style){
        let fmtNames = {'code':'코드','code-fix':'코드수정','analysis':'분석','report':'보고서','step-by-step':'단계별'};
        let parts = [];
        if(data.auto_format) parts.push('형식:' + (fmtNames[data.auto_format]||data.auto_format));
        if(data.auto_style) parts.push('스타일:자동');
        info += ` [📝 ${parts.join(' / ')}]`;
      }
      let truncWarn = data.truncated ? '\n\n⚠️ **응답이 토큰 한도('+maxTokens+')에 도달하여 잘렸습니다.** "계속 이어서 작성해줘"라고 입력하면 이어서 받을 수 있습니다.' : '';
      const assistantDisplayText = data.content + truncWarn + info;
      const assistantRawForDetect = data.content + truncWarn;
      addMsg('assistant', assistantDisplayText, assistantRawForDetect);
      history.push({role:'assistant',content:data.content});
      // markdown-mermaid-writing 또는 md-to-html 스킬: MD/HTML 다운로드 버튼 추가
      // selSkills(수동) 또는 autoLoadedSkills(자동) 모두 체크
      const _mdSkillActive = ['markdown-mermaid-writing','md-to-html'].some(s => selSkills.includes(s) || autoLoadedSkills.includes(s));
      if(_mdSkillActive && !data.content.includes('```markdown')){
        appendMdDownloadBar(data.content, data.html_download_url);
      }
    }
  }catch(e){
    typing.remove();
    if(e.name === 'AbortError'){
      // 사용자가 중지한 경우 - stopGeneration()에서 이미 처리됨
    } else {
      addMsg('assistant','❌ 서버 연결 실패: '+e.message);
    }
  }
  isSending = false;
  chatAbort = null;
  btn.textContent = '▶';
  btn.classList.remove('stop-mode');
  btn.disabled = false;
  const inp=document.getElementById('input');
  inp.focus();
  inp.style.height='auto';
  saveCurrentSession();
}

function renderMd(text){
  // 0) <think> 블록을 접이식 사고과정 박스로 변환 (HTML escape 전)
  text=text.replace(/<think>([\s\S]*?)<\/think>\s*/g, function(_,content){
    const escaped=content.trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    return `<details class="think-box"><summary>💭 사고 과정</summary><div class="think-content">${escaped}</div></details>`;
  });
  // 닫히지 않은 <think> 태그도 처리 (응답 잘림 대비)
  text=text.replace(/<think>([\s\S]*)$/g, function(_,content){
    const escaped=content.trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    return `<details class="think-box"><summary>💭 사고 과정 (진행중...)</summary><div class="think-content">${escaped}</div></details>`;
  });
  // 1) escape HTML (think-box는 이미 처리했으므로 보호)
  const thinkBlocks=[];
  text=text.replace(/<details class="think-box">[\s\S]*?<\/details>/g, m=>{thinkBlocks.push(m);return `__THINK_${thinkBlocks.length-1}__`;});
  let s=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // 2) code blocks (```lang\n...``` 또는 ```lang코드...``` 모두 지원)
  // 2a) drawio 코드블록은 특별 처리
  s=s.replace(/```drawio\s*([\s\S]*?)```/g,(_,code)=>{
    const escapedXml = code.trim();
    // HTML escape된 상태를 원본 XML로 복원 (btoa용)
    const realXml = escapedXml.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const xmlB64 = u2b(realXml);
    return `<div class="drawio-block">
      <div class="drawio-header">
        <span>📐 Draw.io 다이어그램</span>
        <div class="drawio-actions">
          <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
          <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
          <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
        </div>
      </div>
      <pre class="drawio-preview"><code class="language-xml">${escapedXml}</code></pre>
    </div>`;
  });
  // 2a-2) chart 코드블록 → Chart.js 인터랙티브 그래프 렌더링
  s=s.replace(/```chart\s*([\s\S]*?)```/g,(_,code)=>{
    const trimmed=code.trim();
    const raw=trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const chartId='chart_'+Math.random().toString(36).substr(2,9);
    const b64=u2b(raw);
    return `<div class="chart-block">
      <div class="chart-header">
        <span>📊 인터랙티브 차트</span>
        <div class="chart-actions">
          <button class="chart-btn" onclick="downloadChartPng('${chartId}')">📥 PNG 저장</button>
          <button class="chart-btn" onclick="copyChartJson(this)" data-chart="${b64}">📋 JSON 복사</button>
        </div>
      </div>
      <div class="chart-canvas-wrap"><canvas id="${chartId}" data-chart-json="${b64}"></canvas></div>
      <details class="chart-raw-toggle"><summary>📝 원본 JSON 보기</summary><pre>${trimmed}</pre></details>
    </div>`;
  });
  // 2b) markdown 코드블록 → MD 다운로드 UI
  s=s.replace(/```markdown\s*([\s\S]*?)```/g,(_,code)=>{
    const trimmed=code.trim();
    const mdB64=u2b(trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"'));
    return `<div class="md-report-block">
      <div class="md-report-header">
        <span>📋 마크다운 문서</span>
        <div class="md-report-actions">
          <button class="md-report-btn" onclick="copyMdContent(this)" data-md="${mdB64}">📋 복사</button>
          <button class="md-report-btn md-download" onclick="downloadMarkdown(this)" data-md="${mdB64}">📥 MD 다운로드</button>
        </div>
      </div>
      <div class="md-report-preview">${renderMdPreview(trimmed)}</div>
      <details class="md-raw-toggle"><summary>📝 원본 마크다운 보기</summary><pre><code class="language-markdown">${trimmed}</code></pre></details>
    </div>`;
  });
  // 2c) 일반 코드블록 — xml 블록 안에 mxfile이 있으면 drawio로 자동 변환
  s=s.replace(/```(\w*)\s*([\s\S]*?)```/g,(_,lang,code)=>{
    const trimmed=code.trim();
    // 코드블록 안에 <mxfile 또는 <mxGraphModel 있으면 drawio UI로 변환
    if(trimmed.includes('&lt;mxfile') || trimmed.includes('&lt;mxGraphModel')){
      const realXml=trimmed.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
      const xmlB64=u2b(realXml);
      return `<div class="drawio-block">
        <div class="drawio-header">
          <span>📐 Draw.io 다이어그램</span>
          <div class="drawio-actions">
            <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
            <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
            <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
          </div>
        </div>
        <pre class="drawio-preview"><code class="language-xml">${trimmed}</code></pre>
      </div>`;
    }
    const cls=lang?` class="language-${lang}"`:'';
    return `<pre><code${cls}>${trimmed}</code></pre>`;
  });
  // 2c) raw mxfile XML (코드블록 밖) 자동 감지 → drawio UI로 변환
  s=s.replace(/(&lt;mxfile[\s\S]*?&lt;\/mxfile&gt;)/g,(_,xmlEsc)=>{
    const realXml=xmlEsc.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"');
    const xmlB64=u2b(realXml);
    return `<div class="drawio-block">
      <div class="drawio-header">
        <span>📐 Draw.io 다이어그램 (자동 감지)</span>
        <div class="drawio-actions">
          <button class="drawio-btn" onclick="copyDrawioXml(this)" data-xml="${xmlB64}">📋 XML 복사</button>
          <button class="drawio-btn" onclick="downloadDrawio(this)" data-xml="${xmlB64}">💾 .drawio 저장</button>
          <button class="drawio-btn" onclick="openInDrawio(this)" data-xml="${xmlB64}">🔗 Draw.io에서 열기</button>
        </div>
      </div>
      <pre class="drawio-preview"><code class="language-xml">${xmlEsc}</code></pre>
    </div>`;
  });
  // 3) tables
  s=s.replace(/((?:^\|.+\|[ ]*\n){2,})/gm, function(tbl){
    const rows=tbl.trim().split('\n').filter(r=>r.trim());
    if(rows.length<2) return tbl;
    const parseRow=r=>r.replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());
    const hdr=parseRow(rows[0]);
    // skip separator row
    let startIdx=1;
    if(/^[\s|:-]+$/.test(rows[1])) startIdx=2;
    let h='<table><thead><tr>'+hdr.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
    for(let i=startIdx;i<rows.length;i++){
      const cells=parseRow(rows[i]);
      h+='<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>';
    }
    return h+'</tbody></table>';
  });
  // 4) headings
  s=s.replace(/^#### (.+)$/gm,'<h4>$1</h4>');
  s=s.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  s=s.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  s=s.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  // 5) hr
  s=s.replace(/^---+$/gm,'<hr>');
  // 6) blockquote
  s=s.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  // 7) unordered list
  s=s.replace(/(^[\-\*] .+\n?)+/gm, function(block){
    const items=block.trim().split('\n').map(l=>l.replace(/^[\-\*] /,''));
    return '<ul>'+items.map(i=>'<li>'+i+'</li>').join('')+'</ul>';
  });
  // 8) ordered list
  s=s.replace(/(^\d+\. .+\n?)+/gm, function(block){
    const items=block.trim().split('\n').map(l=>l.replace(/^\d+\. /,''));
    return '<ol>'+items.map(i=>'<li>'+i+'</li>').join('')+'</ol>';
  });
  // 9) inline: bold, italic, code
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/\*(.+?)\*/g,'<em>$1</em>');
  s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
  // 10) paragraphs: double newline -> <p>
  s=s.split(/\n{2,}/).map(block=>{
    const t=block.trim();
    if(!t) return '';
    if(/^<(pre|div|h[1-4]|ul|ol|table|hr|blockquote)/.test(t)) return t;
    return '<p>'+t.replace(/\n/g,'<br>')+'</p>';
  }).join('\n');
  // single newlines inside remaining text
  // think 블록 복원
  thinkBlocks.forEach((block,i)=>{s=s.replace(`__THINK_${i}__`,block);});
  return s;
}
function addMsg(role,text,rawForDetect){
  const c=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='msg '+role;
  let html = role==='user' ? text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : renderMd(text);
  d.innerHTML=`<div class="msg-label">${role==='user'?'나':'Demos'}</div>${html}`;
  c.appendChild(d);
  d.querySelectorAll('pre').forEach(pre=>{
    const btn=document.createElement('button');
    btn.className='copy-btn';btn.textContent='복사';
    btn.onclick=()=>{navigator.clipboard.writeText(pre.textContent.replace('복사',''));btn.textContent='✓';setTimeout(()=>btn.textContent='복사',1500);};
    pre.appendChild(btn);
  });
  // PPT 코드 블록 감지 → 생성 & 다운로드 버튼 삽입
  if(role==='assistant') detectAndAddPptxButtons(d, (typeof rawForDetect==='string' ? rawForDetect : text));
  // Chart.js 차트 초기화 (data-chart-json 속성이 있는 canvas 탐색)
  d.querySelectorAll('canvas[data-chart-json]').forEach(cv=>{
    try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
    catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
  });
  // 피드백 버튼 (assistant 응답에만)
  if(role==='assistant'){
    const fbDiv=document.createElement('div');
    fbDiv.className='msg-feedback';
    fbDiv.innerHTML='<button onclick="openFeedback(this,\'good\')" title="좋아요">👍</button><button onclick="openFeedback(this,\'bad\')" title="별로예요">👎</button>';
    d.appendChild(fbDiv);
  }
  // 삭제 버튼 주입
  d.style.position='relative';
  d.insertAdjacentHTML('afterbegin','<button class="msg-del" onclick="event.stopPropagation();deleteMsg(this)">✕</button>');
  try{ const _ct=document.querySelector('.content'); if(_ct) _ct.scrollTop=_ct.scrollHeight; }catch(e){}
}
function injectDeleteButtons(){
  // 옛날 버튼 강제 제거
  document.querySelectorAll('.msg-delete-btn').forEach(el=>el.remove());
  document.querySelectorAll('#msgs .msg').forEach(msg=>{
    const existing = msg.querySelectorAll('.msg-del');
    if(existing.length > 0){
      for(let i=1;i<existing.length;i++) existing[i].remove();
      return;
    }
    msg.style.position='relative';
    msg.insertAdjacentHTML('afterbegin','<button class="msg-del" onclick="event.stopPropagation();deleteMsg(this)">✕</button>');
  });
}
function deleteMsg(btn){
  if(!confirm('이 메시지를 삭제할까요?')) return;
  const msgEl = btn.closest('.msg');
  if(!msgEl) return;
  const msgIndex = Array.from(msgEl.parentElement.children).filter(c=>c.classList.contains('msg')).indexOf(msgEl);
  msgEl.remove();
  if(msgIndex >= 0 && msgIndex < history.length){
    history.splice(msgIndex, 1);
  }
  saveCurrentSession();
}
function addTyping(){
  const c=document.getElementById('msgs');
  const d=document.createElement('div');
  d.className='msg assistant';
  d.innerHTML='<div class="typing"><span></span><span></span><span></span></div>';
  c.appendChild(d);
  try{ const _ct=document.querySelector('.content'); if(_ct) _ct.scrollTop=_ct.scrollHeight; }catch(e){}
  return d;
}
// ===== Session Management =====
function loadSessions(){
  try{ sessions = JSON.parse(localStorage.getItem('domos_sessions') || '{}'); }catch(e){ sessions={}; }
}
function saveSessions(){
  try {
    localStorage.setItem('domos_sessions', JSON.stringify(sessions));
  } catch(e){
    console.warn('세션 저장 실패 (localStorage 용량 초과 가능):', e);
    // 오래된 세션 자동 정리 후 재시도
    const ids = Object.keys(sessions).sort((a,b)=>(sessions[a].updatedAt||0)-(sessions[b].updatedAt||0));
    while(ids.length > 20){ const old=ids.shift(); delete sessions[old]; }
    try { localStorage.setItem('domos_sessions', JSON.stringify(sessions)); } catch(e2){}
  }
}
function saveCurrentSession(){
  if(!currentSessionId) return;
  sessions[currentSessionId] = {
    id: currentSessionId,
    name: sessions[currentSessionId]?.name || '새 세션',
    history: history,
    selEnvs: selEnvs,
    selDomains: selDomains,
    selSkills: selSkills,
    selFormat: selFormat,
    formatManualOverride: formatManualOverride,
    styleManualOverride: styleManualOverride,
    effort: effort,
    writingStyle: document.getElementById('writingStyle')?.value || '',
    writingStyleId: activeStyleId || '',
    systemPrompt: document.getElementById('systemPromptInput')?.value || '',
    systemPromptId: currentPromptId || '',
    thinkMode: document.getElementById('thinkToggle')?.checked || false,
    msgsHtml: document.getElementById('msgs').innerHTML,
    updatedAt: Date.now()
  };
  // Auto-name from first user message
  if(sessions[currentSessionId].name === '새 세션' && history.length > 0){
    const first = history.find(m=>m.role==='user');
    if(first) sessions[currentSessionId].name = first.content.slice(0,30) + (first.content.length>30?'...':'');
  }
  saveSessions();
  renderSessionList();
}
let selectedSessions = new Set();
let _lastCheckedSessionId = null; // Shift+클릭용

function renderSessionList(){
  const el=document.getElementById('sessionList');
  if(!el) return;
  const sorted = Object.values(sessions).filter(s=>!s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  const archived = Object.values(sessions).filter(s=>s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));

  document.getElementById('sessionCount').textContent = sorted.length;

  if(sorted.length===0 && archived.length===0){
    el.innerHTML='<div style="font-size:11px;color:#bbb;padding:4px 8px;">저장된 세션 없음</div>';
    return;
  }

  // 날짜별 그룹핑
  const dateGroups = {};
  sorted.forEach(s=>{
    const d = s.updatedAt ? new Date(s.updatedAt) : new Date();
    const today = new Date();
    const yesterday = new Date(today); yesterday.setDate(yesterday.getDate()-1);
    let label;
    if(d.toDateString()===today.toDateString()) label='오늘';
    else if(d.toDateString()===yesterday.toDateString()) label='어제';
    else label=d.toLocaleDateString('ko',{year:'numeric',month:'long',day:'numeric'});
    if(!dateGroups[label]) dateGroups[label]=[];
    dateGroups[label].push(s);
  });

  // 전체 세션 순서 배열 (shift+click 범위 선택용)
  window._sessionOrder = sorted.map(s=>s.id);

  let html = '';
  for(const [dateLabel, items] of Object.entries(dateGroups)){
    html += `<div style="font-size:10px;color:#999;padding:4px 8px 2px;font-weight:600;border-bottom:1px solid #f0f0f0;margin-top:4px;">${dateLabel} (${items.length})</div>`;
    html += items.map(s=>{
      const checked = selectedSessions.has(s.id) ? 'checked' : '';
      const isCurrent = s.id===currentSessionId;
      const time = s.updatedAt ? new Date(s.updatedAt).toLocaleTimeString('ko',{hour:'2-digit',minute:'2-digit'}) : '';
      return `<div class="session-item${isCurrent?' current':''}" onclick="sessionItemClick('${s.id}',event)">
        <input type="checkbox" class="session-cb" ${checked} onclick="event.stopPropagation();toggleSessionCheck('${s.id}',this,event)">
        <span class="session-name">${s.name||'새 세션'}</span>
        <span class="session-time">${time}</span>
      </div>`;
    }).join('');
  }

  // 보관된 세션
  if(archived.length > 0){
    html += `<div style="padding:4px 8px;margin-top:6px;border-top:1px solid #e5e3de;padding-top:8px;">
      <div style="font-size:10px;color:#999;margin-bottom:4px;font-weight:600;cursor:pointer;" onclick="toggleSection('archivedSection','archivedArrow')">
        <span class="arrow" id="archivedArrow">▶</span> 📦 보관함 (${archived.length})
      </div>
      <div id="archivedSection" style="display:none">
      ${archived.map(s=>{
        const checked = selectedSessions.has(s.id) ? 'checked' : '';
        return `<div class="session-item" style="opacity:.7" onclick="sessionItemClick('${s.id}',event)">
          <input type="checkbox" class="session-cb" ${checked} onclick="event.stopPropagation();toggleSessionCheck('${s.id}',this,event)">
          <span class="session-name">${s.name||'세션'}</span>
        </div>`;
      }).join('')}
      </div>
    </div>`;
  }

  el.innerHTML = html;
  updateSessionActions();
}

function toggleSessionCheck(id, cb, event){
  // Shift+클릭: 범위 선택
  if(event && event.shiftKey && _lastCheckedSessionId && window._sessionOrder){
    const order = window._sessionOrder;
    const startIdx = order.indexOf(_lastCheckedSessionId);
    const endIdx = order.indexOf(id);
    if(startIdx !== -1 && endIdx !== -1){
      const from = Math.min(startIdx, endIdx);
      const to = Math.max(startIdx, endIdx);
      for(let i=from; i<=to; i++){
        selectedSessions.add(order[i]);
      }
      renderSessionList();
      return;
    }
  }
  if(cb.checked) selectedSessions.add(id);
  else selectedSessions.delete(id);
  _lastCheckedSessionId = id;
  updateSessionActions();
}

// ===== 세션 팝업 =====
let _popupSelectedSessions = new Set();
let _popupLastChecked = null;

function openSessionPopup(){
  _popupSelectedSessions = new Set(selectedSessions);
  _popupLastChecked = null;
  const overlay = document.getElementById('sessionPopupOverlay');
  overlay.style.display = 'flex';
  renderSessionPopup();
}

function closeSessionPopup(){
  document.getElementById('sessionPopupOverlay').style.display = 'none';
  selectedSessions = new Set(_popupSelectedSessions);
  renderSessionList();
}

function renderSessionPopup(){
  const body = document.getElementById('sessionPopupBody');
  const sorted = Object.values(sessions).filter(s=>!s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  const archived = Object.values(sessions).filter(s=>s.archived).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));

  document.getElementById('popupSessionCount').textContent = `(${sorted.length}개)`;
  window._popupSessionOrder = sorted.map(s=>s.id);

  // 날짜별 그룹핑
  const dateGroups = {};
  sorted.forEach(s=>{
    const d = s.updatedAt ? new Date(s.updatedAt) : new Date();
    const today = new Date();
    const yesterday = new Date(today); yesterday.setDate(yesterday.getDate()-1);
    let label;
    if(d.toDateString()===today.toDateString()) label='오늘';
    else if(d.toDateString()===yesterday.toDateString()) label='어제';
    else label=d.toLocaleDateString('ko',{year:'numeric',month:'long',day:'numeric'});
    if(!dateGroups[label]) dateGroups[label]=[];
    dateGroups[label].push(s);
  });

  let html = '';
  for(const [dateLabel, items] of Object.entries(dateGroups)){
    html += `<div style="font-size:12px;color:#6b7280;padding:8px 4px 4px;font-weight:700;border-bottom:2px solid #e5e7eb;margin-top:8px;">${dateLabel} <span style="color:#9ca3af;font-weight:400">(${items.length}개)</span></div>`;
    html += items.map(s=>{
      const checked = _popupSelectedSessions.has(s.id) ? 'checked' : '';
      const isCurrent = s.id===currentSessionId;
      const time = s.updatedAt ? new Date(s.updatedAt).toLocaleTimeString('ko',{hour:'2-digit',minute:'2-digit'}) : '';
      const msgCount = s.history ? s.history.length : 0;
      return `<div style="display:flex;align-items:center;gap:8px;padding:8px 4px;border-bottom:1px solid #f3f4f6;cursor:pointer;${isCurrent?'background:#eef2ff;border-radius:6px;':''}" onclick="popupSessionClick('${s.id}',event)">
        <input type="checkbox" ${checked} onclick="event.stopPropagation();popupToggleCheck('${s.id}',this,event)" style="width:16px;height:16px;cursor:pointer;">
        <div style="flex:1;min-width:0;">
          <div style="font-size:13px;font-weight:${isCurrent?'700':'500'};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;${isCurrent?'color:#4f46e5;':'color:#1f2937;'}">${s.name||'새 세션'}${isCurrent?' (현재)':''}</div>
          <div style="font-size:11px;color:#9ca3af;">${msgCount}개 메시지</div>
        </div>
        <div style="font-size:11px;color:#9ca3af;white-space:nowrap;">${time}</div>
      </div>`;
    }).join('');
  }

  if(archived.length > 0){
    html += `<div style="margin-top:12px;border-top:2px solid #e5e7eb;padding-top:8px;">
      <div style="font-size:12px;color:#9ca3af;font-weight:700;padding:4px;cursor:pointer;" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'':'none';this.querySelector('span').textContent=this.nextElementSibling.style.display==='none'?'▶':'▼'">
        <span>▶</span> 📦 보관함 (${archived.length}개)
      </div>
      <div style="display:none">
      ${archived.map(s=>{
        const checked = _popupSelectedSessions.has(s.id) ? 'checked' : '';
        return `<div style="display:flex;align-items:center;gap:8px;padding:6px 4px;opacity:.6;border-bottom:1px solid #f3f4f6;">
          <input type="checkbox" ${checked} onclick="event.stopPropagation();popupToggleCheck('${s.id}',this,event)" style="width:16px;height:16px;">
          <span style="font-size:12px;flex:1;">${s.name||'세션'}</span>
        </div>`;
      }).join('')}
      </div>
    </div>`;
  }

  if(sorted.length===0 && archived.length===0){
    html = '<div style="text-align:center;color:#9ca3af;padding:40px;">저장된 세션이 없습니다.</div>';
  }

  body.innerHTML = html;
}

function popupSessionClick(id, event){
  if(event.shiftKey && _popupLastChecked && window._popupSessionOrder){
    event.preventDefault();
    const order = window._popupSessionOrder;
    const startIdx = order.indexOf(_popupLastChecked);
    const endIdx = order.indexOf(id);
    if(startIdx !== -1 && endIdx !== -1){
      const from = Math.min(startIdx, endIdx);
      const to = Math.max(startIdx, endIdx);
      for(let i=from; i<=to; i++) _popupSelectedSessions.add(order[i]);
      _popupLastChecked = id;
      renderSessionPopup();
      return;
    }
  }
  // 일반 클릭: 세션 로드 후 팝업 닫기
  selectedSessions = new Set(_popupSelectedSessions);
  closeSessionPopup();
  loadSession(id);
}

function popupToggleCheck(id, cb, event){
  if(event && event.shiftKey && _popupLastChecked && window._popupSessionOrder){
    const order = window._popupSessionOrder;
    const startIdx = order.indexOf(_popupLastChecked);
    const endIdx = order.indexOf(id);
    if(startIdx !== -1 && endIdx !== -1){
      const from = Math.min(startIdx, endIdx);
      const to = Math.max(startIdx, endIdx);
      for(let i=from; i<=to; i++) _popupSelectedSessions.add(order[i]);
      renderSessionPopup();
      return;
    }
  }
  if(cb.checked) _popupSelectedSessions.add(id);
  else _popupSelectedSessions.delete(id);
  _popupLastChecked = id;
}

function selectAllSessionsPopup(){
  const all = Object.values(sessions).filter(s=>!s.archived);
  if(_popupSelectedSessions.size === all.length) _popupSelectedSessions.clear();
  else all.forEach(s=> _popupSelectedSessions.add(s.id));
  renderSessionPopup();
}

function deleteSelectedSessionsPopup(){
  if(_popupSelectedSessions.size===0) return;
  if(!confirm(`${_popupSelectedSessions.size}개 세션을 삭제할까요?`)) return;
  let deletedCurrent = false;
  _popupSelectedSessions.forEach(id=>{
    if(id===currentSessionId) deletedCurrent = true;
    delete sessions[id];
  });
  _popupSelectedSessions.clear();
  saveSessions();
  if(deletedCurrent){ currentSessionId = null; createNewSession(); }
  renderSessionPopup();
  renderSessionList();
}

function searchSessionsPopup(query){
  const body = document.getElementById('sessionPopupBody');
  const q = query.trim().toLowerCase();
  if(!q){ renderSessionPopup(); return; }

  const all = Object.values(sessions).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  const results = [];

  all.forEach(s=>{
    const name = (s.name||'').toLowerCase();
    let matchedMsgs = [];

    // 제목 검색
    const nameMatch = name.includes(q);

    // 대화 내용 검색
    if(s.history){
      s.history.forEach((m, idx)=>{
        const content = typeof m.content === 'string' ? m.content : '';
        if(content.toLowerCase().includes(q)){
          // 매칭된 부분 전후 50자 미리보기
          const pos = content.toLowerCase().indexOf(q);
          const start = Math.max(0, pos - 30);
          const end = Math.min(content.length, pos + q.length + 30);
          let preview = (start > 0 ? '...' : '') + content.slice(start, end) + (end < content.length ? '...' : '');
          // 매칭 부분 하이라이트
          preview = preview.replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'gi'), '<mark style="background:#fef08a;padding:0 1px;border-radius:2px;">$&</mark>');
          matchedMsgs.push({ role: m.role, preview, idx });
        }
      });
    }

    if(nameMatch || matchedMsgs.length > 0){
      results.push({ session: s, nameMatch, matchedMsgs });
    }
  });

  if(results.length === 0){
    body.innerHTML = `<div style="text-align:center;color:#9ca3af;padding:40px;">검색 결과 없음: "${query}"</div>`;
    document.getElementById('popupSessionCount').textContent = `(0건)`;
    return;
  }

  document.getElementById('popupSessionCount').textContent = `(${results.length}건 검색됨)`;

  let html = results.map(r=>{
    const s = r.session;
    const isCurrent = s.id === currentSessionId;
    const time = s.updatedAt ? new Date(s.updatedAt).toLocaleString('ko',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
    const archived = s.archived ? ' 📦' : '';

    let nameHtml = s.name || '새 세션';
    if(r.nameMatch){
      nameHtml = nameHtml.replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'gi'), '<mark style="background:#fef08a;padding:0 1px;border-radius:2px;">$&</mark>');
    }

    let msgHtml = '';
    if(r.matchedMsgs.length > 0){
      msgHtml = r.matchedMsgs.slice(0, 3).map(m=>{
        const roleLabel = m.role === 'user' ? '사용자' : 'AI';
        return `<div style="font-size:11px;color:#6b7280;padding:2px 0;"><span style="color:#4f46e5;font-weight:600;">[${roleLabel}]</span> ${m.preview}</div>`;
      }).join('');
      if(r.matchedMsgs.length > 3){
        msgHtml += `<div style="font-size:10px;color:#9ca3af;">...외 ${r.matchedMsgs.length - 3}건 매칭</div>`;
      }
    }

    return `<div style="padding:10px 8px;border-bottom:1px solid #f3f4f6;cursor:pointer;${isCurrent?'background:#eef2ff;border-radius:6px;':''}" onclick="closeSessionPopup();loadSession('${s.id}')">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:13px;font-weight:600;${isCurrent?'color:#4f46e5;':''}">${nameHtml}${archived}</div>
        <div style="font-size:11px;color:#9ca3af;">${time}</div>
      </div>
      ${msgHtml}
    </div>`;
  }).join('');

  body.innerHTML = html;
}

function exportSessionsToMd(){
  // 선택된 세션이 있으면 선택된 것만, 없으면 전체
  const targets = _popupSelectedSessions.size > 0
    ? Object.values(sessions).filter(s=> _popupSelectedSessions.has(s.id))
    : Object.values(sessions).filter(s=>!s.archived);
  if(targets.length===0){ alert('저장할 세션이 없습니다.'); return; }

  targets.sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));

  let md = `# 세션 대화 기록\n\n`;
  md += `> 내보내기 날짜: ${new Date().toLocaleString('ko')}\n`;
  md += `> 세션 수: ${targets.length}개\n\n---\n\n`;

  targets.forEach((s, idx)=>{
    const date = s.updatedAt ? new Date(s.updatedAt).toLocaleString('ko') : '날짜 없음';
    const msgCount = s.history ? s.history.length : 0;
    md += `## ${idx+1}. ${s.name||'새 세션'}\n\n`;
    md += `- 날짜: ${date}\n`;
    md += `- 메시지: ${msgCount}개\n\n`;

    if(s.history && s.history.length > 0){
      s.history.forEach(m=>{
        const role = m.role === 'user' ? '**사용자**' : '**AI**';
        const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
        md += `### ${role}\n\n${content}\n\n`;
      });
    } else {
      md += `_(대화 내용 없음)_\n\n`;
    }
    md += `---\n\n`;
  });

  // 다운로드
  const blob = new Blob([md], {type:'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const dateStr = new Date().toISOString().slice(0,10);
  a.href = url;
  a.download = `sessions_${dateStr}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

function deleteAllSessionsPopup(){
  const all = Object.values(sessions).filter(s=>!s.archived);
  if(all.length===0) return;
  if(!confirm(`전체 ${all.length}개 세션을 삭제할까요?`)) return;
  all.forEach(s=> delete sessions[s.id]);
  _popupSelectedSessions.clear();
  saveSessions();
  currentSessionId = null;
  createNewSession();
  renderSessionPopup();
  renderSessionList();
}

function sessionItemClick(id, event){
  // Shift+클릭: 범위 선택
  if(event.shiftKey && _lastCheckedSessionId && window._sessionOrder){
    event.preventDefault();
    const order = window._sessionOrder;
    const startIdx = order.indexOf(_lastCheckedSessionId);
    const endIdx = order.indexOf(id);
    if(startIdx !== -1 && endIdx !== -1){
      const from = Math.min(startIdx, endIdx);
      const to = Math.max(startIdx, endIdx);
      for(let i=from; i<=to; i++){
        selectedSessions.add(order[i]);
      }
      _lastCheckedSessionId = id;
      renderSessionList();
      return;
    }
  }
  // 일반 클릭: 세션 로드
  loadSession(id);
}

function selectAllSessions(){
  const all = Object.values(sessions).filter(s=>!s.archived);
  if(selectedSessions.size === all.length){
    // 전체 해제
    selectedSessions.clear();
  } else {
    // 전체 선택
    all.forEach(s=> selectedSessions.add(s.id));
  }
  renderSessionList();
}

function deleteAllSessions(){
  const all = Object.values(sessions).filter(s=>!s.archived);
  if(all.length===0) return;
  if(!confirm(`전체 ${all.length}개 세션을 삭제할까요?`)) return;
  all.forEach(s=> delete sessions[s.id]);
  selectedSessions.clear();
  saveSessions();
  currentSessionId = null;
  createNewSession();
}

function updateSessionActions(){
  const el = document.getElementById('sessionActions');
  if(selectedSessions.size === 0){ el.style.display='none'; return; }
  // 체크된 것 중 보관/일반 구분
  let hasArchived=false, hasNormal=false;
  selectedSessions.forEach(id=>{
    if(sessions[id]?.archived) hasArchived=true;
    else hasNormal=true;
  });
  let btns = '';
  if(hasNormal) btns += `<button onclick="archiveSelectedSessions()">📦 보관</button>`;
  if(hasArchived) btns += `<button onclick="restoreSelectedSessions()">↩ 복원</button>`;
  btns += `<button class="danger" onclick="deleteSelectedSessions()">🗑️ 삭제</button>`;
  btns += `<button onclick="clearSessionSelection()">취소</button>`;
  el.innerHTML = btns;
  el.style.display = 'flex';
}

function clearSessionSelection(){
  selectedSessions.clear();
  renderSessionList();
}

function deleteSelectedSessions(){
  if(selectedSessions.size===0) return;
  const count = selectedSessions.size;
  if(!confirm(count + '개 세션을 삭제할까요?')) return;
  let deletedCurrent = false;
  selectedSessions.forEach(id=>{
    if(id===currentSessionId) deletedCurrent = true;
    delete sessions[id];
  });
  selectedSessions.clear();
  saveSessions();
  if(deletedCurrent){
    // createNewSession() 호출 시 saveCurrentSession()이 삭제된 세션을 복원하므로
    // currentSessionId를 먼저 초기화한 후 새 세션 생성
    currentSessionId = null;
    createNewSession();
  } else renderSessionList();
}

function archiveSelectedSessions(){
  if(selectedSessions.size===0) return;
  let archivedCurrent = false;
  selectedSessions.forEach(id=>{
    if(sessions[id]){
      sessions[id].archived = true;
      if(id===currentSessionId) archivedCurrent = true;
    }
  });
  selectedSessions.clear();
  saveSessions();
  if(archivedCurrent){
    currentSessionId = null;
    createNewSession();
  } else renderSessionList();
}

// ===== 사이드바 전체 접기/펼치기 =====
function toggleSidebar(){
  const sb = document.getElementById('sidebar');
  const btn = document.getElementById('sidebarToggle');
  const chatBox = document.querySelector('.chat-box-fixed');
  sb.classList.toggle('collapsed');
  document.body.classList.toggle('sb-collapsed');
  if(sb.classList.contains('collapsed')){
    btn.textContent = '▶';
    btn.title = '사이드바 펼치기';
    if(chatBox) chatBox.style.left = '48px';
  } else {
    btn.textContent = '◀';
    btn.title = '사이드바 접기';
    if(chatBox) chatBox.style.left = '250px';
  }
  // localStorage에 상태 저장
  localStorage.setItem('domos_sidebar_collapsed', sb.classList.contains('collapsed'));
}
// 저장된 상태 복원
(function(){
  const saved = localStorage.getItem('domos_sidebar_collapsed');
  if(saved === 'true'){
    const sb = document.getElementById('sidebar');
    const btn = document.getElementById('sidebarToggle');
    const chatBox = document.querySelector('.chat-box-fixed');
    if(sb){ sb.classList.add('collapsed'); }
    if(btn){ btn.textContent='▶'; btn.title='사이드바 펼치기'; }
    if(chatBox){ chatBox.style.left='48px'; }
    document.body.classList.add('sb-collapsed');
  }
})();
function createNewSession(){
  // Save current before switching
  saveCurrentSession();
  currentSessionId = 'sess_'+Date.now();
  history=[];
  selSkills=[];
  autoLoadedSkills=[];
  dismissedAutoSkills.clear();
  document.getElementById('msgs').innerHTML='';
  document.getElementById('writingStyle').value='';
  document.getElementById('systemPromptInput').value='';
  activeStyleId=null;
  currentPromptId=null;
  selFormat='auto';
  formatManualOverride=false;
  styleManualOverride=false;
  effort=2;
  document.getElementById('effortSlider').value=2;
  document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f==='auto');});
  // 업로드된 파일/CSV 초기화 (이전 세션 데이터 잔류 방지)
  fetch('/api/clear_files', {method:'POST'});
  fetch('/api/clear_csv', {method:'POST'});
  uploadedFilesList = [];
  renderFileList();
  csvLoaded = false;
  csvFilename = '';
  document.getElementById('csvInfoPanel').style.display = 'none';
  chatPendingFiles = [];
  const chatFileWrap = document.getElementById('chatAttachPreview');
  if(chatFileWrap){ chatFileWrap.innerHTML = ''; chatFileWrap.style.display='none'; }
  renderSkills();
  updateLoaded();
  renderStyleChips();
  renderPromptChips();
  updateSpBadge();
  updateEffort();
  sessions[currentSessionId] = { id:currentSessionId, name:'새 세션', history:[], msgsHtml:'', updatedAt:Date.now() };
  saveSessions();
  renderSessionList();
}
function loadSession(id){
  if(id===currentSessionId) return;
  saveCurrentSession();
  const s=sessions[id];
  if(!s) return;
  currentSessionId=id;
  history=s.history||[];
  document.getElementById('msgs').innerHTML=s.msgsHtml||'';
  injectDeleteButtons();
  if(s.writingStyle){ document.getElementById('writingStyle').value=s.writingStyle; syncStyleDropToSidebar(); }
  if(s.writingStyleId){ activeStyleId=s.writingStyleId; renderStyleChips(); }
  styleManualOverride = s.styleManualOverride !== undefined ? s.styleManualOverride : !!s.writingStyle;
  if(s.systemPrompt) document.getElementById('systemPromptInput').value=s.systemPrompt;
  if(s.thinkMode!==undefined) document.getElementById('thinkToggle').checked=s.thinkMode;
  if(s.selFormat){ selFormat=s.selFormat; formatManualOverride=(s.formatManualOverride!==undefined ? s.formatManualOverride : selFormat!=='auto'); document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===selFormat);}); syncFormatDropToChat(); }
  if(s.effort!==undefined){ effort=s.effort; document.getElementById('effortSlider').value=effort; updateEffort(); }
  // 저장된 수동 스킬 복원
  if(s.selSkills && s.selSkills.length > 0){ selSkills = [...s.selSkills]; }
  else { selSkills = []; }
  // 대화 히스토리 기반 자동 스킬 프리로드
  autoLoadedSkills = [];
  dismissedAutoSkills.clear();
  if(autoSkillMode && history.length > 0){
    const lastUser = [...history].reverse().find(m=>m.role==='user');
    if(lastUser){
      fetch('/api/auto-skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:lastUser.content,max_skills:5,history:history})})
      .then(r=>r.json()).then(d=>{
        if(d.skills && d.skills.length > 0){
          const manualSet = new Set(selSkills);
          autoLoadedSkills = d.skills.map(s=>s.id).filter(id=>!manualSet.has(id));
          renderSkills();
          updateLoaded();
        }
      }).catch(()=>{});
    }
  }
  renderSkills();
  updateLoaded();
  renderSessionList();
  // 세션 복원 후 차트 재렌더링
  document.querySelectorAll('#msgs canvas[data-chart-json]').forEach(cv=>{
    try{renderChartBlock(cv.id, b2u(cv.dataset.chartJson));}
    catch(e){cv.parentElement.innerHTML='<p style="color:red;padding:12px;">차트 렌더링 실패: '+e.message+'</p>';}
  });
}
function deleteSession(id){
  delete sessions[id];
  saveSessions();
  if(id===currentSessionId) createNewSession();
  else renderSessionList();
}
function restoreArchivedSession(id){
  if(sessions[id]){ sessions[id].archived=false; saveSessions(); renderSessionList(); }
}
function restoreSelectedSessions(){
  if(selectedSessions.size===0) return;
  selectedSessions.forEach(id=>{
    if(sessions[id]) sessions[id].archived=false;
  });
  selectedSessions.clear();
  saveSessions();
  renderSessionList();
}
function newSession(){ createNewSession(); }

// ===== Section Toggle (분야/스킬 접기/펼치기) =====
function toggleSection(sectionId, arrowId){
  const section = document.getElementById(sectionId);
  const arrow = document.getElementById(arrowId);
  if(section.style.display === 'none'){
    section.style.display = '';
    arrow.textContent = '▼';
  } else {
    section.style.display = 'none';
    arrow.textContent = '▶';
  }
}

// ===== System Prompt =====
function toggleSysPrompt(){
  const body = document.getElementById('spBody');
  const arrow = document.getElementById('spArrow');
  body.classList.toggle('show');
  arrow.classList.toggle('open');
  if(body.classList.contains('show')){
    arrow.textContent='▼';
  } else {
    arrow.textContent='▶';
    updateSpBadge();
  }
}
function updateSpBadge(){
  const badge = document.getElementById('spPreviewBadge');
  const val = document.getElementById('systemPromptInput').value.trim();
  if(val){
    badge.textContent = val.length + '자 커스텀';
    badge.style.background='#fef3c7';
    badge.style.color='#d97706';
  } else {
    badge.textContent = '기본값';
    badge.style.background='#eef2ff';
    badge.style.color='#6366f1';
  }
}
function resetSysPrompt(){
  document.getElementById('systemPromptInput').value='';
  currentPromptId = null;
  updateSpBadge();
  renderPromptChips();
}
document.getElementById('systemPromptInput').addEventListener('input', function(){
  updateSpBadge();
  // 수동 편집하면 active 해제
  if(currentPromptId){
    currentPromptId = null;
    renderPromptChips();
  }
});

// ===== 시스템 프롬프트 프리셋/저장 관리 =====
let currentPromptId = null;
let promptList = [];

async function loadPromptList(){
  try {
    const r = await fetch('/api/prompts');
    const d = await r.json();
    promptList = d.prompts || [];
    renderPromptChips();
  } catch(e){ console.error('프롬프트 목록 로드 실패:', e); }
}

function renderPromptChips(){
  const container = document.getElementById('promptPresets');
  if(!container) return;
  container.innerHTML = '';
  promptList.forEach(p => {
    const chip = document.createElement('span');
    chip.className = 'prompt-chip' + (p.type==='saved'?' saved':'') + (currentPromptId===p.id?' active':'');
    chip.innerHTML = p.icon + ' ' + p.name;
    chip.title = p.preview;
    chip.onclick = () => selectPrompt(p.id);
    container.appendChild(chip);
  });
  // 삭제 버튼 표시/숨김
  const delBtn = document.getElementById('btnDeletePrompt');
  if(delBtn) delBtn.style.display = (currentPromptId && currentPromptId.startsWith('saved:')) ? '' : 'none';
}

async function selectPrompt(pid){
  try {
    const r = await fetch('/api/prompts/' + encodeURIComponent(pid));
    const d = await r.json();
    if(d.content){
      document.getElementById('systemPromptInput').value = d.content;
      currentPromptId = pid;
      updateSpBadge();
      renderPromptChips();
    }
  } catch(e){ console.error('프롬프트 로드 실패:', e); }
}

async function saveCurrentPrompt(){
  const content = document.getElementById('systemPromptInput').value.trim();
  if(!content){ alert('저장할 내용이 없습니다.'); return; }
  const name = prompt('프롬프트 이름을 입력하세요:', currentPromptId?.startsWith('saved:') ? currentPromptId.slice(6) : '');
  if(!name) return;
  try {
    const r = await fetch('/api/prompts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, content})
    });
    const d = await r.json();
    if(d.success){
      currentPromptId = d.id;
      const status = document.getElementById('promptSaveStatus');
      status.style.display='inline'; setTimeout(()=>status.style.display='none', 2000);
      await loadPromptList();
    } else { alert(d.error || '저장 실패'); }
  } catch(e){ alert('저장 오류: ' + e); }
}

async function deleteCurrentPrompt(){
  if(!currentPromptId || !currentPromptId.startsWith('saved:')) return;
  const name = currentPromptId.slice(6);
  if(!confirm('"' + name + '" 프롬프트를 삭제하시겠습니까?')) return;
  try {
    const r = await fetch('/api/prompts/' + encodeURIComponent(currentPromptId), {method:'DELETE'});
    const d = await r.json();
    if(d.success){
      document.getElementById('systemPromptInput').value = '';
      currentPromptId = null;
      updateSpBadge();
      await loadPromptList();
    }
  } catch(e){ alert('삭제 오류: ' + e); }
}

// 페이지 로드시 프롬프트 목록 불러오기
loadPromptList();

// ===== 작성 스타일 프리셋 =====
const STYLE_PRESETS = [
  {id:'concise',    icon:'⚡', name:'간결', value:'간결하고 핵심만. 불필요한 설명 생략.'},
  {id:'detailed',   icon:'📖', name:'상세', value:'상세하고 친절하게. 원리, 배경, 예시 포함.'},
  {id:'practical',  icon:'🔨', name:'실용적', value:'실전에서 바로 활용 가능하게. 핵심 요점과 구체적 방법 위주. 선택된 출력형식에 맞춰 답변하세요.'},
  {id:'academic',   icon:'🎓', name:'학술', value:'학술적 톤. 정확한 용어, 인용, 근거 제시.'},
  {id:'comment-kr', icon:'💬', name:'한글주석', value:'코드 주석을 상세한 한국어로. 변수명은 영어 유지.'},
  {id:'senior-dev', icon:'👨‍💻', name:'시니어', value:'시니어 개발자 관점. 왜 이렇게 하는지, 대안은 뭔지, 주의점 포함.'},
  {id:'debug',      icon:'🐛', name:'디버그', value:'에러 원인 분석 중심. traceback 해석, 재현 조건, 해결책 순서.'},
  {id:'data-story', icon:'📊', name:'데이터', value:'데이터 스토리텔링. 숫자→의미→액션 순서로 해석.'},
];
let activeStyleId = null;

function renderStyleChips(){
  const container = document.getElementById('styleChips');
  if(!container) return;
  container.innerHTML = '';
  // 🤖 자동 칩 (맨 앞)
  const autoChip = document.createElement('span');
  autoChip.className = 'prompt-chip' + (!styleManualOverride ? ' active' : '');
  autoChip.innerHTML = '🤖 자동';
  autoChip.title = '채팅 내용에 따라 자동 선택';
  autoChip.onclick = () => {
    activeStyleId = null;
    styleManualOverride = false;
    document.getElementById('writingStyle').value = '';
    renderStyleChips();
  };
  container.appendChild(autoChip);
  STYLE_PRESETS.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'prompt-chip' + (activeStyleId===s.id ? ' active' : '');
    chip.innerHTML = s.icon + ' ' + s.name;
    chip.title = s.value;
    chip.onclick = () => {
      if(activeStyleId === s.id){
        activeStyleId = null;
        styleManualOverride = false;
        document.getElementById('writingStyle').value = '';
      } else {
        activeStyleId = s.id;
        styleManualOverride = true;
        document.getElementById('writingStyle').value = s.value;
      }
      renderStyleChips();
    };
    container.appendChild(chip);
  });
}
renderStyleChips();

// ===== 채팅창 드롭다운 ↔ 사이드바 동기화 =====
function applyChatStyle(val){
  document.getElementById('writingStyle').value = val;
  const match = STYLE_PRESETS.find(s => s.value === val);
  activeStyleId = match ? match.id : null;
  styleManualOverride = !!val;
  renderStyleChips();
  if(val) syncStyleDropToSidebar();
}
function applyChatFormat(val){
  selFormat = val;
  formatManualOverride = (val !== 'auto');
  // 사이드바 fmt-btn도 동기화
  document.querySelectorAll('.fmt-btn').forEach(b=>{b.classList.toggle('selected',b.dataset.f===val);});
}
function syncStyleDropToSidebar(){
  // 사이드바 변경 → 채팅창 드롭다운 동기화
  const drop = document.getElementById('chatStyleDrop');
  if(!drop) return;
  const val = document.getElementById('writingStyle').value.trim();
  const opts = drop.options;
  let found = false;
  for(let i=0;i<opts.length;i++){
    if(opts[i].value === val){ drop.selectedIndex = i; found = true; break; }
  }
  if(!found) drop.selectedIndex = 0;
}
function syncFormatDropToChat(){
  const drop = document.getElementById('chatFormatDrop');
  if(!drop) return;
  const opts = drop.options;
  for(let i=0;i<opts.length;i++){
    if(opts[i].value === selFormat){ drop.selectedIndex = i; break; }
  }
}

document.getElementById('writingStyle').addEventListener('input', function(){
  // 수동 편집하면 active 해제
  const val = this.value.trim();
  const match = STYLE_PRESETS.find(s => s.value === val);
  activeStyleId = match ? match.id : null;
  renderStyleChips();
});

// ===== Skill Detail Panel =====
let currentSkillDetail = null;

async function openSkillDetail(skillId){
  const overlay = document.getElementById('skillDetailOverlay');
  overlay.style.display = 'flex';
  document.getElementById('sdpTitle').textContent = skillId;
  document.getElementById('sdpBody').innerHTML = '⏳ 로딩중...';

  try{
    const resp = await fetch(`/api/skill/${skillId}`);
    currentSkillDetail = await resp.json();
    renderSdpTabs('scripts');
  }catch(e){
    document.getElementById('sdpBody').innerHTML = `❌ 로딩 실패: ${e.message}`;
  }
}

function closeSkillDetail(){
  document.getElementById('skillDetailOverlay').style.display = 'none';
  currentSkillDetail = null;
}

function renderSdpTabs(active){
  if(!currentSkillDetail) return;
  const d = currentSkillDetail;
  const tabsEl = document.getElementById('sdpTabs');
  const tabs = [
    {id:'scripts', label:`🐍 스크립트 (${(d.scripts||[]).length})`, show: true},
    {id:'references', label:`📄 레퍼런스 (${(d.references||[]).length})`, show: true},
    {id:'assets', label:`📦 에셋 (${(d.assets||[]).length})`, show: (d.assets||[]).length > 0},
    {id:'skillmd', label:'📋 SKILL.md', show: true},
  ];
  tabsEl.innerHTML = '';
  tabs.forEach(t=>{
    if(!t.show) return;
    const el = document.createElement('div');
    el.className = 'sdp-tab' + (active===t.id?' active':'');
    el.textContent = t.label;
    el.onclick = ()=> renderSdpTabs(t.id);
    tabsEl.appendChild(el);
  });
  renderSdpContent(active);
}

function renderSdpContent(tab){
  const body = document.getElementById('sdpBody');
  const d = currentSkillDetail;

  if(tab === 'skillmd'){
    body.innerHTML = `<div class="sdp-code">${esc(d.content || '(내용 없음)')}</div>
      <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;">
        <button class="sdp-btn inject" onclick="injectToChat('skillmd','')">💬 채팅에 주입</button>
        <button class="sdp-btn" style="background:#16a34a;color:#fff;" onclick="downloadExistingSkill('${d.name}')">📥 다운로드</button>
        <button class="sdp-btn" style="background:#f59e0b;color:#fff;" onclick="editSkillMd('${d.name}')">✏️ 편집</button>
      </div>`;
    return;
  }

  const files = d[tab] || [];
  if(files.length === 0){
    body.innerHTML = '<div style="text-align:center;color:#999;padding:40px">파일 없음</div>';
    return;
  }

  const isPy = tab === 'scripts';
  let html = '<ul class="sdp-file-list">';
  files.forEach(f=>{
    const icon = f.name.endsWith('.py') ? '🐍' : f.name.endsWith('.md') ? '📄' : '📎';
    const sizeKb = (f.size / 1024).toFixed(1);
    html += `<li class="sdp-file-item" onclick="viewSkillFile('${d.name}','${f.path}')">
      <span class="sdp-file-icon">${icon}</span>
      <span class="sdp-file-name">${esc(f.name)}</span>
      <span class="sdp-file-size">${sizeKb} KB</span>
      <div class="sdp-file-actions">
        <button class="sdp-btn" onclick="event.stopPropagation();viewSkillFile('${d.name}','${f.path}')">👁️ 보기</button>
        <button class="sdp-btn inject" onclick="event.stopPropagation();injectToChat('${tab}','${f.path}')">💬 주입</button>
        ${isPy && f.name.endsWith('.py') ? `<button class="sdp-btn run" onclick="event.stopPropagation();runSkillScript('${d.name}','${f.path}')">▶ 실행</button>` : ''}
      </div>
    </li>`;
  });
  html += '</ul><div id="sdpFileContent"></div>';
  body.innerHTML = html;
}

async function viewSkillFile(skillId, filePath){
  const container = document.getElementById('sdpFileContent');
  if(!container) return;
  container.innerHTML = '⏳ 로딩중...';
  try{
    const resp = await fetch(`/api/skill/${skillId}/file?path=${encodeURIComponent(filePath)}`);
    const data = await resp.json();
    if(data.error){
      container.innerHTML = `<div class="sdp-result err">${esc(data.error)}</div>`;
    } else {
      container.innerHTML = `<div style="font-size:12px;font-weight:700;margin:10px 0 6px">${esc(data.name)} (${(data.size/1024).toFixed(1)} KB)</div>
        <div class="sdp-code">${esc(data.content)}</div>`;
    }
  }catch(e){
    container.innerHTML = `<div class="sdp-result err">오류: ${e.message}</div>`;
  }
}

async function runSkillScript(skillId, scriptPath){
  const container = document.getElementById('sdpFileContent');
  if(!container) return;
  container.innerHTML = '⏳ 스크립트 실행 중... (최대 60초)';
  try{
    const resp = await fetch(`/api/skill/${skillId}/run`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({script: scriptPath})
    });
    const data = await resp.json();
    if(data.error){
      container.innerHTML = `<div class="sdp-result err">❌ ${esc(data.error)}</div>`;
    } else {
      const cls = data.returncode === 0 ? 'ok' : 'err';
      let output = '';
      if(data.stdout) output += `=== stdout ===\n${data.stdout}\n`;
      if(data.stderr) output += `=== stderr ===\n${data.stderr}\n`;
      output += `\n[종료 코드: ${data.returncode}]`;
      container.innerHTML = `<div style="font-size:12px;font-weight:700;margin:10px 0 6px">▶ ${esc(scriptPath)} 실행 결과</div>
        <div class="sdp-result ${cls}">${esc(output)}</div>`;
    }
  }catch(e){
    container.innerHTML = `<div class="sdp-result err">실행 오류: ${e.message}</div>`;
  }
}

async function injectToChat(type, filePath){
  // 파일 내용을 채팅 입력에 주입
  const input = document.getElementById('input');
  const skillId = currentSkillDetail ? currentSkillDetail.name : '';

  if(type === 'skillmd'){
    input.value += `[${skillId} SKILL.md 참조 중]\n`;
    closeSkillDetail();
    return;
  }

  try{
    const resp = await fetch(`/api/skill/${skillId}/file?path=${encodeURIComponent(filePath)}`);
    const data = await resp.json();
    if(data.content){
      const preview = data.content.length > 500 ? data.content.substring(0,500) + '...' : data.content;
      input.value += `\n--- ${data.name} ---\n${preview}\n---\n`;
    }
  }catch(e){ console.warn('파일 주입 실패:', e); }
  closeSkillDetail();
  input.focus();
}

// ===== 스킬 생성 모달 JS =====
function openSkillCreate(){
  document.getElementById('skillCreateOverlay').style.display = 'flex';
  switchScTab('manual');
}
function closeSkillCreate(){
  document.getElementById('skillCreateOverlay').style.display = 'none';
}
function switchScTab(tab){
  document.querySelectorAll('#scTabs .sdp-tab').forEach((el,i)=>{
    el.className = 'sdp-tab' + (i === (tab==='manual'?0:1) ? ' active' : '');
  });
  document.getElementById('scManual').style.display = tab==='manual' ? 'block' : 'none';
  document.getElementById('scLlm').style.display = tab==='llm' ? 'block' : 'none';
}
function validateScName(){
  const name = document.getElementById('scName').value.trim();
  const el = document.getElementById('scNameStatus');
  if(!name){ el.innerHTML=''; return; }
  if(/^[a-z0-9]+(-[a-z0-9]+)*$/.test(name) && name.length<=64){
    el.innerHTML='<span style="color:#16a34a">✅ 유효한 이름</span>';
  } else {
    el.innerHTML='<span style="color:#ef4444">❌ 소문자+숫자+하이픈만, 64자 이내</span>';
  }
}
function _renderValidation(containerId, data){
  const el = document.getElementById(containerId);
  let html = '';
  if(data.errors && data.errors.length > 0){
    html += data.errors.map(e=>`<div style="color:#ef4444;font-size:12px;margin:2px 0;">❌ ${e}</div>`).join('');
  }
  if(data.warnings && data.warnings.length > 0){
    html += data.warnings.map(w=>`<div style="color:#f59e0b;font-size:12px;margin:2px 0;">⚠️ ${w}</div>`).join('');
  }
  if(data.valid){
    html += '<div style="color:#16a34a;font-size:12px;margin:2px 0;">✅ 검증 통과! 다운로드 가능합니다.</div>';
  }
  el.innerHTML = html;
  return data.valid;
}
async function validateSkillPreview(){
  const name = document.getElementById('scName').value.trim();
  const desc = document.getElementById('scDesc').value.trim();
  const body = document.getElementById('scBody_manual').value.trim();
  const content = `---\nname: ${name}\ndescription: >\n  ${desc}\n---\n\n${body}`;
  try{
    const resp = await fetch('/api/skill/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    const data = await resp.json();
    const valid = _renderValidation('scValidateResult', data);
  }catch(e){
    document.getElementById('scValidateResult').innerHTML = `<div style="color:#ef4444;font-size:12px;">오류: ${e.message}</div>`;
  }
}
async function downloadSkill(){
  const name = document.getElementById('scName').value.trim();
  const desc = document.getElementById('scDesc').value.trim();
  const body = document.getElementById('scBody_manual').value.trim();
  try{
    const resp = await fetch('/api/skill/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,description:desc,body})});
    if(!resp.ok){
      const err = await resp.json();
      alert('오류: '+(err.error||'알 수 없는 오류'));
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${name}.zip`; a.click();
    URL.revokeObjectURL(url);
    document.getElementById('scValidateResult').innerHTML += '<div style="color:#16a34a;font-size:12px;margin-top:4px;">📥 다운로드 완료!</div>';
  }catch(e){
    alert('다운로드 실패: '+e.message);
  }
}
function setScLang(lang){
  document.getElementById('scLang').value = lang;
  const ko = document.getElementById('scLangKo');
  const en = document.getElementById('scLangEn');
  if(lang==='ko'){
    ko.style.background='#6366f1'; ko.style.color='#fff'; ko.style.borderColor='#6366f1';
    en.style.background='#fff'; en.style.color='#333'; en.style.borderColor='#e2e8f0';
  } else {
    en.style.background='#6366f1'; en.style.color='#fff'; en.style.borderColor='#6366f1';
    ko.style.background='#fff'; ko.style.color='#333'; ko.style.borderColor='#e2e8f0';
  }
}
async function generateSkillLLM(){
  const topic = document.getElementById('scTopic').value.trim();
  if(!topic){ alert('스킬 주제를 입력하세요.'); return; }
  const skill_type = document.getElementById('scType').value;
  const details = document.getElementById('scDetails').value.trim();
  const lang = document.getElementById('scLang').value || 'ko';
  document.getElementById('scGenStatus').style.display = 'block';
  document.getElementById('scGenBtn').disabled = true;
  try{
  document.getElementById('scGenResult').style.display = 'none';
    const resp = await fetch('/api/skill/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,skill_type,details,lang})});
    const data = await resp.json();
    document.getElementById('scGenStatus').style.display = 'none';
    document.getElementById('scGenBtn').disabled = false;
    if(data.error){
      alert('생성 실패: '+data.error);
      return;
    }
    document.getElementById('scGenContent').value = data.content;
    document.getElementById('scGenResult').style.display = 'block';
    if(data.model){
      document.getElementById('scGenStatus').style.display = 'block';
      document.getElementById('scGenStatus').textContent = `✅ ${data.model}로 생성 완료`;
    }
  }catch(e){
    document.getElementById('scGenStatus').style.display = 'none';
    document.getElementById('scGenBtn').disabled = false;
    alert('LLM 호출 실패: '+e.message);
  }
}
async function validateGenerated(){
  const content = document.getElementById('scGenContent').value;
  try{
    const resp = await fetch('/api/skill/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    const data = await resp.json();
    const valid = _renderValidation('scGenValidateResult', data);
  }catch(e){
    document.getElementById('scGenValidateResult').innerHTML = `<div style="color:#ef4444;font-size:12px;">오류: ${e.message}</div>`;
  }
}
async function downloadGeneratedSkill(){
  const content = document.getElementById('scGenContent').value;
  // frontmatter에서 name 추출
  const m = content.match(/name:\s*(.+)/);
  const name = m ? m[1].trim() : 'generated-skill';
  // frontmatter에서 description 추출
  const dm = content.match(/description:\s*>?\s*\n?\s*(.+)/);
  const desc = dm ? dm[1].trim() : '';
  // body 추출 (frontmatter 이후)
  const fmEnd = content.indexOf('---', 3);
  const body = fmEnd > 0 ? content.substring(fmEnd+3).trim() : '';
  try{
    const resp = await fetch('/api/skill/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,description:desc||'Auto-generated skill',body})});
    if(!resp.ok){
      const err = await resp.json();
      alert('오류: '+(err.error||'알 수 없는 오류'));
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${name}.zip`; a.click();
    URL.revokeObjectURL(url);
    document.getElementById('scGenValidateResult').innerHTML += '<div style="color:#16a34a;font-size:12px;margin-top:4px;">📥 다운로드 완료!</div>';
  }catch(e){
    alert('다운로드 실패: '+e.message);
  }
}
async function applySkill(){
  const name = document.getElementById('scName').value.trim();
  const desc = document.getElementById('scDesc').value.trim();
  const body = document.getElementById('scBody_manual').value.trim();
  if(!confirm(`스킬 '${name}'을 바로 등록하시겠습니까?\nscientific-skills/${name}/ 폴더에 저장됩니다.`)) return;
  try{
    const resp = await fetch('/api/skill/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,description:desc,body})});
    const data = await resp.json();
    if(data.error){
      document.getElementById('scValidateResult').innerHTML = `<div style="color:#ef4444;font-size:12px;">❌ ${data.error}</div>`;
      if(data.errors) document.getElementById('scValidateResult').innerHTML += data.errors.map(e=>`<div style="color:#ef4444;font-size:11px;">&nbsp;&nbsp;- ${e}</div>`).join('');
    } else {
      document.getElementById('scValidateResult').innerHTML = `<div style="color:#16a34a;font-size:12px;">🚀 ${data.message}</div>`;
      if(data.warnings && data.warnings.length>0) document.getElementById('scValidateResult').innerHTML += data.warnings.map(w=>`<div style="color:#f59e0b;font-size:11px;">⚠️ ${w}</div>`).join('');
      // 스킬 카탈로그 새로고침
      setTimeout(()=>{ loadConfig(); }, 1500);
    }
  }catch(e){
    alert('적용 실패: '+e.message);
  }
}
async function applyGeneratedSkill(){
  const content = document.getElementById('scGenContent').value;
  const m = content.match(/name:\s*(.+)/);
  const name = m ? m[1].trim() : 'generated-skill';
  if(!confirm(`스킬 '${name}'을 바로 등록하시겠습니까?\nscientific-skills/${name}/ 폴더에 저장됩니다.`)) return;
  try{
    const resp = await fetch('/api/skill/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    const data = await resp.json();
    if(data.error){
      document.getElementById('scGenValidateResult').innerHTML = `<div style="color:#ef4444;font-size:12px;">❌ ${data.error}</div>`;
      if(data.errors) document.getElementById('scGenValidateResult').innerHTML += data.errors.map(e=>`<div style="color:#ef4444;font-size:11px;">&nbsp;&nbsp;- ${e}</div>`).join('');
    } else {
      document.getElementById('scGenValidateResult').innerHTML = `<div style="color:#16a34a;font-size:12px;">🚀 ${data.message}</div>`;
      if(data.warnings && data.warnings.length>0) document.getElementById('scGenValidateResult').innerHTML += data.warnings.map(w=>`<div style="color:#f59e0b;font-size:11px;">⚠️ ${w}</div>`).join('');
      setTimeout(()=>{ loadConfig(); }, 1500);
    }
  }catch(e){
    alert('적용 실패: '+e.message);
  }
}
function downloadExistingSkill(skillName){
  window.location.href = `/api/skill/${skillName}/download`;
}
async function editSkillMd(skillName){
  const body = document.getElementById('sdpBody');
  const d = currentSkillDetail;
  body.innerHTML = `<textarea id="sdpEditArea" rows="25" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;resize:vertical;box-sizing:border-box;">${esc(d.content||'')}</textarea>
    <div style="display:flex;gap:6px;margin-top:10px;">
      <button class="sdp-btn" style="background:#16a34a;color:#fff;" onclick="saveSkillMd('${skillName}')">💾 저장</button>
      <button class="sdp-btn" onclick="renderSdpTabs('skillmd')">취소</button>
    </div>
    <div id="sdpEditStatus" style="margin-top:6px;"></div>`;
}
async function saveSkillMd(skillName){
  const content = document.getElementById('sdpEditArea').value;
  const statusEl = document.getElementById('sdpEditStatus');
  try{
    const resp = await fetch(`/api/skill/${skillName}/update`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    const data = await resp.json();
    if(data.error){
      statusEl.innerHTML = `<span style="color:#ef4444">❌ ${data.error}</span>`;
      if(data.errors) statusEl.innerHTML += data.errors.map(e=>`<br><span style="color:#ef4444;font-size:11px;">&nbsp;&nbsp;- ${e}</span>`).join('');
    } else {
      statusEl.innerHTML = '<span style="color:#16a34a">✅ 저장 완료!</span>';
      if(data.warnings && data.warnings.length > 0){
        statusEl.innerHTML += data.warnings.map(w=>`<br><span style="color:#f59e0b;font-size:11px;">⚠️ ${w}</span>`).join('');
      }
      // 상세 패널 새로고침
      setTimeout(()=>openSkillDetail(skillName), 1000);
    }
  }catch(e){
    statusEl.innerHTML = `<span style="color:#ef4444">오류: ${e.message}</span>`;
  }
}

// ===== CSV Upload =====
let csvLoaded = false;
let csvFilename = '';

// handleUnifiedDrop/Select → 채팅 입력 📎 첨부로 대체됨
async function uploadCsvFile(file){
  const formData = new FormData();
  formData.append('file', file);

  try{
    const resp = await fetch('/api/upload_csv', {method:'POST', body:formData});
    const data = await resp.json();
    if(data.error) return;

    csvLoaded = true;
    csvFilename = data.filename;

    // 미리보기 테이블 생성
    let tableHtml = '<table><tr>' + data.headers.map(h=>`<th>${esc(h)}</th>`).join('') + '</tr>';
    if(data.sample_rows){
      data.sample_rows.forEach(row=>{
        tableHtml += '<tr>' + data.headers.map(h=>`<td>${esc(row[h]||'')}</td>`).join('') + '</tr>';
      });
    }
    tableHtml += '</table>';
    if(data.rows > 3) tableHtml += `<div style="text-align:center;color:#999;font-size:10px;margin-top:4px">... 총 ${data.rows}행</div>`;

    const panel = document.getElementById('csvInfoPanel');
    panel.style.display = 'block';
    panel.innerHTML = `
      <div class="csv-info">
        <div class="fname">📊 ${esc(data.filename)}</div>
        <div class="fstats">${data.rows}행 × ${data.cols}열</div>
        <button class="fremove" onclick="removeCsv()">✕ 제거</button>
        <div class="csv-preview">${tableHtml}</div>
      </div>`;

  }catch(e){}
}

// ===== XLSX Upload (시트별 파싱 + 이미지) =====
var _xlsxRawFile = null;  // 시트 변경 시 재업로드용

async function uploadXlsxFile(file, sheetIdx){
  _xlsxRawFile = file;
  const formData = new FormData();
  formData.append('file', file);
  if(typeof sheetIdx === 'number') formData.append('sheet', sheetIdx);

  try{
    const resp = await fetch('/api/upload_xlsx', {method:'POST', body:formData});
    const data = await resp.json();
    if(data.error){ addMsg('assistant', '❌ ' + data.error); return; }

    csvLoaded = true;
    csvFilename = data.filename;

    // 시트 선택 탭 (2개 이상일 때만)
    let sheetTabs = '';
    if(data.sheets && data.sheets.length > 1){
      sheetTabs = '<div style="display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap">';
      data.sheets.forEach(function(s, i){
        var active = i === data.active_sheet;
        sheetTabs += '<button style="padding:3px 10px;border-radius:12px;border:1.5px solid '+(active?'#6366f1':'#ddd')+';background:'+(active?'#6366f1':'#fff')+';color:'+(active?'#fff':'#333')+';font-size:11px;cursor:pointer;font-weight:'+(active?'600':'400')+'" onclick="switchXlsxSheet('+i+')">'+esc(s.name)+' <span style="color:'+(active?'rgba(255,255,255,.7)':'#999')+';font-size:10px">('+s.rows+'행)</span></button>';
      });
      sheetTabs += '</div>';
    }

    // 미리보기 테이블
    let tableHtml = '<table><tr>' + data.headers.map(function(h){return '<th>'+esc(h)+'</th>';}).join('') + '</tr>';
    if(data.sample_rows){
      data.sample_rows.forEach(function(row){
        tableHtml += '<tr>' + data.headers.map(function(h){return '<td>'+esc(row[h]||'')+'</td>';}).join('') + '</tr>';
      });
    }
    tableHtml += '</table>';
    if(data.rows > 3) tableHtml += '<div style="text-align:center;color:#999;font-size:10px;margin-top:4px">... 총 ' + data.rows + '행</div>';

    // 이미지 안내
    var imgInfo = '';
    if(data.images_count > 0){
      imgInfo = '<div style="margin-top:6px;padding:4px 8px;background:#fef3c7;border-radius:6px;font-size:11px;color:#92400e">🖼️ 내장 이미지 ' + data.images_count + '개 감지 (VL 모델로 분석 가능)</div>';
    }

    var sheetInfo = data.sheets && data.sheets.length > 1 ? ' · ' + data.sheets.length + '시트' : '';

    const panel = document.getElementById('csvInfoPanel');
    panel.style.display = 'block';
    panel.innerHTML =
      '<div class="csv-info">' +
        '<div class="fname">📊 ' + esc(data.filename) + ' [' + esc(data.active_sheet_name) + ']</div>' +
        '<div class="fstats">' + data.rows + '행 × ' + data.cols + '열' + sheetInfo + '</div>' +
        '<button class="fremove" onclick="removeCsv()">✕ 제거</button>' +
        sheetTabs +
        '<div class="csv-preview">' + tableHtml + '</div>' +
        imgInfo +
      '</div>';

  }catch(e){ console.warn('XLSX 업로드 실패:', e); }
}

function switchXlsxSheet(idx){
  if(!_xlsxRawFile) return;
  uploadXlsxFile(_xlsxRawFile, idx);
}

function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ===== Unicode-safe Base64 유틸 (btoa/atob는 Latin1만 지원) =====
function u2b(str){
  var bytes=new TextEncoder().encode(str);
  var bin='';
  for(var i=0;i<bytes.length;i++) bin+=String.fromCharCode(bytes[i]);
  return btoa(bin);
}
function b2u(b64){
  var bin=atob(b64);
  var bytes=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

// ===== Draw.io 기능 =====
function decodeDrawioXml(btn){
  return b2u(btn.dataset.xml);
}

// Draw.io XML 자동 수정 (LLM이 생성한 XML의 흔한 오류 교정)
function sanitizeDrawioXml(xml){
  // 허용되는 HTML 태그 목록 (Draw.io value 속성 안에서 허용)
  const allowedHtmlTags = ['br','br/','br /','font','b','i','u','sub','sup','hr','span','div','p','strong','em'];
  const allowedPattern = allowedHtmlTags.map(t => t.replace('/','\\/').replace(' ','')).join('|');

  // value/label 속성값 안의 이스케이프 수정 함수
  function fixAttrValue(val){
    // Step 1: & 이스케이프 (이미 된 건 건드리지 않음)
    let s = val.replace(/&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)/g, '&amp;');
    // Step 2: 모든 < > 를 임시 치환 후, 허용된 HTML 태그만 복원
    s = s.replace(/</g, '\x00LT\x00').replace(/>/g, '\x00GT\x00');
    // 허용된 여는 태그 복원: <font ...>, <br>, <br/>, <b>, etc.
    for(const tag of allowedHtmlTags){
      const base = tag.replace(/[\s\/]/g,'');
      // 여는 태그 (속성 포함): <font style="...">
      const openRe = new RegExp('\x00LT\x00(' + base + ')(\\s[^\\x00]*?)?\x00GT\x00', 'gi');
      s = s.replace(openRe, '<$1$2>');
      // 닫는 태그: </font>
      const closeRe = new RegExp('\x00LT\x00/' + base + '\x00GT\x00', 'gi');
      s = s.replace(closeRe, '</' + base + '>');
    }
    // self-closing: <br/>, <br />, <hr/>
    s = s.replace(/\x00LT\x00(br|hr)\s*\/?\x00GT\x00/gi, '<$1/>');
    // 남은 \x00LT\x00, \x00GT\x00 는 진짜 이스케이프 대상
    s = s.replace(/\x00LT\x00/g, '&lt;').replace(/\x00GT\x00/g, '&gt;');
    return s;
  }

  // 1) value="..." 속성 수정
  xml = xml.replace(/(value|label)="([^"]*)"/g, (m, attr, val) => {
    const fixed = fixAttrValue(val);
    return fixed !== val ? `${attr}="${fixed}"` : m;
  });

  // 2) 중복 속성 제거: style="..." style="..." → 첫 번째만
  xml = xml.replace(/(\w+="[^"]*")\s+\1/g, '$1');

  // 3) 중첩된 mxCell 분리 (mxCell 안에 mxCell이 있으면 평탄화)
  //    주의: mxCell > mxGeometry 관계는 정상이므로 건드리지 않음
  //    (기존 regex가 자식 mxGeometry 있는 mxCell을 self-closing으로 바꾸는 버그 수정)

  // 4) mxfile 래퍼가 없으면 추가
  if (!xml.includes('<mxfile') && xml.includes('<mxGraphModel')) {
    xml = `<mxfile host="app.diagrams.net" type="device">\n<diagram id="d1" name="Page-1">\n${xml}\n</diagram>\n</mxfile>`;
  }
  if (!xml.includes('<mxfile') && xml.includes('<mxCell')) {
    xml = `<mxfile host="app.diagrams.net" type="device">\n<diagram id="d1" name="Page-1">\n<mxGraphModel dx="1422" dy="762" grid="1" pageWidth="1169" pageHeight="827">\n<root>\n<mxCell id="0"/>\n<mxCell id="1" parent="0"/>\n${xml}\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>`;
  }

  // 5) 최종 검증: DOMParser로 파싱 시도, 실패하면 단계별 수정
  try {
    let parser = new DOMParser();
    let doc = parser.parseFromString(xml, 'application/xml');
    if (doc.querySelector('parsererror')) {
      // 5a) 속성값 안의 이스케이프 안 된 < > 수정
      xml = xml.replace(/="([^"]*)"/g, (m, val) => {
        if (val.includes('<') && !val.match(/^[^<]*<(mxGeometry|mxPoint|Array)/)) {
          const forced = val.replace(/</g, '&lt;').replace(/>/g, '&gt;');
          return `="${forced}"`;
        }
        return m;
      });

      // 5b) 재검증 - 여전히 실패하면 구조 수정 시도
      doc = parser.parseFromString(xml, 'application/xml');
      if (doc.querySelector('parsererror')) {
        // mxCell 구조 재구성: 모든 mxCell/mxGeometry를 추출해서 올바른 구조로 재조립
        const cellRe = /<mxCell\s[^>]*?\/>/g;
        const cellWithChildRe = /<mxCell\s[^>]*?>[\s\S]*?<\/mxCell>/g;
        const geoRe = /<mxGeometry\s[^>]*?\/>/g;
        const geoWithChildRe = /<mxGeometry\s[^>]*?>[\s\S]*?<\/mxGeometry>/g;

        // mxfile/diagram/mxGraphModel 래퍼 추출
        const wrapMatch = xml.match(/([\s\S]*?<root>)([\s\S]*?)(<\/root>[\s\S]*)/);
        if (wrapMatch) {
          const [, header, body, footer] = wrapMatch;
          // body에서 모든 완전한 요소 추출 (self-closing + with-children 모두)
          const elements = [];
          const allRe = /<(mxCell|mxGeometry|UserObject|Array|mxPoint)\s[^>]*?(?:\/>|>[\s\S]*?<\/\1>)/g;
          // 먼저 mxCell+children 패턴 추출
          const cellFullRe = /<mxCell\s[^>]*?>(?:\s*<mxGeometry[\s\S]*?(?:\/>|<\/mxGeometry>))*\s*<\/mxCell>/g;
          const cellSelfRe = /<mxCell\s[^>]*?\/>/g;
          let match;
          while ((match = cellFullRe.exec(body)) !== null) elements.push(match[0]);
          while ((match = cellSelfRe.exec(body)) !== null) {
            if (!elements.some(e => e.includes(match[0]))) elements.push(match[0]);
          }
          // UserObject 패턴도 포함
          const uoRe = /<UserObject[\s\S]*?<\/UserObject>/g;
          while ((match = uoRe.exec(body)) !== null) elements.push(match[0]);

          if (elements.length > 0) {
            xml = header + '\n' + elements.join('\n') + '\n' + footer;
          }
        }
      }
    }
  } catch(e) {}

  return xml;
}

function copyDrawioXml(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  navigator.clipboard.writeText(xml).then(()=>{
    const orig = btn.textContent;
    btn.textContent = '✓ 복사됨';
    setTimeout(()=> btn.textContent = orig, 1500);
  });
}
function downloadDrawio(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  const blob = new Blob([xml], {type:'application/xml'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'diagram.drawio';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function openInDrawio(btn){
  const xml = sanitizeDrawioXml(decodeDrawioXml(btn));
  const encoded = encodeURIComponent(xml);
  window.open('https://app.diagrams.net/#R' + encoded, '_blank');
}

// ===== Chart.js 인터랙티브 차트 =====
const _chartInstances = {};

function renderChartBlock(canvasId, jsonStr) {
  // jsonStr은 이미 b2u()로 유니코드 디코딩된 상태
  let raw = jsonStr;
  // 한 줄 주석 제거
  raw = raw.replace(/\/\/[^\n]*/g, '');
  let config;
  try {
    config = JSON.parse(raw);
  } catch(e) {
    // JSON5 스타일 허용: trailing comma, 작은따옴표
    raw = raw.replace(/,\s*([\]}])/g, '$1').replace(/'/g, '"');
    config = JSON.parse(raw);
  }

  // 기본값 보강
  if (!config.type) config.type = 'bar';
  if (!config.options) config.options = {};
  if (!config.options.responsive) config.options.responsive = true;
  if (!config.options.maintainAspectRatio) config.options.maintainAspectRatio = true;
  if (!config.options.plugins) config.options.plugins = {};

  // 한글 폰트 기본 설정
  if (!config.options.plugins.legend) config.options.plugins.legend = {};

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // 이전 인스턴스 정리
  if (_chartInstances[canvasId]) {
    _chartInstances[canvasId].destroy();
  }
  _chartInstances[canvasId] = new Chart(ctx, config);
}

function downloadChartPng(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const url = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chart.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function copyChartJson(btn) {
  const b64 = btn.getAttribute('data-chart');
  const json = b2u(b64);
  navigator.clipboard.writeText(json).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ 복사됨';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

// ===== MD 다운로드/복사/미리보기 =====
function decodeMdB64(btn){
  const b64 = btn.getAttribute('data-md');
  return b2u(b64);
}
function downloadMarkdown(btn){
  const md = decodeMdB64(btn);
  // 제목에서 파일명 추출
  const titleMatch = md.match(/^#\s+(.+)$/m);
  let filename = 'report.md';
  if(titleMatch){
    filename = titleMatch[1].replace(/[^\w가-힣\s-]/g,'').trim().replace(/\s+/g,'_').substring(0,50) + '.md';
  }
  const blob = new Blob([md], {type:'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function copyMdContent(btn){
  const md = decodeMdB64(btn);
  navigator.clipboard.writeText(md).then(()=>{
    const orig = btn.textContent;
    btn.textContent = '✅ 복사됨';
    setTimeout(()=>btn.textContent=orig, 1500);
  });
}
function renderMdPreview(escapedMd){
  // HTML escaped 마크다운을 간단히 렌더링 (미리보기용)
  let s = escapedMd;
  // 헤딩
  s = s.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  s = s.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  s = s.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  s = s.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  s = s.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  // bold / italic
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // 체크리스트
  s = s.replace(/^- \[x\]\s+(.+)$/gm, '<div>☑️ $1</div>');
  s = s.replace(/^- \[ \]\s+(.+)$/gm, '<div>⬜ $1</div>');
  // 리스트
  s = s.replace(/^- (.+)$/gm, '<li>$1</li>');
  s = s.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
  // 수평선
  s = s.replace(/^---+$/gm, '<hr>');
  // 블록쿼트
  s = s.replace(/^&gt;\s+(.+)$/gm, '<blockquote>$1</blockquote>');
  // 인라인 코드
  s = s.replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:12px">$1</code>');
  // 테이블 (간단 처리)
  s = s.replace(/((?:^\|.+\|[ ]*\n){2,})/gm, function(tbl){
    const rows = tbl.trim().split('\n').filter(r=>r.trim());
    if(rows.length < 2) return tbl;
    const parseRow = r => r.replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());
    const hdr = parseRow(rows[0]);
    let startIdx = 1;
    if(rows[1] && /^[\s|:-]+$/.test(rows[1])) startIdx = 2;
    let html = '<table><thead><tr>' + hdr.map(h=>`<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    for(let i=startIdx;i<rows.length;i++){
      const cells = parseRow(rows[i]);
      html += '<tr>' + cells.map(c=>`<td>${c}</td>`).join('') + '</tr>';
    }
    return html + '</tbody></table>';
  });
  // 줄바꿈
  s = s.replace(/\n/g, '<br>');
  return s;
}

// ===== 피드백 시스템 =====
var _fbCurrentRating = '';
var _fbCurrentBtn = null;

function openFeedback(btn, rating){
  _fbCurrentRating = rating;
  _fbCurrentBtn = btn;
  var modal = document.getElementById('feedbackModal');
  var title = document.getElementById('fbModalTitle');
  var display = document.getElementById('fbRatingDisplay');
  var ta = document.getElementById('fbComment');
  if(rating === 'good'){
    title.textContent = '어떤 점이 좋았나요?';
    display.textContent = '👍 좋아요';
    ta.placeholder = '좋았던 점을 알려주세요 (선택사항)';
  } else {
    title.textContent = '어떤 점이 아쉬웠나요?';
    display.textContent = '👎 별로예요';
    ta.placeholder = '개선할 점을 알려주세요 (선택사항)';
  }
  ta.value = '';
  modal.classList.add('show');
  setTimeout(function(){ ta.focus(); }, 100);
}

function closeFeedbackModal(){
  document.getElementById('feedbackModal').classList.remove('show');
  _fbCurrentRating = '';
  _fbCurrentBtn = null;
}

async function submitFeedback(){
  var comment = document.getElementById('fbComment').value;
  // 해당 메시지 내용 가져오기
  var msgEl = _fbCurrentBtn ? _fbCurrentBtn.closest('.msg') : null;
  var msgText = msgEl ? (msgEl.textContent || '').slice(0, 500) : '';
  // 직전 유저 메시지
  var userQuery = '';
  if(msgEl){
    var prev = msgEl.previousElementSibling;
    while(prev){
      if(prev.classList.contains('user')){ userQuery = (prev.textContent || '').slice(0, 300); break; }
      prev = prev.previousElementSibling;
    }
  }
  try{
    await fetch('/api/feedback', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ rating: _fbCurrentRating, comment: comment, message: msgText, user_query: userQuery })
    });
  }catch(e){}
  // 버튼 시각 피드백
  if(_fbCurrentBtn){
    var fbDiv = _fbCurrentBtn.parentElement;
    fbDiv.querySelectorAll('button').forEach(function(b){ b.classList.remove('fb-selected-good','fb-selected-bad'); });
    _fbCurrentBtn.classList.add(_fbCurrentRating === 'good' ? 'fb-selected-good' : 'fb-selected-bad');
  }
  closeFeedbackModal();
}

// 모달 바깥 클릭으로 닫기
document.addEventListener('click', function(e){
  var modal = document.getElementById('feedbackModal');
  if(e.target === modal) closeFeedbackModal();
});

// ===== 응답 전체를 MD 다운로드 가능하게 =====
function appendMdDownloadBar(rawContent, htmlDownloadUrl){
  // 마지막 assistant 메시지에 MD/HTML 다운로드 바 추가
  const msgs = document.querySelectorAll('.msg.assistant');
  if(msgs.length === 0) return;
  const lastMsg = msgs[msgs.length - 1];
  const mdB64 = u2b(rawContent);
  const bar = document.createElement('div');
  bar.className = 'md-report-header';
  bar.style.marginTop = '10px';
  bar.style.borderRadius = '8px';
  bar.innerHTML = `<span>📋 마크다운 문서로 저장</span>
    <div class="md-report-actions">
      <button class="md-report-btn" onclick="copyMdContent(this)" data-md="${mdB64}">📋 복사</button>
      <button class="md-report-btn md-download" onclick="downloadMarkdown(this)" data-md="${mdB64}">📥 MD 다운로드</button>
      <button class="md-report-btn" style="background:#6366f1;color:#fff;" onclick="downloadHtmlDoc(this)" data-md="${mdB64}" ${htmlDownloadUrl ? 'data-html-url="'+htmlDownloadUrl+'"' : ''}>📄 HTML 다운로드</button>
    </div>`;
  lastMsg.appendChild(bar);
}

async function downloadHtmlDoc(btn){
  const htmlUrl = btn.dataset.htmlUrl;
  if(htmlUrl){
    const a = document.createElement('a');
    a.href = htmlUrl;
    a.download = 'document.html';
    a.click();
    return;
  }
  // URL 없으면 API로 변환 요청
  const md = b2u(btn.dataset.md);
  btn.disabled = true;
  btn.textContent = '⏳ 변환 중...';
  try{
    const res = await fetch('/api/generate_html',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({markdown:md, title:'문서'})
    });
    const data = await res.json();
    if(data.download_url){
      const a = document.createElement('a');
      a.href = data.download_url;
      a.download = 'document.html';
      a.click();
    } else {
      alert(data.error || 'HTML 변환 실패');
    }
  }catch(e){
    alert('HTML 변환 실패: '+e.message);
  }finally{
    btn.disabled = false;
    btn.textContent = '📄 HTML 다운로드';
  }
}

async function removeCsv(){
  await fetch('/api/clear_csv', {method:'POST'});
  csvLoaded = false;
  csvFilename = '';
  document.getElementById('csvInfoPanel').style.display = 'none';
}

// ===== 파일 관리 =====
let uploadedFilesList = [];

function renderFileList(){
  const panel = document.getElementById('fileListPanel');
  if(uploadedFilesList.length === 0){
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';
  let html = '<div class="uploaded-files-header"><span>📎 첨부 파일 ('+uploadedFilesList.length+')</span><button onclick="clearAllFiles()" style="background:none;border:none;color:#e74c3c;cursor:pointer;font-size:10px;padding:0">전체 제거</button></div>';
  uploadedFilesList.forEach((f,i)=>{
    const sizeStr = f.size > 1048576 ? (f.size/1048576).toFixed(1)+'MB' : (f.size/1024).toFixed(1)+'KB';
    const pendingCls = f.pending ? ' ufi-pending' : (f.drm ? ' ufi-drm' : '');
    const statusLabel = f.pending ? ' ⏳' : (f.drm ? ' 🔒' : '');
    const removeAction = f.pending
      ? `removeChatAttachByName('${esc(f.filename)}')`
      : `removeFile('${esc(f.filename)}',${i})`;
    const titleText = f.drm ? 'DRM 보호 파일 - 내용 분석 불가' : esc(f.preview||'');
    html += `<div class="uploaded-file-item${pendingCls}">
      <span class="ufi-icon">${f.icon}</span>
      <span class="ufi-name" title="${titleText}">${esc(f.filename)}${statusLabel}</span>
      <span class="ufi-size">${sizeStr}</span>
      <button class="ufi-remove" onclick="${removeAction}">✕</button>
    </div>`;
  });
  panel.innerHTML = html;
}

async function removeFile(filename, idx){
  await fetch('/api/remove_file', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename})});
  uploadedFilesList.splice(idx, 1);
  renderFileList();
}

async function clearAllFiles(){
  await fetch('/api/clear_files', {method:'POST'});
  uploadedFilesList = [];
  renderFileList();
}

// ===== 채팅 입력 첨부파일 =====
let chatPendingFiles = [];  // [{file:File, name:string}]

// 파일 확장자 → 추천 스킬 매핑 (프론트엔드용)
const fileExtSkillHints = {
  'py':['agent-python-pro'], 'js':['agent-nextjs-developer'], 'ts':['agent-nextjs-developer'], 'tsx':['agent-nextjs-developer'],
  'csv':['exploratory-data-analysis','statistical-analysis'],
  'tsv':['exploratory-data-analysis'], 'xlsx':['exploratory-data-analysis'],
  'docx':['docx'], 'pdf':['pdf'], 'pptx':['pptx'],
  'drawio':['drawio-diagram'], 'xml':['drawio-diagram'],
  'md':['markdown-mermaid-writing'], 'html':['web-artifacts-builder'],
};

function handleChatFileSelect(files){
  let hintSkills = [];
  const iconMap = {py:'🐍',js:'📜',ts:'📘',tsx:'📘',jsx:'📜',html:'🌐',css:'🎨',json:'📋',
    yaml:'📋',yml:'📋',md:'📄',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',
    png:'🖼️',jpg:'🖼️',jpeg:'🖼️',gif:'🖼️',bmp:'🖼️',webp:'🖼️',svg:'🖼️',
    docx:'📄',xlsx:'📊',pdf:'📕',pptx:'📽️',csv:'📊',txt:'📝',sql:'🗃️',rb:'💎',php:'🐘',vue:'💚',r:'📈'};
  for(const f of files){
    chatPendingFiles.push({file:f, name:f.name});
    // 파일 확장자로 추천 스킬 수집
    const ext = f.name.split('.').pop().toLowerCase();
    if(fileExtSkillHints[ext]){
      hintSkills.push(...fileExtSkillHints[ext]);
    }
    // 즉시 사이드바에 pending 상태로 표시
    uploadedFilesList.push({
      filename: f.name, type: ext, ext: ext,
      size: f.size, icon: iconMap[ext]||'📎', preview: '', pending: true
    });
  }
  renderChatAttach();
  renderFileList();
  document.getElementById('chatFileInput').value = '';

  // 파일 기반 추천 스킬 프리로드 (수동 선택 없을 때만)
  if(hintSkills.length > 0 && selSkills.length === 0){
    const unique = [...new Set(hintSkills)].filter(id => !selSkills.includes(id));
    autoLoadedSkills = [...new Set([...autoLoadedSkills, ...unique])];
    renderSkills();
    updateLoaded();
  }
}

function removeChatAttach(idx){
  const removed = chatPendingFiles.splice(idx, 1);
  // 사이드바 pending 항목도 함께 제거
  if(removed.length > 0){
    const pi = uploadedFilesList.findIndex(f => f.pending && f.filename === removed[0].name);
    if(pi >= 0) uploadedFilesList.splice(pi, 1);
    renderFileList();
  }
  renderChatAttach();
}
function removeChatAttachByName(name){
  const ci = chatPendingFiles.findIndex(f => f.name === name);
  if(ci >= 0) chatPendingFiles.splice(ci, 1);
  const pi = uploadedFilesList.findIndex(f => f.pending && f.filename === name);
  if(pi >= 0) uploadedFilesList.splice(pi, 1);
  renderChatAttach();
  renderFileList();
}

// 입력창 드래그앤드롭 파일 첨부
(function(){
  const box = document.querySelector('.chat-box-fixed');
  if(!box) return;
  box.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();box.style.borderTopColor='#6366f1';});
  box.addEventListener('dragleave', e=>{e.preventDefault();box.style.borderTopColor='';});
  box.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();box.style.borderTopColor='';
    if(e.dataTransfer.files.length > 0) handleChatFileSelect(e.dataTransfer.files);
  });
})();

// Ctrl+V / Cmd+V 이미지 붙여넣기 (스크린샷, 클립보드 이미지)
(function(){
  const input = document.getElementById('input');
  if(!input) return;
  input.addEventListener('paste', function(e){
    const items = e.clipboardData && e.clipboardData.items;
    if(!items) return;
    const imageFiles = [];
    for(let i=0; i<items.length; i++){
      if(items[i].type.startsWith('image/')){
        const blob = items[i].getAsFile();
        if(blob){
          // 파일명 생성: paste_날짜시간.확장자
          const ext = items[i].type.split('/')[1] || 'png';
          const now = new Date();
          const ts = now.getFullYear()+String(now.getMonth()+1).padStart(2,'0')+String(now.getDate()).padStart(2,'0')+'_'+String(now.getHours()).padStart(2,'0')+String(now.getMinutes()).padStart(2,'0')+String(now.getSeconds()).padStart(2,'0');
          const fname = 'paste_'+ts+'.'+ext.replace('jpeg','jpg');
          const file = new File([blob], fname, {type: items[i].type});
          imageFiles.push(file);
        }
      }
    }
    if(imageFiles.length > 0){
      e.preventDefault();  // 텍스트로 붙여넣기 방지
      handleChatFileSelect(imageFiles);
    }
  });
})();

function renderChatAttach(){
  const wrap = document.getElementById('chatAttachPreview');
  if(chatPendingFiles.length === 0){
    wrap.style.display = 'none';
    wrap.innerHTML = '';
    return;
  }
  wrap.style.display = 'flex';
  wrap.innerHTML = chatPendingFiles.map((f,i)=>
    `<span class="chat-attach-badge"><span class="fname">${esc(f.name)}</span><span class="remove-attach" onclick="removeChatAttach(${i})">✕</span></span>`
  ).join('');
}

async function uploadChatPendingFiles(){
  // 첨부된 파일들을 서버로 업로드 (CSV는 csv 엔드포인트, 나머지는 범용)
  // pending 항목을 실제 업로드 결과로 교체
  const results = [];
  for(const pf of chatPendingFiles){
    const ext = pf.name.split('.').pop().toLowerCase();
    const formData = new FormData();
    formData.append('file', pf.file);

    // pending 항목 찾기
    const pendingIdx = uploadedFilesList.findIndex(f => f.pending && f.filename === pf.name);

    // CSV/TSV → CSV 전용 엔드포인트 (데이터 분석용)
    if(['csv','tsv'].includes(ext) && !csvLoaded){
      try{
        await uploadCsvFile(pf.file);
        results.push(pf.name);
        if(pendingIdx >= 0) uploadedFilesList[pendingIdx].pending = false;
      }catch(e){
        if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
      }
      continue;
    }

    // XLSX → 전용 엔드포인트 (시트별 파싱 + 이미지 추출)
    if(ext === 'xlsx' && !csvLoaded){
      try{
        await uploadXlsxFile(pf.file);
        results.push(pf.name);
        if(pendingIdx >= 0) uploadedFilesList[pendingIdx].pending = false;
      }catch(e){
        if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
      }
      continue;
    }

    // 나머지 → 범용 파일 엔드포인트
    try{
      const resp = await fetch('/api/upload_file', {method:'POST', body:formData});
      const data = await resp.json();
      if(!data.error){
        // pending 항목을 실제 데이터로 교체
        const updated = {
          filename: data.filename, type: data.type, ext: data.ext,
          size: data.size, icon: data.icon, preview: data.preview,
          drm: !!data.drm_warning,
        };
        if(pendingIdx >= 0) uploadedFilesList[pendingIdx] = updated;
        else uploadedFilesList.push(updated);
        results.push(data.filename);
        // DRM 보호 파일이면 채팅에 안내 메시지 표시
        if(data.drm_warning){
          addMsg('assistant', `🔒 **DRM 보호 파일 감지**: ${data.filename}\n${data.drm_warning}\n\n파일은 첨부되었지만, DRM 때문에 내용 분석이 불가능합니다.`);
        }
      } else {
        // 업로드 실패 시 pending 제거 + 사용자에게 알림
        if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
        addMsg('assistant', data.error);
      }
    }catch(e){
      if(pendingIdx >= 0) uploadedFilesList.splice(pendingIdx, 1);
    }
  }
  chatPendingFiles = [];
  renderChatAttach();
  renderFileList();
  return results;
}

// ===== 오른쪽 코드 어시스턴트 패널 =====
let rpFiles = [];  // [{name, path, content, ext, size}]
let rpSelectedMode = null;
let rpTreeData = null;  // 트리 구조 데이터
let rpSelectedFile = null;  // 현재 선택된 파일 인덱스
const rpModeLabels = {
  explain:'코드 설명', find_bugs:'버그 탐지', improve:'개선 제안',
  tests:'테스트 생성', docstring:'문서화', refactor:'리팩토링'
};
const rpModePrompts = {
  explain:'[EXPLAIN] 아래 코드를 분석해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어와 프레임워크를 판별한 후, 해당 언어의 관점에서 분석하세요.\n1. 감지된 언어/프레임워크 명시\n2. 전체 목적과 아키텍처\n3. 주요 함수/클래스 구조\n4. 실행 흐름과 핵심 알고리즘\n5. 의존성 분석\n한국어로 설명해주세요.',
  find_bugs:'[FIND_BUGS] 아래 코드에서 버그를 찾아주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어 특유의 버그 패턴을 중점 검토하세요.\n1. 감지된 언어 명시\n2. 언어 특화 버그 패턴 (Python: 인덴트/타입, Java: NPE/동시성, C#: async/dispose, JS: 타입강제/this바인딩)\n3. 로직 오류, 보안 취약점\n4. 심각도 🔴🟡🟢 표시 + 수정 코드 제시\n한국어로 분석해주세요.',
  improve:'[IMPROVE] 아래 코드의 개선점을 제안해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 베스트 프랙티스 기준으로 개선하세요.\n1. 감지된 언어 명시\n2. 언어별 관용적 표현 적용 (Python: PEP8/컴프리헨션, Java: Stream API, C#: LINQ, JS: ES6+)\n3. 가독성, 성능, 유지보수성 개선\n4. 개선 전/후 코드 비교\n한국어로 설명해주세요.',
  tests:'[TESTS] 아래 코드에 대한 테스트 코드를 생성해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 표준 테스트 프레임워크를 사용하세요.\n(Python→pytest, Java→JUnit5, C#→xUnit, JS/TS→Jest, Go→go test, Rust→cargo test)\n1. 감지된 언어/프레임워크 명시\n2. 정상/엣지/에러 케이스\n3. 즉시 실행 가능한 코드\n한국어 주석으로 설명해주세요.',
  docstring:'[DOCSTRING] 아래 코드에 문서화를 추가해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 표준 문서화 스타일을 적용하세요.\n(Python→Google/NumPy docstring, Java→Javadoc, C#→XML 주석, JS/TS→JSDoc)\n1. 감지된 언어 명시\n2. 함수/클래스별 문서화 (파라미터, 반환값, 예외, 예시)\n3. 핵심 로직 인라인 주석 (한국어)\n⚠️ 중요: 전체 파일 내용을 반복하지 말고, 문서화가 추가된 핵심 함수/클래스만 제시하세요.',
  refactor:'[REFACTOR] 아래 코드를 리팩토링해주세요.\n⚠️ 먼저 코드의 프로그래밍 언어를 판별하고, 해당 언어의 디자인 패턴과 관례에 맞게 리팩토링하세요.\n1. 감지된 언어 명시\n2. SOLID 원칙 적용\n3. 언어별 관용 패턴 (Python: 데코레이터/컨텍스트매니저, Java: Builder/DI, C#: async패턴)\n4. 리팩토링 전/후 비교\n한국어로 설명해주세요.'
};

// 코드 어시스턴트 전용 언어/도구 감지 매핑
const rpExtLangMap = {
  'py':'Python','js':'JavaScript','ts':'TypeScript','tsx':'TypeScript(React)','jsx':'JavaScript(React)',
  'java':'Java','c':'C','cpp':'C++','h':'C/C++ Header','cs':'C#',
  'go':'Go','rs':'Rust','rb':'Ruby','php':'PHP','swift':'Swift','kt':'Kotlin',
  'html':'HTML','css':'CSS','json':'JSON','xml':'XML','yaml':'YAML','yml':'YAML',
  'sh':'Shell','bat':'Batch','sql':'SQL','r':'R','scala':'Scala','dart':'Dart',
  'md':'Markdown','dockerfile':'Docker','makefile':'Make','toml':'TOML','cfg':'Config','ini':'Config'
};
const rpExtSkillMap = {
  'py':['agent-python-pro'], 'js':['agent-javascript-pro','agent-nextjs-developer'],
  'ts':['agent-javascript-pro','agent-nextjs-developer'], 'tsx':['agent-javascript-pro','agent-nextjs-developer'],
  'jsx':['agent-javascript-pro','agent-nextjs-developer'],
  'java':['code-assistant'], 'c':['code-assistant'], 'cpp':['code-assistant'], 'cs':['code-assistant'],
  'go':['code-assistant'], 'rs':['code-assistant'], 'rb':['code-assistant'], 'php':['code-assistant'],
  'swift':['code-assistant'], 'kt':['code-assistant'], 'html':['web-artifacts-builder'],
  'css':['web-artifacts-builder'], 'sql':['code-assistant'], 'r':['code-assistant'],
  'dart':['code-assistant'], 'scala':['code-assistant']
};
const rpLangColors = {
  'Python':'#3776ab','JavaScript':'#f7df1e','TypeScript':'#3178c6','TypeScript(React)':'#3178c6',
  'JavaScript(React)':'#61dafb','Java':'#b07219','C':'#555555','C++':'#f34b7d','C#':'#178600',
  'Go':'#00add8','Rust':'#dea584','Ruby':'#701516','PHP':'#4f5d95','Swift':'#f05138','Kotlin':'#a97bff',
  'HTML':'#e34c26','CSS':'#563d7c','Shell':'#89e051','SQL':'#e38c00','R':'#198ce7','Dart':'#00b4ab'
};

// 코드 요약: 큰 파일은 시그니처+핵심부분만 추출
function summarizeCode(content, ext, maxLines){
  maxLines = maxLines || 150;
  const lines = content.split('\n');
  if(lines.length <= maxLines) return content; // 작은 파일은 전체 전달

  const result = [];
  const sigPatterns = {
    'py': /^\s*(class |def |async def |import |from .+ import|@)/,
    'java': /^\s*(public |private |protected |class |interface |import |@|package )/,
    'cs': /^\s*(public |private |protected |internal |class |interface |using |namespace |\[)/,
    'js': /^\s*(function |const |let |var |class |export |import |async |module\.)/,
    'ts': /^\s*(function |const |let |var |class |export |import |async |interface |type |module\.)/,
    'tsx': /^\s*(function |const |let |var |class |export |import |async |interface |type )/,
    'jsx': /^\s*(function |const |let |var |class |export |import |async )/,
    'go': /^\s*(func |type |package |import |var |const )/,
    'rs': /^\s*(fn |pub |struct |enum |impl |use |mod |trait )/,
    'rb': /^\s*(class |def |module |require |include |attr_)/,
    'php': /^\s*(function |class |namespace |use |public |private |protected )/,
    'swift': /^\s*(func |class |struct |enum |protocol |import |var |let )/,
    'kt': /^\s*(fun |class |object |interface |import |val |var |data class)/,
    'c': /^\s*(#include |#define |typedef |struct |int |void |char |float |double |static )/,
    'cpp': /^\s*(#include |#define |typedef |struct |class |template |namespace |int |void )/,
    'h': /^\s*(#include |#define |#ifndef |typedef |struct |class |extern )/,
  };
  const pat = sigPatterns[ext] || /^\s*(function |class |def |import |export |public |private )/;

  // 1) 첫 30줄 (imports, 설정)
  result.push('// === 파일 시작 (첫 30줄) ===');
  result.push(...lines.slice(0, 30));

  // 2) 시그니처 라인 추출 (함수/클래스 정의 + 전후 2줄)
  result.push('');
  result.push('// === 주요 시그니처 & 구조 ===');
  const added = new Set();
  lines.forEach((line, i)=>{
    if(i < 30) return; // 이미 포함됨
    if(pat.test(line)){
      for(let j=Math.max(i-1,30); j<=Math.min(lines.length-1,i+3); j++){
        if(!added.has(j)){
          added.add(j);
          result.push(`[L${j+1}] ${lines[j]}`);
        }
      }
      result.push('  ...');
    }
  });

  // 3) 마지막 10줄
  result.push('');
  result.push('// === 파일 끝 (마지막 10줄) ===');
  result.push(...lines.slice(-10));

  result.push(`\n// [총 ${lines.length}줄 중 시그니처 요약 - 전체 코드 필요시 요청하세요]`);
  return result.join('\n');
}

// catalog에 실제 존재하는 스킬인지 확인
function isSkillInCatalog(skillId){
  for(const did of Object.keys(catalog)){
    const d = catalog[did];
    if(d && d.skills && d.skills.some(s=>s.id===skillId && s.available)) return true;
  }
  return false;
}

function detectRpLanguages(){
  const langCount = {};
  const detectedSkills = new Set();

  rpFiles.forEach(f=>{
    const lang = rpExtLangMap[f.ext];
    if(lang){
      langCount[lang] = (langCount[lang]||0) + 1;
    }
    const skills = rpExtSkillMap[f.ext];
    if(skills) skills.forEach(s=>detectedSkills.add(s));
  });

  const panel = document.getElementById('rpLangDetect');
  const tagsEl = document.getElementById('rpLangTags');
  const autoEl = document.getElementById('rpAutoSkills');

  if(rpFiles.length === 0 || Object.keys(langCount).length === 0){
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';

  // 감지된 언어 태그 렌더링 (파일 수 기준 내림차순)
  const sorted = Object.entries(langCount).sort((a,b)=>b[1]-a[1]);
  const primaryLang = sorted[0][0];
  tagsEl.innerHTML = sorted.map(([lang, cnt])=>{
    const color = rpLangColors[lang] || '#6b7280';
    const isPrimary = lang === primaryLang;
    return `<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:10px;background:${color}22;border:${isPrimary?'2':'1'}px solid ${color}${isPrimary?'':'55'};color:${color};font-weight:600;font-size:10px;">${isPrimary?'⭐ ':''}${lang} <span style="font-weight:400;color:#888">${cnt}</span></span>`;
  }).join('');

  // 자동 스킬 로드 - catalog에 실제 있는 것만
  const newSkills = [];
  const loadedNames = [];
  // code-assistant 항상 먼저
  if(isSkillInCatalog('code-assistant') && !selSkills.includes('code-assistant')){
    selSkills.push('code-assistant');
    newSkills.push('code-assistant');
  }
  if(selSkills.includes('code-assistant')) loadedNames.push('code-assistant');

  detectedSkills.forEach(skill=>{
    if(skill === 'code-assistant') return;
    if(isSkillInCatalog(skill)){
      if(!selSkills.includes(skill)){
        selSkills.push(skill);
        newSkills.push(skill);
      }
      loadedNames.push(skill);
    }
  });

  // autoLoadedSkills 업데이트 (UI에서 🧠자동 배지 표시용)
  autoLoadedSkills = [...new Set([...autoLoadedSkills, ...newSkills])];

  if(newSkills.length > 0){
    renderSkills();
    updateLoaded();
  }

  // 상태 표시
  const availableLoaded = loadedNames.filter(s=>s!=='code-assistant');
  if(availableLoaded.length > 0){
    autoEl.innerHTML = '✅ <b>자동 로드:</b> ' + availableLoaded.join(', ') + ' + code-assistant';
  } else {
    autoEl.innerHTML = '✅ <b>주 언어:</b> ' + primaryLang + ' → LLM이 자동 판별하여 분석합니다';
  }
}

function toggleRightPanel(){
  const rp = document.getElementById('rightPanel');
  const btn = document.getElementById('rightPanelToggle');
  rp.classList.toggle('collapsed');
  document.body.classList.toggle('rp-collapsed');
  if(rp.classList.contains('collapsed')){
    btn.textContent = '◀';
    document.querySelector('.chat-box-fixed').style.right = '0';
  } else {
    btn.textContent = '▶';
    document.querySelector('.chat-box-fixed').style.right = '340px';
  }
  localStorage.setItem('demos_rp_collapsed', rp.classList.contains('collapsed'));
}

// 초기 상태 복원
(function(){
  const saved = localStorage.getItem('demos_rp_collapsed');
  if(saved === 'true'){
    const rp = document.getElementById('rightPanel');
    const btn = document.getElementById('rightPanelToggle');
    if(rp){ rp.classList.add('collapsed'); document.body.classList.add('rp-collapsed'); btn.textContent='◀'; }
  }
})();

// 드래그앤드롭
(function(){
  const drop = document.getElementById('rpFileDrop');
  if(!drop) return;
  drop.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();drop.classList.add('dragover');});
  drop.addEventListener('dragleave', e=>{e.preventDefault();drop.classList.remove('dragover');});
  drop.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();drop.classList.remove('dragover');
    if(e.dataTransfer.files.length > 0) handleRpFileSelect(e.dataTransfer.files);
  });
})();

function switchUploadMode(mode, btn){
  document.querySelectorAll('.rp-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('rpFolderMode').style.display = mode==='folder' ? 'block' : 'none';
  document.getElementById('rpFileMode').style.display = mode==='files' ? 'block' : 'none';
}

function handleRpFolderSelect(files){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt','toml','cfg','ini','env','gitignore','dockerfile','makefile'];
  const skipDirs = ['node_modules','.git','__pycache__','.venv','venv','.next','dist','build','.idea','.vscode'];
  let pending = 0;
  let totalSize = 0;

  for(const f of files){
    const path = f.webkitRelativePath || f.name;
    // 불필요한 디렉토리 스킵
    const parts = path.split('/');
    if(parts.some(p => skipDirs.includes(p))) continue;

    const ext = f.name.split('.').pop().toLowerCase();
    if(!codeExts.includes(ext) && f.name.indexOf('.') !== -1) continue;
    if(f.size > 500000) continue; // 500KB 이상 파일 스킵

    pending++;
    totalSize += f.size;
    const reader = new FileReader();
    reader.onload = function(e){
      rpFiles.push({name:f.name, path:path, content:e.target.result, ext:ext, size:f.size});
      pending--;
      if(pending === 0){
        buildRpTree();
        renderRpTree();
        updateRpRunBtn();
        detectRpLanguages();
      }
    };
    reader.readAsText(f);
  }
  document.getElementById('rpFolderInput').value = '';
}

function buildRpTree(){
  rpTreeData = {name:'root', children:{}, files:[]};
  rpFiles.forEach((f,idx)=>{
    const parts = (f.path || f.name).split('/');
    let node = rpTreeData;
    // 첫 번째 파트는 루트 폴더명
    for(let i=0; i<parts.length-1; i++){
      if(!node.children[parts[i]]){
        node.children[parts[i]] = {name:parts[i], children:{}, files:[]};
      }
      node = node.children[parts[i]];
    }
    node.files.push({name:parts[parts.length-1], idx:idx, ext:f.ext, size:f.size});
  });
}

let rpCheckedFiles = new Set();  // 체크된 파일 인덱스

function renderRpTree(){
  const container = document.getElementById('rpTree');
  const stats = document.getElementById('rpTreeStats');
  const selectBar = document.getElementById('rpSelectBar');
  if(rpFiles.length === 0){
    container.style.display = 'none';
    stats.style.display = 'none';
    selectBar.style.display = 'none';
    return;
  }
  container.style.display = 'block';
  stats.style.display = 'flex';
  selectBar.style.display = 'flex';

  // 기본: 전체 체크
  if(rpCheckedFiles.size === 0){
    rpFiles.forEach((_,i)=>rpCheckedFiles.add(i));
  }

  // 통계
  const folders = new Set();
  rpFiles.forEach(f=>{
    const parts = (f.path||f.name).split('/');
    for(let i=1;i<parts.length;i++) folders.add(parts.slice(0,i).join('/'));
  });
  const totalSize = rpFiles.reduce((s,f)=>s+f.size,0);
  const sizeStr = totalSize > 1048576 ? (totalSize/1048576).toFixed(1)+'MB' : (totalSize/1024).toFixed(1)+'KB';
  document.getElementById('rpStatsInfo').textContent = rpFiles.length+' 파일 / '+folders.size+' 폴더';
  document.getElementById('rpStatsSize').textContent = sizeStr;

  container.innerHTML = renderTreeNode(rpTreeData, true);
  updateCheckedCount();
  updateParentFolderChecks();
}

function renderTreeNode(node, isRoot){
  let html = '';
  // 폴더들
  const childKeys = Object.keys(node.children).sort();
  childKeys.forEach(key=>{
    const child = node.children[key];
    const fileCount = countFiles(child);
    const folderIdxs = collectFileIdxs(child);
    const allChecked = folderIdxs.every(i=>rpCheckedFiles.has(i));
    const someChecked = folderIdxs.some(i=>rpCheckedFiles.has(i));
    html += `<div class="rp-tree-node rp-tree-folder">
      <input type="checkbox" class="rp-tree-cb" data-folder-idxs="${folderIdxs.join(',')}" ${allChecked?'checked':''} onclick="event.stopPropagation();toggleFolderCheck(this)" onchange="event.stopPropagation()">
      <span class="rp-tree-node-content" onclick="toggleTreeFolder(this.parentElement)"><span class="rp-tree-toggle">▶</span><span class="rp-tree-icon">📁</span>${esc(key)} <span style="color:#bbb;font-weight:400">(${fileCount})</span></span>
    </div><div class="rp-tree-children" style="display:none">${renderTreeNode(child, false)}</div>`;
  });
  // 파일들
  node.files.sort((a,b)=>a.name.localeCompare(b.name)).forEach(f=>{
    const langIcon = {py:'🐍',js:'📜',html:'🌐',css:'🎨',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',json:'📋',yaml:'📋',yml:'📋',md:'📄',ts:'📘',tsx:'📘'}[f.ext]||'📄';
    const checked = rpCheckedFiles.has(f.idx);
    html += `<div class="rp-tree-node rp-tree-file" data-idx="${f.idx}">
      <input type="checkbox" class="rp-tree-cb" data-idx="${f.idx}" ${checked?'checked':''} onclick="event.stopPropagation();toggleFileCheck(${f.idx},this)">
      <span class="rp-tree-node-content" onclick="selectTreeFile(${f.idx},this.parentElement)"><span class="rp-tree-toggle"> </span><span class="rp-tree-icon">${langIcon}</span>${esc(f.name)}</span>
    </div>`;
  });
  return html;
}

function collectFileIdxs(node){
  let idxs = node.files.map(f=>f.idx);
  Object.values(node.children).forEach(ch=>{ idxs = idxs.concat(collectFileIdxs(ch)); });
  return idxs;
}

function toggleFileCheck(idx, cb){
  if(cb.checked) rpCheckedFiles.add(idx);
  else rpCheckedFiles.delete(idx);
  updateCheckedCount();
  updateParentFolderChecks();
}

function toggleFolderCheck(cb){
  const idxs = cb.dataset.folderIdxs.split(',').filter(Boolean).map(Number).filter(n=>!isNaN(n));
  if(cb.checked) idxs.forEach(i=>rpCheckedFiles.add(i));
  else idxs.forEach(i=>rpCheckedFiles.delete(i));
  // 하위 파일 + 하위 폴더 체크박스 모두 동기화
  const parent = cb.closest('.rp-tree-folder');
  const childContainer = parent ? parent.nextElementSibling : null;
  if(childContainer && childContainer.classList.contains('rp-tree-children')){
    childContainer.querySelectorAll('input.rp-tree-cb').forEach(childCb=>{
      childCb.checked = cb.checked;
      childCb.indeterminate = false;
    });
  }
  updateCheckedCount();
  updateParentFolderChecks();
}

function updateParentFolderChecks(){
  document.querySelectorAll('.rp-tree-cb[data-folder-idxs]').forEach(cb=>{
    const idxs = cb.dataset.folderIdxs.split(',').filter(Boolean).map(Number).filter(n=>!isNaN(n));
    const checkedCount = idxs.filter(i=>rpCheckedFiles.has(i)).length;
    cb.checked = checkedCount === idxs.length;
    cb.indeterminate = checkedCount > 0 && checkedCount < idxs.length;
  });
}

function rpCheckAll(checked){
  if(checked) rpFiles.forEach((_,i)=>rpCheckedFiles.add(i));
  else rpCheckedFiles.clear();
  document.querySelectorAll('.rp-tree-cb').forEach(cb=>{ cb.checked=checked; cb.indeterminate=false; });
  updateCheckedCount();
}

function updateCheckedCount(){
  const count = rpCheckedFiles.size;
  document.getElementById('rpCheckedCount').textContent = count+'개 선택';
  updateRpRunBtn();
}

function countFiles(node){
  let c = node.files.length;
  Object.values(node.children).forEach(ch=>{ c += countFiles(ch); });
  return c;
}

function toggleTreeFolder(el){
  const children = el.nextElementSibling;
  const toggle = el.querySelector('.rp-tree-toggle');
  if(children.style.display === 'none'){
    children.style.display = 'block';
    toggle.textContent = '▼';
  } else {
    children.style.display = 'none';
    toggle.textContent = '▶';
  }
}

function selectTreeFile(idx, el){
  document.querySelectorAll('.rp-tree-file').forEach(e=>e.classList.remove('active'));
  if(el) el.classList.add('active');
  rpSelectedFile = idx;
}

function clearRpProject(){
  rpFiles = [];
  rpTreeData = null;
  rpSelectedFile = null;
  rpCheckedFiles.clear();
  document.getElementById('rpTree').style.display = 'none';
  document.getElementById('rpTreeStats').style.display = 'none';
  document.getElementById('rpLangDetect').style.display = 'none';
  document.getElementById('rpSelectBar').style.display = 'none';
  document.getElementById('rpFileList').innerHTML = '';
  updateRpRunBtn();
}

function handleRpFileSelect(files){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt'];
  for(const f of files){
    const ext = f.name.split('.').pop().toLowerCase();
    if(!codeExts.includes(ext)){ alert(f.name+': 지원하지 않는 파일 형식입니다.'); continue; }
    const reader = new FileReader();
    reader.onload = function(e){
      rpFiles.push({name:f.name, content:e.target.result, ext:ext, size:f.size});
      renderRpFiles();
      updateRpRunBtn();
      detectRpLanguages();
    };
    reader.readAsText(f);
  }
  document.getElementById('rpFileInput').value = '';
}

function renderRpFiles(){
  const list = document.getElementById('rpFileList');
  if(rpFiles.length === 0){ list.innerHTML = ''; return; }
  list.innerHTML = rpFiles.map((f,i)=>{
    const sizeStr = f.size > 1048576 ? (f.size/1048576).toFixed(1)+'MB' : (f.size/1024).toFixed(1)+'KB';
    const langIcon = {py:'🐍',js:'📜',html:'🌐',css:'🎨',java:'☕',c:'⚙️',cpp:'⚙️',go:'🐹',rs:'🦀',sh:'💻',json:'📋',yaml:'📋',yml:'📋',md:'📄',ts:'📘',tsx:'📘'}[f.ext] || '📄';
    return `<div class="rp-file-item">
      <span>${langIcon}</span><span class="fname">${esc(f.name)}</span><span class="fsize">${sizeStr}</span>
      <button class="fremove" onclick="event.stopPropagation();removeRpFile(${i})">✕</button>
    </div>`;
  }).join('');
}

function removeRpFile(idx){
  rpFiles.splice(idx, 1);
  renderRpFiles();
  updateRpRunBtn();
}


function selectRpMode(btn){
  document.querySelectorAll('.rp-action-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  rpSelectedMode = btn.dataset.mode;
  updateRpRunBtn();
}

function updateRpRunBtn(){
  const btn = document.getElementById('rpRunBtn');
  const checkedCount = rpCheckedFiles.size;
  if(rpFiles.length > 0 && rpSelectedMode && checkedCount > 0){
    btn.disabled = false;
    btn.textContent = '▶ ' + rpModeLabels[rpSelectedMode] + ' (' + checkedCount + '개 파일)';
  } else if(rpFiles.length > 0 && checkedCount === 0){
    btn.disabled = true;
    btn.textContent = '▶ 분석할 파일을 선택하세요 (체크박스)';
  } else {
    btn.disabled = true;
    btn.textContent = rpFiles.length === 0 ? '▶ 코드 파일을 업로드하세요' : '▶ 분석 모드를 선택하세요';
  }
}

async function runCodeAssistant(){
  if(rpFiles.length === 0 || !rpSelectedMode) return;
  if(!selEnvs||selEnvs.length===0){ alert('먼저 LLM 환경을 선택해주세요.'); return; }

  // code-assistant 스킬 자동 로드
  if(!selSkills.includes('code-assistant')){
    selSkills.push('code-assistant');
    renderSkills();
    updateLoaded();
  }

  // 체크된 파일만 필터링
  const selectedFiles = rpFiles.filter((_,i)=>rpCheckedFiles.has(i));
  if(selectedFiles.length === 0){ alert('분석할 파일을 선택해주세요 (체크박스).'); return; }

  // 프로젝트 트리 구조 생성
  let treeText = '';
  if(rpTreeData){
    treeText = '=== 프로젝트 구조 ===\n' + buildTreeText(rpTreeData, '') + '\n';
    treeText += `\n(전체 ${rpFiles.length}개 중 ${selectedFiles.length}개 파일 선택됨)\n`;
  }

  // 선택된 파일 내용을 스마트하게 추려서 구성
  let codeContent = treeText;
  const maxLinesPerFile = selectedFiles.length <= 3 ? 300 : selectedFiles.length <= 8 ? 150 : 80;

  selectedFiles.forEach(f=>{
    const summarized = summarizeCode(f.content, f.ext, maxLinesPerFile);
    codeContent += `\n--- 파일: ${f.path||f.name} (${f.ext}) ---\n${summarized}\n`;
  });

  const prompt = rpModePrompts[rpSelectedMode] + '\n\n' + codeContent;
  const modeLabel = rpModeLabels[rpSelectedMode] || rpSelectedMode;

  // 모든 분석 모드 → 채팅으로 전송 (대화 이어가기 위해)
  {
    const fileNames = selectedFiles.map(f=>f.name).join(', ');
    const shortLabel = '\u{1f4ca} [' + modeLabel + '] ' + fileNames;
    addMsg('user', shortLabel);
    history.push({role:'user', content:prompt});

    const typing = addTyping();
    isSending = true;
    const btn = document.getElementById('sendBtn');
    btn.textContent = '\u23f9';
    btn.classList.add('stop-mode');
    btn.disabled = false;
    chatAbort = new AbortController();

    // 버튼 비활성화
    const runBtn = document.getElementById('rpRunBtn');
    runBtn.disabled = true;
    runBtn.textContent = '\u23f3 분석 중...';

    try{
      const resp = await fetch('/api/chat',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        signal: chatAbort.signal,
        body:JSON.stringify({
          env: ['coder-480b'],
          messages: history,
          skills: [...selSkills],
          effort,
          format: formatManualOverride ? selFormat : 'auto',
          writing_style: styleManualOverride ? document.getElementById('writingStyle').value.trim() : 'auto',
          system_prompt: document.getElementById('systemPromptInput').value.trim(),
          max_tokens: maxTokens,
          think_mode: document.getElementById('thinkToggle').checked,
        })
      });
      const data = await resp.json();
      typing.remove();
      if(data.error){
        addMsg('assistant','\u274c '+data.error);
      } else {
        let info = '';
        if(data.loaded_skills && data.loaded_skills.length > 0){
          let extra = data.tokens_budget ? ' ['+data.tokens_budget+']' : '';
          let _se2 = selEnvs[0]||'auto';
          let mName = data.model_used || (envs[_se2] ? envs[_se2].name : (selEnvs.length>=2 ? selEnvs.length+'개 병렬' : _se2));
          info = '\n[\u2705 '+data.loaded_skills.join(', ')+'] ['+mName+'] ('+(data.system_prompt_length ?? 0)+'\uc790)'+extra;
        }
        if(data.auto_routed){ info += ' [\uD83E\uDD16 자동: '+data.model_used+' ('+data.route_reason+')]'; }
        if(data.fallback_used){ info += ' [\u26A0\uFE0F 대체: '+data.fallback_from+' \u2192 '+data.model_used+']'; }
        if(data.parallel_agents && data.parallel_agents >= 2){
          let pG = (data.parallel_groups||[]).join(', ');
          let pM = (data.parallel_models||[]).join(', ');
          let pS = data.parallel_synthesis==='fallback_concat'?'(단순연결)':'(합성)';
          info += ' [🔀 병렬 '+data.parallel_agents+'에이전트: '+pG+'] [모델: '+pM+'] '+pS;
          if(data.parallel_failed>0) info += ' [⚠️ '+data.parallel_failed+'개 실패]';
        }
        let truncWarn = data.truncated ? '\n\n⚠️ **응답이 토큰 한도('+maxTokens+')에 도달하여 잘렸습니다.** "계속 이어서 작성해줘"라고 입력하면 이어서 받을 수 있습니다.' : '';
        const assistantDisplayText = data.content + truncWarn + info;
        const assistantRawForDetect = data.content + truncWarn;
        addMsg('assistant', assistantDisplayText, assistantRawForDetect);
        history.push({role:'assistant', content:data.content});
      }
    }catch(e){
      typing.remove();
      if(e.name !== 'AbortError'){
        addMsg('assistant','\u274c 서버 연결 실패: '+e.message);
      }
    }
    isSending = false;
    chatAbort = null;
    btn.textContent = '\u25b6';
    btn.classList.remove('stop-mode');
    btn.disabled = false;
    document.getElementById('input').focus();
    runBtn.disabled = false;
    updateRpRunBtn();
    saveCurrentSession();
  }
}

let rpLastAnalysisContent = '';

function toggleRpResultExpand(){
  document.getElementById('rpResult').classList.toggle('expanded');
}
function copyRpResult(){
  const body = document.querySelector('.rp-result-body');
  if(body) navigator.clipboard.writeText(body.innerText).then(()=>alert('복사 완료!'));
}
function sendRpResultToChat(){
  if(!rpLastAnalysisContent) return;
  addMsg('assistant', rpLastAnalysisContent);
  history.push({role:'assistant', content: rpLastAnalysisContent});
  saveCurrentSession();
}

function buildTreeText(node, prefix){
  let text = '';
  const childKeys = Object.keys(node.children).sort();
  const allItems = [...childKeys.map(k=>({type:'dir',name:k,node:node.children[k]})), ...node.files.map(f=>({type:'file',name:f.name}))];
  allItems.forEach((item,i)=>{
    const isLast = i === allItems.length - 1;
    const connector = isLast ? '└── ' : '├── ';
    const nextPrefix = prefix + (isLast ? '    ' : '│   ');
    if(item.type === 'dir'){
      text += prefix + connector + '📁 ' + item.name + '/\n';
      text += buildTreeText(item.node, nextPrefix);
    } else {
      text += prefix + connector + item.name + '\n';
    }
  });
  return text;
}

// 폴더 드래그앤드롭
(function(){
  const drop = document.getElementById('rpFolderDrop');
  if(!drop) return;
  drop.addEventListener('dragover', e=>{e.preventDefault();e.stopPropagation();drop.classList.add('dragover');});
  drop.addEventListener('dragleave', e=>{e.preventDefault();drop.classList.remove('dragover');});
  drop.addEventListener('drop', e=>{
    e.preventDefault();e.stopPropagation();drop.classList.remove('dragover');
    // 폴더 드롭은 DataTransferItem으로 처리
    if(e.dataTransfer.items){
      const items = [...e.dataTransfer.items];
      const entries = items.map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
      if(entries.length > 0){
        readEntriesRecursive(entries);
        return;
      }
    }
    if(e.dataTransfer.files.length > 0) handleRpFolderSelect(e.dataTransfer.files);
  });
})();

function readEntriesRecursive(entries){
  const codeExts = ['py','js','html','css','json','xml','yaml','yml','c','cpp','h','java','go','rs','sh','bat','md','txt','ts','tsx','jsx','rb','php','swift','kt','toml','cfg','ini'];
  const skipDirs = ['node_modules','.git','__pycache__','.venv','venv','.next','dist','build','.idea','.vscode'];
  let pending = 0;

  function readEntry(entry, path){
    if(entry.isFile){
      const ext = entry.name.split('.').pop().toLowerCase();
      if(!codeExts.includes(ext)) return;
      pending++;
      entry.file(f=>{
        if(f.size > 500000){ pending--; return; }
        const reader = new FileReader();
        reader.onload = function(e){
          rpFiles.push({name:f.name, path:path+f.name, content:e.target.result, ext:ext, size:f.size});
          pending--;
          if(pending===0){ buildRpTree(); renderRpTree(); updateRpRunBtn(); detectRpLanguages(); }
        };
        reader.readAsText(f);
      });
    } else if(entry.isDirectory){
      if(skipDirs.includes(entry.name)) return;
      const dirReader = entry.createReader();
      dirReader.readEntries(entries=>{
        entries.forEach(e=>readEntry(e, path+entry.name+'/'));
      });
    }
  }
  entries.forEach(e=>readEntry(e, ''));
}

// 초기 실행 버튼 상태
updateRpRunBtn();

// ==================== 인트로 → 모드 선택 시스템 ====================
// ═══ 로그인 시스템 ═══
var currentUser = null; // {id, name, role, token}

function dismissIntroAndShowMode(){
  var ov = document.getElementById('introOverlay');
  if(!ov || ov._gone) return;
  ov._gone = true;
  var vi = document.getElementById('introVideo');
  if(vi) vi.pause();
  ov.classList.add('fade-out');
  setTimeout(function(){
    if(ov.parentNode) ov.parentNode.removeChild(ov);
    // 매번 로그인 필수
    showLoginScreen();
  }, 700);
}

function showLoginScreen(){
  // 메인 UI 숨기기
  document.getElementById('sidebar').style.display='none';
  var mainEl = document.querySelector('.main');
  if(mainEl) mainEl.style.display='none';
  var chatBox = document.querySelector('.chat-box-fixed');
  if(chatBox) chatBox.style.display='none';
  var sideToggle = document.querySelector('.sidebar-toggle');
  if(sideToggle) sideToggle.style.display='none';
  // 로그인 화면 표시
  var lo = document.getElementById('loginOverlay');
  if(lo) lo.style.display='flex';
  var idInput = document.getElementById('loginId');
  if(idInput) idInput.focus();
  // PPT 설계 버튼 숨김 (로그인 전)
  var pptBtn = document.getElementById('pptBuilderBtn');
  if(pptBtn) pptBtn.style.display='none';
}

function showMainUI(){
  var lo = document.getElementById('loginOverlay');
  if(lo) lo.style.display='none';
  document.getElementById('sidebar').style.display='';
  var mainEl = document.querySelector('.main');
  if(mainEl) mainEl.style.display='';
  var chatBox = document.querySelector('.chat-box-fixed');
  if(chatBox) chatBox.style.display='';
  var sideToggle = document.querySelector('.sidebar-toggle');
  if(sideToggle) sideToggle.style.display='';
  // 헤더 버튼 표시/숨김
  var hdAdmin = document.getElementById('hdAdminBtn');
  if(hdAdmin) hdAdmin.style.display = (currentUser && currentUser.role === 'admin') ? '' : 'none';
  // PPT 설계 버튼 노출 (로그인 완료)
  var pptBtn = document.getElementById('pptBuilderBtn');
  if(pptBtn) pptBtn.style.display='flex';
}

async function doLogin(){
  var uid = document.getElementById('loginId').value.trim();
  var pw = document.getElementById('loginPw').value;
  var errEl = document.getElementById('loginError');
  if(!uid || !pw){ errEl.textContent='아이디와 비밀번호를 입력하세요.'; errEl.style.display='block'; return; }
  try {
    var resp = await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:uid,password:pw})});
    var data = await resp.json();
    if(data.error){ errEl.textContent=data.error; errEl.style.display='block'; return; }
    currentUser = data;
    localStorage.setItem('demos_user', JSON.stringify(data));
    errEl.style.display='none';
    showMainUI();
  } catch(e){
    errEl.textContent='서버 연결 실패'; errEl.style.display='block';
  }
}

function doLogout(){
  currentUser = null;
  localStorage.removeItem('demos_user');
  var adminBtn = document.getElementById('adminBtn');
  if(adminBtn) adminBtn.remove();
  var logoutBtn = document.getElementById('logoutBtn');
  if(logoutBtn) logoutBtn.remove();
  showLoginScreen();
}

// ═══ ADMIN 사용자 관리 팝업 ═══
function openAdminPopup(){
  document.getElementById('adminPopup').style.display='flex';
  loadAdminUsers();
}
function closeAdminPopup(){
  document.getElementById('adminPopup').style.display='none';
}

async function loadAdminUsers(){
  var list = document.getElementById('adminUserList');
  try {
    var resp = await fetch('/api/auth/users',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({admin_id:currentUser.id})});
    var data = await resp.json();
    if(data.error){ list.innerHTML='<div style="color:#ef4444;">'+data.error+'</div>'; return; }
    list.innerHTML = data.users.map(function(u){
      var isMe = u.id === currentUser.id;
      var roleBadge = u.role==='admin' ? '<span style="background:#f59e0b;color:#000;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;">ADMIN</span>' : '<span style="background:#6366f1;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;">USER</span>';
      var actions = isMe ? '<span style="color:#666;font-size:11px;">현재 사용자</span>' : '<button onclick="deleteUser(\''+u.id+'\')" style="padding:3px 8px;background:#ef4444;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;">삭제</button>';
      return '<div style="display:flex;justify-content:space-between;align-items:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px;margin-bottom:4px;"><div><span style="font-weight:600;">'+u.name+'</span> <span style="color:#888;font-size:12px;">('+u.id+')</span> '+roleBadge+'<br><span style="color:#666;font-size:11px;">등록: '+u.created+'</span></div><div>'+actions+'</div></div>';
    }).join('');
  } catch(e){
    list.innerHTML='<div style="color:#ef4444;">로드 실패</div>';
  }
}

async function registerUser(){
  var msg = document.getElementById('regMsg');
  var name = document.getElementById('regName').value.trim();
  var id = document.getElementById('regId').value.trim();
  var pw = document.getElementById('regPw').value;
  var role = document.getElementById('regRole').value;
  if(!name||!id||!pw){ msg.innerHTML='<span style="color:#ef4444;">모든 항목을 입력하세요.</span>'; return; }
  try {
    var resp = await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({admin_id:currentUser.id,new_id:id,new_name:name,new_password:pw,new_role:role})});
    var data = await resp.json();
    if(data.error){ msg.innerHTML='<span style="color:#ef4444;">'+data.error+'</span>'; return; }
    msg.innerHTML='<span style="color:#22c55e;">✅ '+data.message+'</span>';
    document.getElementById('regName').value='';
    document.getElementById('regId').value='';
    document.getElementById('regPw').value='';
    loadAdminUsers();
  } catch(e){
    msg.innerHTML='<span style="color:#ef4444;">등록 실패</span>';
  }
}

async function deleteUser(targetId){
  if(!confirm(targetId+' 사용자를 삭제하시겠습니까?')) return;
  try {
    await fetch('/api/auth/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({admin_id:currentUser.id,target_id:targetId})});
    loadAdminUsers();
  } catch(e){}
}

// ═══ 소개 페이지 팝업 ═══
function openAboutPopup(){
  var popup = window.open('/beno/about.html', 'aboutDemos', 'width=900,height=700,scrollbars=yes,resizable=yes');
  if(popup) popup.focus();
}

// ═══ 지식 도메인 관리 ═══
function openKnowledgePopup(){
  document.getElementById('knowledgePopup').style.display='flex';
  switchKdTab('upload');
  loadKdFiles();
}
function closeKnowledgePopup(){
  document.getElementById('knowledgePopup').style.display='none';
}
function switchKdTab(tab){
  document.getElementById('kdUploadTab').style.display = tab==='upload' ? 'block' : 'none';
  document.getElementById('kdManualTab').style.display = tab==='manual' ? 'block' : 'none';
  document.getElementById('kdTabUpload').style.background = tab==='upload' ? '#6366f1' : '#2a2a3e';
  document.getElementById('kdTabUpload').style.color = tab==='upload' ? '#fff' : '#888';
  document.getElementById('kdTabManual').style.background = tab==='manual' ? '#6366f1' : '#2a2a3e';
  document.getElementById('kdTabManual').style.color = tab==='manual' ? '#fff' : '#888';
}
function kdFileSelected(input){
  var name = input.files[0]?.name || '';
  document.getElementById('kdFileName').textContent = name ? '선택: '+name : '';
}
async function uploadKdFile(){
  var input = document.getElementById('kdFileInput');
  var msg = document.getElementById('kdUploadMsg');
  if(!input.files[0]){ msg.innerHTML='<span style="color:#ef4444;">파일을 선택하세요.</span>'; return; }
  var fd = new FormData();
  fd.append('file', input.files[0]);
  fd.append('user_id', currentUser.id);
  try {
    var resp = await fetch('/api/knowledge/upload',{method:'POST',body:fd});
    var data = await resp.json();
    if(data.error){ msg.innerHTML='<span style="color:#ef4444;">'+data.error+'</span>'; return; }
    msg.innerHTML='<span style="color:#22c55e;">'+data.message+'</span>';
    input.value='';
    document.getElementById('kdFileName').textContent='';
    loadKdFiles();
  } catch(e){ msg.innerHTML='<span style="color:#ef4444;">업로드 실패</span>'; }
}
async function saveKdManual(){
  var msg = document.getElementById('kdManualMsg');
  var title = document.getElementById('kdTitle').value.trim();
  var cat = document.getElementById('kdCategory').value;
  if(cat==='_custom'){
    cat = document.getElementById('kdCategoryCustom').value.trim();
    if(!cat){ msg.innerHTML='<span style="color:#ef4444;">카테고리명을 입력하세요.</span>'; return; }
  }
  var tags = document.getElementById('kdTags').value.trim();
  var content = document.getElementById('kdContent').value.trim();
  if(!title||!content){ msg.innerHTML='<span style="color:#ef4444;">제목과 내용을 입력하세요.</span>'; return; }
  try {
    var resp = await fetch('/api/knowledge/manual',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:currentUser.id,title:title,category:cat,content:content,tags:tags})});
    var data = await resp.json();
    if(data.error){ msg.innerHTML='<span style="color:#ef4444;">'+data.error+'</span>'; return; }
    msg.innerHTML='<span style="color:#22c55e;">'+data.message+'</span>';
    document.getElementById('kdTitle').value='';
    document.getElementById('kdContent').value='';
    document.getElementById('kdTags').value='';
    loadKdFiles();
  } catch(e){ msg.innerHTML='<span style="color:#ef4444;">저장 실패</span>'; }
}
async function loadKdFiles(){
  var list = document.getElementById('kdFileList');
  try {
    var resp = await fetch('/api/knowledge/list',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:currentUser.id})});
    var data = await resp.json();
    if(!data.files||data.files.length===0){ list.innerHTML='<div style="color:#666;padding:8px;">등록된 파일이 없습니다.</div>'; return; }
    var catLabels = {column_definition:'컬럼',architecture:'아키텍처',specification:'명세',changelog:'변경',plan:'계획',guide:'가이드'};
    list.innerHTML = data.files.map(function(f){
      var catBadge = catLabels[f.category]||f.category;
      return '<div style="display:flex;justify-content:space-between;align-items:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px;margin-bottom:3px;">'
        +'<div><span style="font-weight:600;color:#e2e8f0;">'+f.title+'</span>'
        +' <span style="background:#334155;color:#94a3b8;padding:1px 6px;border-radius:4px;font-size:10px;">'+catBadge+'</span>'
        +'<br><span style="color:#555;font-size:11px;">'+f.filename+' | '+f.size_kb+'KB | '+f.date+'</span></div>'
        +'<button onclick="deleteKdFile(\''+f.filename+'\')" style="padding:3px 8px;background:#ef4444;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;">삭제</button></div>';
    }).join('');
  } catch(e){ list.innerHTML='<div style="color:#ef4444;">로드 실패</div>'; }
}
async function deleteKdFile(fname){
  if(!confirm(fname+' 파일을 삭제하시겠습니까?')) return;
  try {
    await fetch('/api/knowledge/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:currentUser.id,filename:fname})});
    loadKdFiles();
  } catch(e){}
}

var uioCollapsed = false;

function toggleUioMode(){
  var btn = document.getElementById('uioBackBtn');
  if(!uioCollapsed){
    // 접기: UIO 숨기고 메인 UI 복원 (iframe 유지)
    document.getElementById('uioContainer').style.display='none';
    document.getElementById('sidebar').style.display='';
    var mainEl = document.querySelector('.main');
    if(mainEl) mainEl.style.display='';
    var chatBox = document.querySelector('.chat-box-fixed');
    if(chatBox) chatBox.style.display='';
    var sideToggle = document.querySelector('.sidebar-toggle');
    if(sideToggle) sideToggle.style.display='';
    btn.textContent='🎮 UIO 펼치기';
    btn.style.position='fixed';
    btn.style.display='block';
    uioCollapsed = true;
  } else {
    // 펼치기: UIO 다시 전체화면
    document.getElementById('uioContainer').style.display='block';
    document.getElementById('sidebar').style.display='none';
    var mainEl = document.querySelector('.main');
    if(mainEl) mainEl.style.display='none';
    var chatBox = document.querySelector('.chat-box-fixed');
    if(chatBox) chatBox.style.display='none';
    var sideToggle = document.querySelector('.sidebar-toggle');
    if(sideToggle) sideToggle.style.display='none';
    btn.textContent='← 접기';
    uioCollapsed = false;
  }
}

function exitUioMode(){
  document.getElementById('uioContainer').style.display='none';
  document.getElementById('uioFrame').src='about:blank';
  document.getElementById('uioBackBtn').style.display='';
  uioCollapsed = false;
  // 기본 UI 복원
  document.getElementById('sidebar').style.display='';
  var mainEl = document.querySelector('.main');
  if(mainEl) mainEl.style.display='';
  var chatBox = document.querySelector('.chat-box-fixed');
  if(chatBox) chatBox.style.display='';
  var sideToggle = document.querySelector('.sidebar-toggle');
  if(sideToggle) sideToggle.style.display='';
}

(function(){
  // 메인 UI를 처음에 숨기기 (로그인 전)
  document.getElementById('sidebar').style.display='none';
  var mainEl = document.querySelector('.main');
  if(mainEl) mainEl.style.display='none';
  var chatBox = document.querySelector('.chat-box-fixed');
  if(chatBox) chatBox.style.display='none';
  var sideToggle = document.querySelector('.sidebar-toggle');
  if(sideToggle) sideToggle.style.display='none';

  // 바로 로그인 화면 표시
  showLoginScreen();
})();

// ==================== PPT 코드 감지 & 생성 ====================
function detectAndAddPptxButtons(msgEl, rawText){
  // python-pptx 코드 블록 감지 (from pptx import ... 또는 python-pptx 패턴)
  const codeBlockRegex = /```(?:python|py)?\s*\n([\s\S]*?)```/gi;
  let match;
  const pptxCodes = [];
  while((match = codeBlockRegex.exec(rawText)) !== null){
    const code = match[1];
    if(code.includes('from pptx') || code.includes('import pptx') || code.includes('Presentation(') || code.includes('add_slide')){
      pptxCodes.push(code.trim());
    }
  }
  if(pptxCodes.length === 0) return;

  // 각 PPT 코드 블록 아래에 생성 버튼 삽입
  const allPre = msgEl.querySelectorAll('pre');
  let codeIdx = 0;
  allPre.forEach(pre => {
    const preText = pre.textContent;
    if((preText.includes('from pptx') || preText.includes('import pptx') || preText.includes('Presentation(') || preText.includes('add_slide')) && codeIdx < pptxCodes.length){
      const code = pptxCodes[codeIdx];
      codeIdx++;
      const block = document.createElement('div');
      block.className = 'pptx-gen-block';
      block.innerHTML = `
        <div class="pptx-gen-header">
          <span>📽️ PowerPoint 생성</span>
          <div class="pptx-gen-actions">
            <span class="pptx-gen-status"></span>
            <button class="pptx-gen-btn secondary" onclick="copyPptxCode(this)">📋 실행용 코드 복사</button>
            <button class="pptx-gen-btn" onclick="generatePptx(this)">📽️ PPT 생성 & 다운로드</button>
          </div>
        </div>
        <div class="pptx-remake-bar">
          <span class="pptx-remake-label">🔄 다시 만들기:</span>
          <button class="pptx-remake-btn" onclick="_remakePpt('그래프/차트 없이 텍스트와 표 위주로 PPT 다시 만들어줘')">📝 그래프 없이</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('그래프와 차트를 포함해서 PPT 다시 만들어줘')">📊 그래프 포함</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('디자인을 더 심플하게 PPT 다시 만들어줘')">🎨 심플하게</button>
          <button class="pptx-remake-btn" onclick="_remakePpt('슬라이드 수를 줄여서 핵심만 PPT 다시 만들어줘')">✂️ 핵심만</button>
        </div>`;
      block.dataset.pptxCode = u2b(code);
      pre.parentNode.insertBefore(block, pre.nextSibling);
    }
  });
}

async function copyPptxCode(btn){
  const block = btn.closest('.pptx-gen-block');
  const rawCode = b2u(block.dataset.pptxCode);
  let codeToCopy = rawCode;

  try{
    const resp = await fetch('/api/prepare_pptx_code', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code: rawCode})
    });
    if(resp.ok){
      const data = await resp.json();
      if(data && data.code) codeToCopy = data.code;
    }
  }catch(e){}

  navigator.clipboard.writeText(codeToCopy);
  btn.textContent = '✓ 복사됨';
  setTimeout(()=>{ btn.textContent = '📋 실행용 코드 복사'; }, 1500);
}

async function generatePptx(btn){
  const block = btn.closest('.pptx-gen-block');
  const code = b2u(block.dataset.pptxCode);
  const status = block.querySelector('.pptx-gen-status');

  btn.disabled = true;
  btn.textContent = '⏳ 생성 중...';
  status.textContent = 'python-pptx 코드 실행 중...';

  try{
    const resp = await fetch('/api/generate_pptx', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({code, filename:'presentation.pptx'})
    });

    if(!resp.ok){
      const err = await resp.json();
      status.textContent = '❌ ' + (err.error || '생성 실패');
      btn.textContent = '📽️ 재시도';
      btn.disabled = false;
      return;
    }

    const json = await resp.json();
    if(json.download_url){
        window.location.href = json.download_url;
        status.textContent = '✅ 다운로드 완료!';
    } else if(json.message) {
        status.textContent = '✅ ' + json.message;
    } else {
        status.textContent = '❌ 파일 응답 형식 오류';
    }
    btn.textContent = '📽️ 다시 생성';
    btn.disabled = false;
  }catch(e){
    status.textContent = '❌ 네트워크 오류';
    btn.textContent = '📽️ 재시도';
    btn.disabled = false;
  }
}

function _remakePpt(text){
  const input = document.getElementById('input');
  if(input){ input.value = text; input.focus(); input.style.height='auto'; input.style.height=input.scrollHeight+'px'; }
  // 자동 전송
  setTimeout(()=>send(), 100);
}

// ===== PPT 디자인 스타일 시스템 =====
const PPT_STYLE_PROMPTS = {
  minimal: '미니멀 디자인: 흰 배경(#FFFFFF), 검정/회색 텍스트, 여백 충분히, 장식 최소화, sans-serif 폰트(맑은 고딕/Arial), 깔끔한 선 구분. 색상은 #333333(제목), #666666(본문), #E0E0E0(구분선) 위주.',
  corporate: '기업 프레젠테이션: 네이비(#1B2A4A) 상단 바 + 화이트 배경, 포인트 색상 #2563EB, 각 슬라이드에 상단 컬러 스트라이프, 표와 차트 적극 활용, 깔끔하고 전문적인 톤.',
  colorful: '컬러풀 디자인: 슬라이드마다 다양한 밝은 배경 그라데이션, 색상 팔레트(#FF6B6B, #4ECDC4, #45B7D1, #FFA07A, #98D8C8, #F7DC6F), 둥근 도형과 카드형 레이아웃, 생동감 있는 구성.',
  dark: '다크 테마: 어두운 배경(#1A1A2E 또는 #0F0F1F), 밝은 텍스트(#FFFFFF, #E0E0E0), 네온 포인트 컬러(#6366F1 메인, #A855F7 보조, #22D3EE 강조), 고급스럽고 세련된 느낌.',
  academic: '학술 발표: 깔끔한 흰 배경, 보수적 배색(#2C3E50 제목, #34495E 본문), 번호 매긴 섹션 구조, 참고문헌 슬라이드 포함, 데이터는 차트/표 중심으로 시각화, 최소 장식.',
  startup: '스타트업 피치덱: 대담한 헤드라인(큰 폰트), 핵심 숫자 크게 강조, 카드형 레이아웃, 모던 색상(#6366F1 메인, #EC4899 보조), 마지막에 CTA(Call-to-Action) 슬라이드 포함.',
};
var activePptStyle = '';
var pptRefDesign = '';

function applyPptStyle(val){
  if(val === 'ref'){
    var inp = document.createElement('input');
    inp.type='file'; inp.accept='.pptx';
    inp.onchange = function(){ if(inp.files[0]) uploadPptRef(inp.files[0]); };
    inp.click();
    document.getElementById('pptStyleDrop').value = activePptStyle;
    return;
  }
  activePptStyle = val;
  pptRefDesign = '';
  document.getElementById('pptRefBadge').style.display = 'none';
}

async function uploadPptRef(file){
  var fd = new FormData(); fd.append('file', file);
  try{
    var resp = await fetch('/api/analyze_ppt_style', {method:'POST', body:fd});
    var data = await resp.json();
    if(data.error){
      addMsg('assistant', '❌ PPT 스타일 분석 실패: ' + data.error);
      return;
    }
    if(data.design){
      pptRefDesign = data.design;
      activePptStyle = '';
      document.getElementById('pptStyleDrop').value = '';
      document.getElementById('pptRefBadge').style.display = 'inline-flex';
      addMsg('assistant', '📎 참고 PPT 스타일 분석 완료: **' + file.name + '**\n' + data.summary + '\n\n새 PPT 생성 시 이 스타일을 반영합니다.');
    }
  }catch(e){
    addMsg('assistant', '❌ PPT 스타일 분석 중 오류 발생');
  }
}

function checkPptStyleVisibility(){
  var drop = document.getElementById('pptStyleDrop');
  if(!drop) return;
  var text = (document.getElementById('input').value || '').toLowerCase();
  var isPpt = /ppt|피피티|파워포인트|프레젠테이션|슬라이드|발표자료/.test(text);
  var hasPptxSkill = selSkills.includes('pptx') || autoLoadedSkills.includes('pptx');
  drop.style.display = (isPpt || hasPptxSkill) ? 'inline-block' : 'none';
}

/* ── 하네스 저장 세션 (기존 팝업에 통합) ── */
function openHarnessSessionTab(){
  const overlay=document.getElementById('sessionPopupOverlay');
  overlay.style.display='flex';
  /* 하네스 탭 내용을 팝업 body에 로드 */
  const body=document.getElementById('sessionPopupBody');
  if(!body)return;
  body.innerHTML='<div style="text-align:center;color:#999;padding:20px">하네스 저장 세션 불러오는 중...</div>';
  fetch('/api/harness/session/list')
    .then(r=>r.json())
    .then(data=>{
      if(!data.sessions||data.sessions.length===0){
        body.innerHTML='<div style="text-align:center;color:#999;padding:30px">하네스 저장 세션이 없습니다.</div>';
        return;
      }
      let html='<div style="padding:4px 0;font-size:12px;color:#6366f1;font-weight:600;margin-bottom:8px;">💾 하네스 자동 저장 세션 ('+data.sessions.length+'개)</div>';
      data.sessions.forEach(s=>{
        const d=new Date(s.timestamp*1000);
        const dateStr=d.toLocaleDateString('ko-KR',{month:'short',day:'numeric'})+' '+d.toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'});
        const skills=s.skills_used&&s.skills_used.length>0?s.skills_used.slice(0,2).join(', '):'';
        const preview=s.first_message||'';
        const previewText=preview.length>40?preview.substring(0,40)+'...':preview;
        html+=`<div style="padding:6px 10px;border:1px solid #e5e3de;border-radius:8px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;gap:8px">
          <div style="flex:1;min-width:0;display:flex;align-items:center;gap:8px">
            <span style="font-size:11px;color:#999;flex-shrink:0">${dateStr}</span>
            <span style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${previewText||skills||s.message_count+'개 메시지'}</span>
          </div>
          <div style="display:flex;gap:4px;flex-shrink:0">
            <button onclick="harnessRestore('${s.session_id}')" style="padding:2px 8px;background:#6366f1;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px">복원</button>
            <button onclick="harnessDelete('${s.session_id}')" style="padding:2px 6px;color:#ef4444;border:1px solid #fca5a5;background:none;border-radius:4px;cursor:pointer;font-size:11px">x</button>
          </div>
        </div>`;
      });
      body.innerHTML=html;
    })
    .catch(()=>{body.innerHTML='<div style="color:#ef4444;text-align:center;padding:20px">하네스 브릿지 미연결</div>';});
}

function harnessRestore(sid){
  fetch('/api/harness/session/load/'+sid).then(r=>r.json()).then(data=>{
    if(!data.messages||!data.messages.length){alert('복원할 메시지 없음');return;}
    const ct=document.querySelector('.content-inner')||document.querySelector('.content');
    if(!ct)return;
    ct.querySelectorAll('.msg').forEach(m=>m.remove());
    data.messages.forEach(msg=>{
      const div=document.createElement('div');
      div.className='msg '+(msg.role==='user'?'user':'assistant');
      div.innerHTML='<div class="msg-text">'+msg.content.replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</div>';
      ct.appendChild(div);
    });
    ct.scrollTop=ct.scrollHeight;
    closeSessionPopup();
  }).catch(()=>alert('복원 실패'));
}

function harnessDelete(sid){
  if(!confirm('삭제할까요?'))return;
  fetch('/api/harness/session/delete/'+sid,{method:'DELETE'}).then(()=>openHarnessSessionTab()).catch(()=>alert('삭제 실패'));
}

// ═══ 📑 PPT 설계 팝업 (MD → PPT 결정론적 변환, LLM 미사용) ═══
function openPptBuilder(){ document.getElementById('pptBuilderOverlay').style.display='flex'; }
function closePptBuilder(){ document.getElementById('pptBuilderOverlay').style.display='none'; }
function pptSwitchTab(tab){
  const mdTab=document.getElementById('pptTabMd'), fileTab=document.getElementById('pptTabFile');
  const mdC=document.getElementById('pptTabMdContent'), fileC=document.getElementById('pptTabFileContent');
  if(tab==='md'){ mdTab.style.background='#eab308';mdTab.style.color='#fff';fileTab.style.background='#e5e7eb';fileTab.style.color='#6b7280';mdC.style.display='flex';fileC.style.display='none'; }
  else { fileTab.style.background='#eab308';fileTab.style.color='#fff';mdTab.style.background='#e5e7eb';mdTab.style.color='#6b7280';fileC.style.display='flex';mdC.style.display='none'; }
}
function pptShowLoading(msg){
  document.getElementById('pptResultPanel').innerHTML =
    `<div style="text-align:center;padding-top:80px;color:#eab308;"><div style="font-size:36px;margin-bottom:10px;">⏳</div>${msg||'변환 중...'}</div>`;
}
function pptShowResult(data){
  const panel=document.getElementById('pptResultPanel');
  if(data.error){ panel.innerHTML=`<div style="color:#ef4444;"><b>❌ 실패</b><br><br>${data.error}</div>`; return; }
  const slides=(data.slides_summary||[]).map((s,i)=>
    `<div style="padding:6px 10px;background:#f1f5f9;border-radius:4px;margin-bottom:4px;display:flex;gap:8px;font-size:11px;">
       <span style="color:#eab308;font-weight:600;">Slide ${i+1}</span>
       <span style="color:#64748b;">[${s.type}]</span>
       <span style="flex:1;color:#334155;">${(s.title||'(제목 없음)').replace(/</g,'&lt;')}</span>
     </div>`).join('');
  panel.innerHTML = `
    <div style="margin-bottom:14px;">
      <div style="font-size:14px;font-weight:700;color:#16a34a;margin-bottom:6px;">✅ 변환 완료</div>
      <div style="font-size:11px;color:#64748b;">파일명: ${data.filename||''}</div>
      <div style="font-size:11px;color:#64748b;">슬라이드: ${data.slide_count||0}개 · 크기 ${((data.size||0)/1024).toFixed(1)} KB</div>
    </div>
    <a href="${data.download_url}" download
       style="display:block;padding:10px;background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;text-decoration:none;border-radius:6px;font-weight:700;text-align:center;margin-bottom:14px;font-size:13px;">💾 .pptx 다운로드</a>
    <div style="font-size:11px;color:#64748b;margin-bottom:6px;">슬라이드 구성</div>${slides}`;
}
async function pptGenerateFromMd(){
  const md=document.getElementById('pptMdInput').value.trim();
  if(!md){ alert('MD 내용을 입력해주세요'); return; }
  pptShowLoading('MD → PPT 변환 중...');
  try{
    const r=await fetch('/api/ppt/from-md',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({md})});
    pptShowResult(await r.json());
  }catch(e){ pptShowResult({error:'네트워크 오류: '+e.message}); }
}
async function pptGenerateFromFile(file){
  if(!file) return;
  pptShowLoading('파일 업로드 중... ('+file.name+')');
  const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/ppt/from-md-file',{method:'POST',body:fd});
    pptShowResult(await r.json());
  }catch(e){ pptShowResult({error:'네트워크 오류: '+e.message}); }
}
// 드롭존 이벤트
(function(){
  const zone=document.getElementById('pptDropZone'); if(!zone) return;
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.style.borderColor='#eab308';zone.style.background='#fefce8';});
  zone.addEventListener('dragleave',e=>{zone.style.borderColor='#cbd5e1';zone.style.background='#f8fafc';});
  zone.addEventListener('drop',e=>{
    e.preventDefault();
    zone.style.borderColor='#cbd5e1';zone.style.background='#f8fafc';
    const f=e.dataTransfer.files && e.dataTransfer.files[0];
    if(f) pptGenerateFromFile(f);
  });
})();
</script>

<!-- 📑 PPT 설계 플로팅 버튼 (로그인 후에만 노출) -->
<button id="pptBuilderBtn" onclick="openPptBuilder()"
        title=".md 파일/텍스트 → .pptx 자동 변환 (LLM 미사용, 결정론적)"
        style="position:fixed;left:20px;bottom:20px;z-index:9998;padding:12px 18px;
               background:linear-gradient(135deg,#eab308,#d97706);color:#fff;border:none;
               border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;
               box-shadow:0 4px 14px rgba(234,179,8,0.4);display:none;align-items:center;gap:6px;">
  📑 PPT 설계
</button>

<!-- 📑 PPT 설계 팝업 -->
<div id="pptBuilderOverlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,0.5);z-index:10000;justify-content:center;align-items:center;"
     onclick="if(event.target===this)closePptBuilder()">
  <div style="width:min(1100px,94vw);height:min(780px,90vh);background:#fff;border:1px solid #cbd5e1;
       border-radius:12px;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,0.3);">
    <div style="padding:14px 20px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;
         justify-content:space-between;background:linear-gradient(135deg,#fefce8,#fef3c7);border-radius:12px 12px 0 0;">
      <div>
        <div style="font-size:16px;font-weight:700;color:#a16207;">📑 PPT 설계</div>
        <div style="font-size:11px;color:#78716c;margin-top:2px;">
          MD 파일 드롭 · 텍스트 붙여넣기 → .pptx 자동 변환 (LLM 안 거침, 결정론적)
        </div>
      </div>
      <button onclick="closePptBuilder()" style="background:transparent;border:none;color:#78716c;font-size:22px;cursor:pointer;padding:4px 10px;">✕</button>
    </div>
    <div style="flex:1;display:flex;overflow:hidden;">
      <div style="flex:1;border-right:1px solid #e2e8f0;display:flex;flex-direction:column;padding:14px;">
        <div style="display:flex;gap:8px;margin-bottom:10px;">
          <button id="pptTabMd" onclick="pptSwitchTab('md')"
                  style="flex:1;padding:8px;background:#eab308;color:#fff;border:none;border-radius:6px;cursor:pointer;font-weight:700;font-size:12px;">📝 MD 텍스트</button>
          <button id="pptTabFile" onclick="pptSwitchTab('file')"
                  style="flex:1;padding:8px;background:#e5e7eb;color:#6b7280;border:none;border-radius:6px;cursor:pointer;font-weight:700;font-size:12px;">📎 파일 업로드</button>
        </div>
        <div id="pptTabMdContent" style="flex:1;display:flex;flex-direction:column;">
          <label style="font-size:11px;color:#64748b;margin-bottom:4px;">
            마크다운 입력 (# 제목 / ## 슬라이드 / - 불릿 / | 표 | / ```코드``` 지원)
          </label>
          <textarea id="pptMdInput"
            style="flex:1;background:#f8fafc;color:#1e293b;border:1px solid #cbd5e1;border-radius:6px;
                   padding:10px;font-family:Consolas,monospace;font-size:12px;resize:none;min-height:300px;"
            placeholder="# 발표 제목&#10;> 부제 또는 날짜&#10;&#10;## 첫 슬라이드&#10;- 요점 1&#10;- 요점 2&#10;&#10;## 표 슬라이드&#10;| 항목 | 값 |&#10;|---|---|&#10;| A | 1 |&#10;&#10;## 코드 슬라이드&#10;```python&#10;def hello():&#10;    print('hi')&#10;```"></textarea>
          <button onclick="pptGenerateFromMd()"
                  style="margin-top:10px;padding:10px;background:linear-gradient(135deg,#eab308,#d97706);
                         color:#fff;border:none;border-radius:6px;font-weight:700;cursor:pointer;font-size:13px;">🚀 PPT 만들기</button>
        </div>
        <div id="pptTabFileContent" style="flex:1;display:none;flex-direction:column;">
          <label style="font-size:11px;color:#64748b;margin-bottom:4px;">.md / .markdown / .txt 파일 드래그-드롭 또는 클릭</label>
          <div id="pptDropZone"
               style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
                      border:2px dashed #cbd5e1;border-radius:8px;cursor:pointer;padding:40px;
                      transition:all .2s;background:#f8fafc;min-height:300px;"
               onclick="document.getElementById('pptFileInput').click()">
            <div style="font-size:48px;margin-bottom:12px;">📎</div>
            <div style="color:#a16207;font-size:14px;font-weight:700;">파일 드롭 또는 클릭</div>
            <div style="color:#78716c;font-size:11px;margin-top:6px;">.md · .markdown · .txt (UTF-8 / CP949 자동 감지)</div>
          </div>
          <input type="file" id="pptFileInput" accept=".md,.markdown,.txt" style="display:none;"
                 onchange="pptGenerateFromFile(this.files[0])"/>
        </div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;padding:14px;overflow-y:auto;">
        <label style="font-size:11px;color:#64748b;margin-bottom:4px;">결과</label>
        <div id="pptResultPanel"
             style="flex:1;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                    padding:16px;color:#1e293b;font-size:12px;overflow-y:auto;min-height:300px;">
          <div style="text-align:center;color:#94a3b8;padding-top:80px;">
            <div style="font-size:36px;margin-bottom:10px;">⬅️</div>
            왼쪽에서 MD 입력 또는 파일 업로드 후<br>결과가 여기 나옵니다
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

</body>
</html>
"""

# ============================================
# 실행
# ============================================
