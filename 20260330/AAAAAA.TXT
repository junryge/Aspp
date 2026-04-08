"""CLI entry point - main application loop with agent tool-use.

Usage:
    oh                          # Interactive mode, auto-detect model
    oh --model prod             # Use specific model
    oh run "fix the tests"      # Headless single-shot mode
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import glob as glob_mod
from pathlib import Path
from dataclasses import dataclass

from .api.token import load_token, print_token_status, find_token_file
from .api.provider import Provider, TOOL_DEFINITIONS
from .api.models import MODEL_REGISTRY, DEFAULT_MODEL
from .engine.registry import ToolRegistry
from .engine.router import ToolRouter
from .engine.loop import HarnessEngine, EngineConfig
from .tools.builtin import register_builtin_tools
from .skills.loader import SkillLoader
from .plugins.loader import PluginLoader
from .permissions import ToolPermissionContext
from .hooks import HookRegistry
from .commands.registry import CommandRegistry, Command, DEFAULT_COMMANDS, run_command

BANNER = """
  ==========================================
     OpenHarness  v0.1.0
     Open Agent Harness for Teams
  ==========================================
"""

PACKAGE_DIR = Path(__file__).parent.parent.parent  # openharness/ root

SYSTEM_PROMPT = """You are OpenHarness, an AI coding assistant running in the user's terminal.
You have access to tools to read files, write files, list directories, run commands, and search.
When the user asks you to do something with files or code, USE THE TOOLS to actually do it.
Do NOT just describe what you would do - actually do it by calling tools.
Always read files before analyzing them. Always run commands to execute code.
The current working directory is: {cwd}
Operating system: {os_name}
"""


@dataclass
class AppContext:
    """Shared context passed to commands and handlers."""
    provider: Provider
    registry: ToolRegistry
    router: ToolRouter
    engine: HarnessEngine
    skill_loader: SkillLoader
    plugin_loader: PluginLoader
    hook_registry: HookRegistry
    command_registry: CommandRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oh",
        description="OpenHarness - Open Agent Harness",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--trust", "-t",
        action="store_true",
        help="Auto-approve all tool permissions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Additional skills directory to scan",
    )
    parser.add_argument(
        "--plugins-dir",
        default=None,
        help="Additional plugins directory to scan",
    )

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a single prompt headlessly")
    run_parser.add_argument("prompt", help="The prompt to execute")
    run_parser.add_argument("--max-turns", type=int, default=5)

    sub.add_parser("status", help="Show system status")
    sub.add_parser("skills", help="List loaded skills")
    sub.add_parser("plugins", help="List loaded plugins")
    sub.add_parser("models", help="List available models")

    return parser


def initialize(args: argparse.Namespace) -> AppContext:
    """Initialize all subsystems and return app context."""
    token = load_token()
    model = args.model or DEFAULT_MODEL
    provider = Provider(token=token, default_model=model)

    registry = ToolRegistry()
    builtin_count = register_builtin_tools(registry)

    # Skills
    skill_search = []
    bundled_skills = PACKAGE_DIR / "skills" / "anthropic"
    if bundled_skills.is_dir():
        skill_search.append(bundled_skills)
    pkg_skills = PACKAGE_DIR / "skills"
    if pkg_skills.is_dir() and pkg_skills != bundled_skills:
        skill_search.append(pkg_skills)
    user_skills = Path.home() / ".openharness" / "skills"
    skill_search.append(user_skills)
    install_skills = Path("D:/openharness/skills/anthropic")
    if install_skills.is_dir() and install_skills not in skill_search:
        skill_search.insert(0, install_skills)
    if args.skills_dir:
        skill_search.append(Path(args.skills_dir))

    skill_loader = SkillLoader(search_paths=skill_search)
    skill_count = skill_loader.register_all(registry)

    # Plugins
    plugin_search = []
    bundled_plugins = PACKAGE_DIR / "plugins" / "anthropic"
    if bundled_plugins.is_dir():
        plugin_search.append(bundled_plugins)
    pkg_plugins = PACKAGE_DIR / "plugins"
    if pkg_plugins.is_dir() and pkg_plugins != bundled_plugins:
        plugin_search.append(pkg_plugins)
    user_plugins = Path.home() / ".openharness" / "plugins"
    plugin_search.append(user_plugins)
    install_plugins = Path("D:/openharness/plugins/anthropic")
    if install_plugins.is_dir() and install_plugins not in plugin_search:
        plugin_search.insert(0, install_plugins)
    if args.plugins_dir:
        plugin_search.append(Path(args.plugins_dir))

    plugin_loader = PluginLoader(search_paths=plugin_search)
    plugin_count = plugin_loader.register_all(registry)

    hook_registry = HookRegistry()
    engine_config = EngineConfig(structured_output=args.json)
    router = ToolRouter(registry)
    engine = HarnessEngine.create(registry, engine_config)

    command_registry = CommandRegistry()
    for cmd in DEFAULT_COMMANDS:
        command_registry.register(cmd)
    for skill_name in skill_loader.list_skills():
        meta = skill_loader.get_skill(skill_name)
        if meta:
            content = meta.content
            def make_skill_cmd(c: str):
                def handler(a: str, ctx: object) -> str:
                    return c[:3000]
                return handler
            command_registry.register(Command(
                name=skill_name,
                description=meta.description,
                handler=make_skill_cmd(content),
            ))
    for plugin_name in plugin_loader.list_plugins():
        plugin = plugin_loader.get_plugin(plugin_name)
        if plugin:
            for cmd in plugin.commands:
                content = cmd.content
                def make_plugin_cmd(c: str):
                    def handler(a: str, ctx: object) -> str:
                        return c[:3000]
                    return handler
                command_registry.register(Command(
                    name=cmd.name,
                    description=f"[{plugin_name}] {cmd.description}",
                    handler=make_plugin_cmd(content),
                ))

    ctx = AppContext(
        provider=provider,
        registry=registry,
        router=router,
        engine=engine,
        skill_loader=skill_loader,
        plugin_loader=plugin_loader,
        hook_registry=hook_registry,
        command_registry=command_registry,
    )
    return ctx


def print_startup(ctx: AppContext) -> None:
    """Print startup banner and status."""
    print(BANNER)
    ctx.provider.print_status()
    print(f"  Tools: {ctx.registry.count} ({ctx.skill_loader.count} skills, "
          f"{ctx.plugin_loader.count} plugins)")
    print(f"  Type /help for commands, /exit to quit\n")


# ── Tool Execution ──

def execute_tool_call(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if name == "read_file":
            path = arguments.get("path", "")
            p = Path(path).expanduser()
            if not p.is_file():
                return f"Error: file not found: {path}"
            content = p.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            if len(lines) > 500:
                return "\n".join(lines[:500]) + f"\n... ({len(lines)} total lines, truncated)"
            return content

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"

        elif name == "list_directory":
            path = arguments.get("path", ".")
            p = Path(path).expanduser()
            if not p.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(p.iterdir())
            lines = []
            for e in entries[:100]:
                prefix = "[DIR]  " if e.is_dir() else "[FILE] "
                lines.append(f"{prefix}{e.name}")
            if len(entries) > 100:
                lines.append(f"... ({len(entries)} total)")
            return "\n".join(lines) if lines else "(empty directory)"

        elif name == "run_command":
            command = arguments.get("command", "")
            if not command:
                return "Error: no command provided"
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=60, cwd=os.getcwd(),
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            return output[:10000] if output else "(no output)"

        elif name == "search_files":
            pattern = arguments.get("pattern", "")
            matches = sorted(glob_mod.glob(pattern, recursive=True))
            if not matches:
                return f"No files matching: {pattern}"
            result = "\n".join(matches[:50])
            if len(matches) > 50:
                result += f"\n... ({len(matches)} total)"
            return result

        elif name == "grep":
            import re
            pattern = arguments.get("pattern", "")
            search_path = arguments.get("path", ".")
            EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml",
                          ".js", ".ts", ".html", ".css", ".csv", ".xml"}
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                regex = re.compile(re.escape(pattern), re.IGNORECASE)
            root = Path(search_path).expanduser()
            matched = []
            for fpath in root.rglob("*"):
                if len(matched) >= 30:
                    break
                if not fpath.is_file() or fpath.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    if regex.search(text):
                        matched.append(str(fpath))
                except OSError:
                    continue
            if matched:
                return f"Found in {len(matched)} files:\n" + "\n".join(matched)
            return f"No matches for: {pattern}"

        else:
            return f"Unknown tool: {name}"

    except subprocess.TimeoutExpired:
        return "Error: command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def agent_loop(
    ctx: AppContext,
    messages: list[dict],
    max_turns: int = 10,
    print_output: bool = True,
) -> str:
    """Run the agent loop: send to LLM -> execute tool calls -> feed back -> repeat."""

    for turn in range(max_turns):
        response = ctx.provider.chat(
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        # If LLM returned text content, print it
        if response.content and print_output:
            print(f"\n{response.content}")

        # If no tool calls, we're done
        if not response.tool_calls:
            return response.content

        # Execute each tool call
        assistant_msg: dict = {"role": "assistant", "content": response.content or None}
        if response.tool_calls:
            assistant_msg["tool_calls"] = []
            for tc in response.tool_calls:
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    }
                })
        messages.append(assistant_msg)

        for tc in response.tool_calls:
            if print_output:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                print(f"\n  >> tool: {tc.name}({args_str})")

            result = execute_tool_call(tc.name, tc.arguments)

            if print_output:
                # Show truncated result
                preview = result[:300]
                if len(result) > 300:
                    preview += f"\n  ... ({len(result)} chars total)"
                print(f"  << {preview}")

            # Feed tool result back to LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Continue the loop - LLM will see tool results and decide next step

    if print_output:
        print("\n  (max turns reached)")
    return response.content


# ── Main Interactive Loop ──

def interactive_loop(ctx: AppContext) -> None:
    """Main interactive REPL loop with agent tool-use."""
    cwd = os.getcwd()
    os_name = "Windows" if os.name == "nt" else "Linux/Mac"
    system_msg = SYSTEM_PROMPT.format(cwd=cwd, os_name=os_name)

    # Persistent conversation history
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
    ]

    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        # Slash commands
        if prompt.startswith("/"):
            output = run_command(prompt, ctx, ctx.command_registry)
            if output:
                print(output)
            continue

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Run agent loop (LLM + tool execution)
        try:
            agent_loop(ctx, messages, max_turns=10, print_output=True)
        except Exception as e:
            print(f"\n  Error: {e}")
            # Remove failed message to keep history clean
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        print()


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    ctx = initialize(args)

    if args.command == "run":
        cwd = os.getcwd()
        os_name = "Windows" if os.name == "nt" else "Linux/Mac"
        system_msg = SYSTEM_PROMPT.format(cwd=cwd, os_name=os_name)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": args.prompt},
        ]
        try:
            agent_loop(ctx, messages, max_turns=args.max_turns, print_output=True)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "status":
        print(run_command("/status", ctx, ctx.command_registry))
        return
    if args.command == "skills":
        print(run_command("/skills", ctx, ctx.command_registry))
        return
    if args.command == "plugins":
        print(run_command("/plugins", ctx, ctx.command_registry))
        return
    if args.command == "models":
        print(run_command("/models", ctx, ctx.command_registry))
        return

    # Interactive mode
    print_startup(ctx)
    interactive_loop(ctx)
