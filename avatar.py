#!/usr/bin/env python3
"""
Avatar Agent — CLI entry point.

Usage:
    python avatar.py plan "Call Alex on Teams to schedule a meeting"
    python avatar.py plan "Call Alex on Teams" --dry-run
    python avatar.py health
    python avatar.py version
"""
import argparse
import asyncio
import json
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add engine directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))


def _load_engine():
    """Load config and logging. Returns (config, log)."""
    from engine.config import load_config
    from engine.logging_config import setup_logging, get_logger

    config_path = os.environ.get("AVATAR_CONFIG", "engine/config.yaml")
    config = load_config(config_path)
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        max_size_mb=config.logging.max_size_mb,
        backup_count=config.logging.backup_count,
    )
    log = get_logger("cli")
    return config, log


async def cmd_plan(instruction: str, dry_run: bool = False, config_path: str = None) -> int:
    """Execute the 'plan' subcommand."""
    from engine.config import load_config
    from engine.logging_config import setup_logging, get_logger
    from engine.modules.llm.client import create_llm_client
    from engine.modules.browser.registry import list_supported_apps

    cfg_path = config_path or os.environ.get("AVATAR_CONFIG", "engine/config.yaml")
    config = load_config(cfg_path)
    setup_logging(level=config.logging.level)
    log = get_logger("cli")

    log.info("plan_requested", instruction=instruction)
    print(f"\n📋 Instruction: {instruction}\n")

    llm = create_llm_client(config.llm)
    available_apps = list_supported_apps()

    try:
        print("🤔 Generating action plan...")
        plan = await llm.generate_plan(instruction, available_apps)
    except Exception as e:
        print(f"\n❌ Failed to generate plan: {e}", file=sys.stderr)
        log.error("plan_failed", error=str(e))
        return 1

    # Display the plan
    print(f"\n✅ Plan: {plan.mission_summary}")
    print(f"   Estimated duration: {plan.estimated_duration}")
    if plan.conversation_goal:
        print(f"   Goal: {plan.conversation_goal}")
    print(f"\n   Steps ({len(plan.steps)}):")
    for i, step in enumerate(plan.steps, 1):
        params_str = ", ".join(f"{k}={v!r}" for k, v in step.get("params", {}).items())
        print(f"   {i}. {step['action']}({params_str})")

    if dry_run:
        print("\n   [dry-run] Plan generated. Not executing.")
        return 0

    print("\nExecute this plan? [y/N] ", end="", flush=True)
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    if answer != "y":
        print("Aborted.")
        return 0

    # Execute the plan steps
    from engine.modules.browser.pool import BrowserPool
    from engine.modules.browser.interface import ActionStatus

    print("\n🚀 Executing plan...\n")
    async with BrowserPool(config.browser) as pool:
        preflight = await pool.run_preflight_check(config.browser.default_app)
        if preflight.status == ActionStatus.NEEDS_INTERVENTION:
            print(f"⚠️  {preflight.error}")
            print("   Please complete the action in the browser window, then press Enter to retry.")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return 1
            preflight = await pool.run_preflight_check(config.browser.default_app)
            if not preflight.succeeded:
                print(f"❌ Preflight failed: {preflight.error}")
                return 1

        for i, step in enumerate(plan.steps, 1):
            params_str = ", ".join(f"{k}={v!r}" for k, v in step.get("params", {}).items())
            print(f"   [{i}/{len(plan.steps)}] {step['action']}({params_str})... ", end="", flush=True)
            result = await pool.execute_step(step)
            if result.succeeded:
                print("✓")
            elif result.status == ActionStatus.NEEDS_INTERVENTION:
                print(f"\n⚠️  Intervention needed: {result.error}")
                if result.screenshot_path:
                    print(f"   Screenshot: {result.screenshot_path}")
                print("   Press Enter to retry, or Ctrl+C to abort.")
                try:
                    input()
                    result = await pool.execute_step(step)
                    if not result.succeeded:
                        print(f"❌ Step failed after retry: {result.error}")
                        return 1
                    print("   ✓ Recovered")
                except KeyboardInterrupt:
                    print("\nAborted.")
                    return 1
            else:
                print(f"\n❌ Step failed: {result.error}")
                return 1

    print("\n✅ Plan executed successfully.")
    return 0


async def cmd_speak(text: str, device: str = None, config_path: str = None) -> int:
    """Execute the 'speak' subcommand — Phase 2 TTS demo."""
    from engine.config import load_config
    from engine.logging_config import setup_logging
    from engine.modules.audio import create_tts
    from engine.modules.audio.virtual_devices import inject_audio, get_virtual_mic_device

    cfg_path = config_path or os.environ.get("AVATAR_CONFIG", "engine/config.yaml")
    config = load_config(cfg_path)
    setup_logging(level=config.logging.level)

    mic_device = device or get_virtual_mic_device(config.virtual_devices)

    print(f"🎙 Synthesizing: {text!r}")
    print(f"   TTS: {config.audio.tts.primary}  →  device: {mic_device}")

    tts = create_tts(config.audio)
    try:
        audio_stream = tts.synthesize(text)
        await inject_audio(audio_stream, device=mic_device)
        print("✅ Done.")
        return 0
    except Exception as e:
        print(f"\n❌ speak failed: {e}", file=sys.stderr)
        return 1
    finally:
        await tts.aclose()


async def cmd_health() -> int:
    """Execute the 'health' subcommand."""
    from engine.config import load_config
    cfg_path = os.environ.get("AVATAR_CONFIG", "engine/config.yaml")
    try:
        config = load_config(cfg_path)
        print(f"Config:       ✓ loaded from {cfg_path}")
        print(f"Server port:  {config.server.port}")
        print(f"LLM primary:  {config.llm.primary} ({config.llm.anthropic.model})")
        print(f"STT primary:  {config.audio.stt.primary}")
        print(f"TTS primary:  {config.audio.tts.primary}")
        print(f"LipSync:      {'enabled' if config.lipsync.enabled else 'disabled'}")
        print(f"Browser app:  {config.browser.default_app}")

        # Check API keys
        anthropic_key = config.llm.anthropic.api_key
        elevenlabs_key = config.audio.tts.elevenlabs.api_key
        print(f"ANTHROPIC_API_KEY: {'✓ set' if anthropic_key else '✗ not set'}")
        print(f"ELEVENLABS_API_KEY: {'✓ set' if elevenlabs_key else '✗ not set'}")
        return 0
    except Exception as e:
        print(f"❌ Health check failed: {e}", file=sys.stderr)
        return 1


async def cmd_selector_health(app: str = "teams", config_path: str = None) -> int:
    """Execute the 'selector-health' subcommand — smoke-test browser selectors."""
    from engine.config import load_config
    from engine.modules.browser.health_check import SelectorHealthChecker

    cfg_path = config_path or os.environ.get("AVATAR_CONFIG", "engine/config.yaml")
    try:
        config = load_config(cfg_path)
    except Exception as e:
        print(f"❌ Config load failed: {e}", file=sys.stderr)
        return 1

    checker = SelectorHealthChecker()
    print(f"Checking selectors for '{app}' at {getattr(config.browser.app_urls, app, '?')}...")
    report = await checker.check(app, config.browser)

    if report.error:
        print(f"❌ Error: {report.error}")
        return 1

    width = max(len(r.name) for r in report.results) if report.results else 12
    for r in report.results:
        status = "✓" if r.found else "✗"
        latency = f"{r.latency_ms}ms"
        err = f"  ({r.error})" if r.error else ""
        print(f"  {status} {r.name:<{width}}  {latency:>7}{err}")

    print(f"\n  {report.passed} passed, {report.failed} failed")
    return 0 if report.ok else 1


def cmd_version() -> int:
    """Execute the 'version' subcommand."""
    print("Avatar Agent v0.1.0")
    print("Phases 1-6 implemented")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avatar",
        description="AI Avatar Agent — automates voice calls on your behalf",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Config file path [default: engine/config.yaml]",
    )
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Generate and execute an action plan")
    plan_parser.add_argument("instruction", help="Natural language instruction")
    plan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )

    speak_parser = subparsers.add_parser(
        "speak",
        help="[Phase 2] Synthesize text and inject into virtual mic",
    )
    speak_parser.add_argument("text", help="Text to speak")
    speak_parser.add_argument(
        "--device",
        metavar="NAME",
        default=None,
        help="sounddevice output device name (overrides config)",
    )

    subparsers.add_parser("health", help="Show engine component status")
    subparsers.add_parser("version", help="Show version information")

    health_parser = subparsers.add_parser(
        "selector-health",
        help="[Phase 6] Smoke-test browser selectors for a platform app",
    )
    health_parser.add_argument(
        "--app",
        default="teams",
        choices=["teams", "slack", "discord"],
        help="App to check (default: teams)",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        return asyncio.run(cmd_plan(args.instruction, dry_run=args.dry_run, config_path=args.config))
    elif args.command == "speak":
        return asyncio.run(cmd_speak(args.text, device=args.device, config_path=args.config))
    elif args.command == "health":
        return asyncio.run(cmd_health())
    elif args.command == "version":
        return cmd_version()
    elif args.command == "selector-health":
        return asyncio.run(cmd_selector_health(app=args.app, config_path=args.config))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
