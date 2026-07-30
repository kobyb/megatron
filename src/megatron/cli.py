import argparse

from .config import Config


def main():
    p = argparse.ArgumentParser(prog="megatron", description="Real-time local fact-checker")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="listen + fact-check live")
    run_p.add_argument("--device", help='listen ONLY to this source: "system", "mic", or an input '
                                        'device name (default: system + mic in parallel)')
    run_p.add_argument("--no-mic", action="store_true", help="system audio only")
    run_p.add_argument("--port", type=int)
    run_p.add_argument("--no-web", action="store_true", help="disable web search fallback")

    idx_p = sub.add_parser("index", help="index repos/docs for evidence retrieval")
    idx_p.add_argument("paths", nargs="+")

    sub.add_parser("devices", help="list audio devices")

    args = p.parse_args()
    cfg = Config()

    if args.cmd == "devices":
        from .audio import list_devices
        print(list_devices())
    elif args.cmd == "index":
        from .indexer import index
        index(cfg, args.paths)
    else:
        if args.device:
            cfg.audio_devices = [args.device]
        if args.no_mic:
            cfg.audio_devices = [d for d in cfg.audio_devices if d != "mic"]
        if args.port:
            cfg.port = args.port
        if args.no_web:
            cfg.web_search = False
        from .server import run
        run(cfg)
