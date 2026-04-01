import os
import sys


def _ensure_repo_root_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    _ensure_repo_root_on_path()
    from minimal_inference.s2v_streaming_interact import main as _main

    _main()


if __name__ == "__main__":
    main()

