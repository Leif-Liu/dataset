from __future__ import annotations

from pathlib import Path

from flows.requirement_to_testcases import RequirementToTestcaseFlow, write_output
from llms.factory import init_llamaindex_settings
from settings import load_config, resolve_paths


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    config_path = root_dir / "configs" / "system.yaml"

    config = load_config(str(config_path))
    config = resolve_paths(config, root_dir)

    init_llamaindex_settings(config)

    flow = RequirementToTestcaseFlow(config)
    result = flow.run()

    output_path = write_output(config["system"]["output_dir"], result)
    print(result)
    print(f"\nSaved test cases to: {output_path}")


if __name__ == "__main__":
    main()

