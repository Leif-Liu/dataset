from __future__ import annotations

from pathlib import Path

from flows.requirement_to_testcases import RequirementToTestcaseFlow, write_output
from llms.factory import init_models
from settings import load_config, load_env_file, resolve_paths


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    config_path = root_dir / "configs" / "system.yaml"
    load_env_file(root_dir / ".env")

    config = load_config(str(config_path))
    config = resolve_paths(config, root_dir)

    llm, embeddings = init_models(config)

    flow = RequirementToTestcaseFlow(config, llm, embeddings)
    result = flow.run()

    output_path = write_output(config["system"]["output_dir"], result)
    print(result)
    print(f"\nSaved test cases to: {output_path}")


if __name__ == "__main__":
    main()

