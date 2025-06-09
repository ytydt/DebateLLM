# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A simple Python replacement for the shell entry scripts.

This module allows running evaluation from Python, which makes IDE debugging
simpler. It mirrors the behaviour of the various `eval_*.sh` scripts.
"""

from __future__ import annotations

import argparse
import subprocess

DEFAULT_DATASETS = "mmlu,pubmedqa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument(
        "--system", default="gpt", help="System name to evaluate, e.g. 'gpt'"
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help="Comma separated list of datasets",
    )
    parser.add_argument(
        "--multirun",
        action="store_true",
        help="Enable Hydra multirun mode",
    )
    parser.add_argument("extras", nargs=argparse.REMAINDER, help="Extra arguments")

    args = parser.parse_args()

    cmd = ["python", "experiments/evaluate.py"]
    if args.multirun:
        cmd.append("--multirun")
    cmd.append(f"system={args.system}")
    if args.datasets:
        cmd.append(f"dataset={args.datasets}")
    cmd.extend(args.extras)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
