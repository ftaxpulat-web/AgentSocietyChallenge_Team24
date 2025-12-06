# run_experiment.py

import os
import json
import argparse
import datetime
import numpy as np

from websocietysimulator import Simulator

from team24_agent.agent import Team24Agent
from team24_agent.llm import GeminiLLM
from team24_agent.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Run AgentSociety experiments with Team24Agent")
    parser.add_argument("--exp_name", type=str, default="test_run")

    parser.add_argument(
        "--memory",
        type=str,
        default="none",
        choices=["none", "dilu", "generative", "tp", "voyager", "hybrid_rag"],
        help="Memory module to use",
    )
    parser.add_argument(
        "--planning",
        type=str,
        default="io",
        choices=["io", "deps", "td", "voyager", "openagi", "hugginggpt"],
        help="Planning module to use",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="io",
        choices=["io", "cot", "cotsc", "tot", "dilu", "self_refine", "step_back"],
        help="Reasoning module to use",
    )

    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")

    args = parser.parse_args()

    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")

    TASK_SET = "yelp"  # or "amazon" or "goodreads"
    TASK_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "tasks")
    GT_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "groundtruth")

    # --- Output dirs ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", args.exp_name, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Config object ---
    config = ExperimentConfig(
        exp_name=args.exp_name,
        memory_type=args.memory,
        reasoning_type=args.reasoning,
        planning_type=args.planning,
        task_count=args.tasks,
        model_name=args.model,
        base_dir=BASE_DIR,
        data_dir=DATA_DIR,
    )

    # Save config
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    print(f"\n>>> EXPERIMENT: {args.exp_name}")
    print(f">>> MEMORY: {args.memory}")
    print(f">>> PLANNING: {args.planning}")
    print(f">>> REASONING: {args.reasoning}")
    print(f">>> MODEL: {args.model}")
    print(f">>> TASKS: {args.tasks}")

    try:
        # --- 1. Initialize simulator ---
        simulator = Simulator(data_dir=DATA_DIR, device="auto", cache=True)
        simulator.set_task_and_groundtruth(task_dir=TASK_DIR, groundtruth_dir=GT_DIR)

        # --- 2. Configure Agent class that uses this config ---
        class ConfiguredAgent(Team24Agent):
            def __init__(self, llm):
                super().__init__(llm=llm, config=config)

        simulator.set_agent(ConfiguredAgent)

        # --- 3. LLM ---
        simulator.set_llm(GeminiLLM(model=args.model))

        # --- 4. Run simulation ---
        print(f"Running {args.tasks} tasks...")
        outputs = simulator.run_simulation(
            number_of_tasks=args.tasks,
            enable_threading=False,  # simpler for debugging initially
        )

        # --- 5. Evaluate with built-in metrics ---
        print("Calculating metrics...")
        eval_results = simulator.evaluate()

        # --- 6. Compute RMSE over stars and build logs ---
        gt_subset = simulator.groundtruth_data[: len(outputs)]
        stars_pred, stars_real = [], []
        detailed_logs = []

        for i, (agent_res, gt_res) in enumerate(zip(outputs, gt_subset)):
            if not agent_res or "output" not in agent_res:
                continue

            out = agent_res["output"]
            p_stars = out.get("stars", 0.0)
            r_stars = gt_res.get("stars", 0.0)
            stars_pred.append(p_stars)
            stars_real.append(r_stars)

            detailed_logs.append(
                {
                    "task_id": i,
                    "user_id": agent_res["task"]["user_id"],
                    "item_id": agent_res["task"]["item_id"],
                    "real_stars": r_stars,
                    "pred_stars": p_stars,
                    "real_review": gt_res.get("review", ""),
                    "pred_review": out.get("review", ""),
                }
            )

        if stars_pred:
            mse = float(np.mean((np.array(stars_pred) - np.array(stars_real)) ** 2))
            rmse = float(np.sqrt(mse))
            eval_results.setdefault("metrics", {})
            eval_results["metrics"]["rmse"] = rmse
            print(f">>> RMSE: {rmse:.4f}")

        # --- 7. Save results ---
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=4)

        with open(os.path.join(OUTPUT_DIR, "detailed_logs.json"), "w") as f:
            json.dump(detailed_logs, f, indent=4)

        print(f">>> DONE. Results saved to: {OUTPUT_DIR}")

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
