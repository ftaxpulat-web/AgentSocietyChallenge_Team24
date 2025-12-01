import os
import json
import argparse
import datetime
import numpy as np
from websocietysimulator import Simulator

# Import from your new package
from team24_agent.agent import MySimulationAgent
from team24_agent.llm import GeminiLLM
from team24_agent.config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description="Run Agent Society Experiments")
    parser.add_argument("--exp_name", type=str, default="test_run")
    parser.add_argument("--memory", type=str, default="none", choices=["none", "dilu", "global_rag"])
    parser.add_argument("--reflection", action="store_true")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    # Setup Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    
    # Task Paths
    TASK_SET = "amazon"
    TASK_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "tasks")
    GT_DIR = os.path.join(BASE_DIR, "example", "track1", TASK_SET, "groundtruth")
    
    # Output Paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", args.exp_name, timestamp)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Config Object
    config = ExperimentConfig(
        exp_name=args.exp_name,
        memory_type=args.memory,
        use_reflection=args.reflection,
        task_count=args.tasks,
        model_name=args.model,
        base_dir=BASE_DIR,
        data_dir=DATA_DIR,
        global_db_path=os.path.join(BASE_DIR, "global_chroma_db")
    )

    # Save Config
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    print(f"\n>>> EXPERIMENT: {args.exp_name}")
    print(f">>> MEMORY: {args.memory}")
    print(f">>> REFLECTION: {args.reflection}")

    try:
        # Initialize Simulator
        simulator = Simulator(data_dir=DATA_DIR, device="auto", cache=True)
        simulator.set_task_and_groundtruth(task_dir=TASK_DIR, groundtruth_dir=GT_DIR)

        # Inject Configured Agent
        class ConfiguredAgent(MySimulationAgent):
            def __init__(self, llm):
                super().__init__(llm, config=config)

        simulator.set_agent(ConfiguredAgent)
        simulator.set_llm(GeminiLLM(model=args.model))

        # Run
        print(f"Running {args.tasks} tasks...")
        outputs = simulator.run_simulation(number_of_tasks=args.tasks, enable_threading=False)

        # Evaluate
        print("Calculating metrics...")
        eval_results = simulator.evaluate()

        # Calculate RMSE & Compile Logs
        gt_subset = simulator.groundtruth_data[:len(outputs)]
        stars_pred, stars_real = [], []
        detailed_logs = []

        for i, (agent_res, gt_res) in enumerate(zip(outputs, gt_subset)):
            if not agent_res or 'output' not in agent_res: continue
            
            p_stars = agent_res['output'].get('stars', 0.0)
            r_stars = gt_res.get('stars', 0.0)
            stars_pred.append(p_stars)
            stars_real.append(r_stars)
            
            detailed_logs.append({
                "task_id": i,
                "user_id": agent_res['task']['user_id'],
                "item_id": agent_res['task']['item_id'],
                "real_stars": r_stars,
                "pred_stars": p_stars,
                "real_review": gt_res.get('review', ''),
                "pred_review": agent_res['output'].get('review', ''),
                "rag_context": agent_res['output'].get('rag_context', 'N/A'),
                "persona": agent_res['output'].get('persona', 'N/A')
            })

        if stars_pred:
            mse = np.mean((np.array(stars_pred) - np.array(stars_real)) ** 2)
            rmse = np.sqrt(mse)
            eval_results['metrics']['rmse'] = float(rmse)
            print(f">>> RMSE: {rmse:.4f}")

        # Save Results
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        with open(os.path.join(OUTPUT_DIR, "detailed_logs.json"), "w") as f:
            json.dump(detailed_logs, f, indent=4)

        print(f">>> DONE. Results: {OUTPUT_DIR}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()