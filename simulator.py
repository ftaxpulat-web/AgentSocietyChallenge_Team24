from websocietysimulator import Simulator
from agent import MySimulationAgent

# Initialize Simulator
simulator = Simulator(data_dir="data", device="auto", cache=False)
# The cache parameter controls whether to use cache for interaction tool.
# If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
# If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

# Load scenarios
simulator.set_task_and_groundtruth(task_dir="path/to/task_directory", groundtruth_dir="path/to/groundtruth_directory")

# Set your custom agent
simulator.set_agent(MySimulationAgent)

# Set LLM client
simulator.set_llm(DeepSeekLLM(api_key="Your API Key"))

# Run evaluation
# If you don't set the number of tasks, the simulator will run all tasks.
agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

# Evaluate the agent
evaluation_results = simulator.evaluate()