from dwave.system import FixedEmbeddingComposite, DWaveSampler
import numpy as np
import pickle

# ----------------------------
# Configuration
# ----------------------------

# Uncomment and set these variables with your actual paths and token
# LOCAL_DIR = "your_local_directory_to_save_results"  # Replace with your desired path
TOKEN = "your_dwave_token"  # Replace with your D-Wave API token

HAMILTONIAN_FILE = "data_20.pkl"
RESULTS_FILE = "results.pkl"  # Saves results in the current directory

# ----------------------------
# Parameters
# ----------------------------
NUM_READS = 1000
ANNEALING_TIME = 0.05  # Fast simulation regime

# ----------------------------
# Load Embedding Data
# ----------------------------
data = []
hamiltonian_path = HAMILTONIAN_FILE  # Since you're already inside 'data_files'

try:
    with open(hamiltonian_path, 'rb') as file:
        while True:
            data.append(pickle.load(file))
except EOFError:
    pass
except FileNotFoundError:
    print(f"Error: The file '{hamiltonian_path}' was not found.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading '{hamiltonian_path}': {e}")
    exit(1)

if not data:
    print("Error: No data loaded. Please check the pickle file.")
    exit(1)

# ----------------------------
# Initialize D-Wave Sampler
# ----------------------------
try:
    dwave_hw = data[0]['solver']
    qpu_sampler = DWaveSampler(solver=dwave_hw, token=TOKEN)
except KeyError:
    print("Error: 'solver' key not found in the first data instance.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while initializing the sampler: {e}")
    exit(1)

# ----------------------------
# Sampling Process
# ----------------------------
for idx, instance in enumerate(data, start=1):
    try:
        embedding = instance["embedding"]
        J = instance["J"]
        h = instance["h"]
    except KeyError as ke:
        print(f"Error: Missing key {ke} in data instance {idx}. Skipping this instance.")
        continue
    except Exception as e:
        print(f"An unexpected error occurred while accessing data instance {idx}: {e}")
        continue

    try:
        sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
        sampleset = sampler.sample_ising(
            h={}, 
            J=J,
            num_reads=NUM_READS,
            annealing_time=ANNEALING_TIME,
            fast_anneal=True,
            auto_scale=False,
            answer_mode="raw",
            flux_biases=h,
            chain_strength=2
        )
        
        # Aggregate samples
        aggregated_samples = sampleset.aggregate()
        
        # Save results
        result = {'sampleset': aggregated_samples}
        with open(RESULTS_FILE, 'ab') as result_file:
            pickle.dump(result, result_file)
        
        print(f"Instance {idx}: Sampling and saving completed successfully.")
        
    except Exception as e:
        print(f"An error occurred during sampling for instance {idx}: {e}")
        continue

print("All instances have been processed.")
