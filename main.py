from dwave.system import FixedEmbeddingComposite, DWaveSampler
import numpy as np
import pickle

# Configuration
#LOCAL_DIR = "your_local_directory_to_save_results" # Replace with your path
#TOKEN = "your_dwave_token" # Replace with your token
HAMILTONIAN_FILE = "data_20.pkl"
EMBEDDING_DIR = "data_files"
RESULTS_FILE = f"{LOCAL_DIR}/results.pkl"

# Parameters
NUM_READS = 1000
ANNEALING_TIME = 0.05  # Fast simulation regime

# Load embedding data
data = []
hamiltonian_path = f"{EMBEDDING_DIR}/{HAMILTONIAN_FILE}"
with open(hamiltonian_path, 'rb') as file:
    try:
        while True:
            data.append(pickle.load(file))
    except EOFError:
        pass

# Initialize D-Wave sampler
dwave_hw = data[0]['solver']
qpu_sampler = DWaveSampler(solver=dwave_hw, token=TOKEN)

# Iterate through each instance and sample
for instance in data:
    embedding = instance["embedding"]
    J = instance["J"]
    h = instance["h"]
    
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
