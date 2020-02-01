import os
import pickle
from record import Record
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__) + "/battle_results"
records_file = dirname + "/records.pickle"

records = pickle.load(open(records_file, "rb"))

brute_force_records = []
dwave_records = []
for record in records:
    if record.strategy == 'brute_force':
        brute_force_records += [record]
    else:
        dwave_records += [record]


successes = [record for record in dwave_records if not record.failure]
failures = [record for record in dwave_records if record.failure]


print('n_failures: {} out of a total: {}'.format(len(dwave_records) - len(successes), len(dwave_records)))

batch_sizes = sorted(list(set([r.batch_size for r in dwave_records])))
batch_numbers = sorted(list(set([r.n_batches for r in dwave_records])))
layers = sorted(list(set([r.layers for r in records])))

# ========================================= plot total running times =========================================== #

for size in batch_numbers:
    average_times = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.n_batches == size and not r.failure]
        times = [r.running_time for r in matching]
        avg_time = np.mean(times)
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label='{} Batches'.format(size))

avg_brute_force_times = []
for l in layers:
    matching = [r for r in brute_force_records if r.layers == l]
    times = [r.running_time for r in matching]
    avg_time = np.mean(times)
    avg_brute_force_times += [avg_time]
plt.plot(layers, avg_brute_force_times, 'x-', label='Brute force')

plt.yscale("log")
plt.xticks(layers)
plt.title('Total running time vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('average running time [s]')
plt.legend()
plt.show()

# ===================== plot failures ==========================================#

plt.figure()
for size in batch_numbers:
    failure_chances = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.n_batches == size]
        matching_failures = [r for r in failures if r.layers == l and r.n_batches == size]
        try:
            chance = 100.0 * len(matching_failures)/len(matching)
        except:
            chance = np.NaN
        failure_chances += [chance]
    plt.plot(layers, failure_chances, 'x-', label='{} Batches'.format(size))

plt.xticks(layers)
plt.title('Failure chance vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('probability [%]')
plt.legend()
plt.show()


# ======================================= plot QPU times ========================================== #

plt.figure()
for size in batch_numbers:
    average_times = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.n_batches == size]
        times = [r.timing.get('qpu_access_time', False) for r in matching]
        avg_time = np.mean([t for t in times if t])
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label='{} Batches'.format(size))

avg_brute_force_times = []
for l in layers:
    matching = [r for r in brute_force_records if r.layers == l]
    times = [r.running_time for r in matching]
    avg_time = np.mean(times)
    avg_brute_force_times += [avg_time]
plt.plot(layers, avg_brute_force_times, 'x-', label='Brute force')


plt.xticks(layers)
plt.title('Total QPU time vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('average running time [s]')
plt.legend()
plt.show()

# ========================================= plot total running times with embedding time =========================================== #

for size in batch_numbers:
    average_times = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.n_batches == size]
        times = [r.running_time + r.embedding_time for r in matching]
        avg_time = np.mean(times)
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label='{} Batches'.format(size))

avg_brute_force_times = []
for l in layers:
    matching = [r for r in brute_force_records if r.layers == l]
    times = [r.running_time for r in matching]
    avg_time = np.mean(times)
    avg_brute_force_times += [avg_time]
plt.plot(layers, avg_brute_force_times, 'x-', label='Brute force')

plt.yscale("log")
plt.xticks(layers)
plt.title('Total running time including embedding vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('average running time [s]')
plt.legend()
plt.show()
