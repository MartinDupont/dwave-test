import os
import pickle
from record import Record
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================================== #
date_string = '2020_02_01_20:30'
# ==================================================================================================================== #

dirname = os.path.dirname(__file__) + "/battle_results_qbsolv"
records_file = dirname + "/records_{}.pickle".format(date_string)

records = pickle.load(open(records_file, "rb"))

brute_force_records = []
dwave_records = []
for record in records:
    if record.strategy == 'BRUTE_FORCE':
        brute_force_records += [record]
    else:
        dwave_records += [record]


successes = [record for record in dwave_records if not record.failure]
failures = [record for record in dwave_records if record.failure]


print('n_failures: {} out of a total: {}'.format(len(dwave_records) - len(successes), len(dwave_records)))

strategies = sorted(list(set([r.strategy for r in records])))
layers = sorted(list(set([r.layers for r in records])))

# ========================================= plot total running times =========================================== #

for s in strategies:
    average_times = []
    for l in layers:
        matching = [r for r in records if r.layers == l and r.strategy == s and not r.failure]
        times = [r.running_time for r in matching]
        avg_time = np.mean(times)
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label=s)

plt.yscale("log")
plt.xticks(layers)
plt.title('Total running time vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('average running time [s]')
plt.legend()
plt.show()

# ===================== plot failures ==========================================#

plt.figure()
for s in strategies:
    failure_chances = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.strategy == s]
        matching_failures = [r for r in failures if r.layers == l and r.strategy == s]
        try:
            chance = 100.0 * len(matching_failures)/len(matching)
        except:
            chance = np.NaN
        failure_chances += [chance]
    plt.plot(layers, failure_chances, 'x-', label=s)

plt.xticks(layers)
plt.title('Failure chance vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('probability [%]')
plt.legend()
plt.show()


# ======================================= plot QPU times ========================================== #

plt.figure()
for s in strategies:
    average_times = []
    for l in layers:
        matching = [r for r in dwave_records if r.layers == l and r.strategy == s]
        times = [r.timing.get('qpu_access_time', False) for r in matching]
        avg_time = np.mean([t for t in times if t])
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label=s)


plt.xticks(layers)
plt.title('Total QPU time vs problem size')
plt.xlabel('number of layers in problem')
plt.ylabel('average running time [s]')
plt.legend()
plt.show()
