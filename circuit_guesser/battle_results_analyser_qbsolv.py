import os
import pickle
from record import Record
import numpy as np
import matplotlib.pyplot as plt

# Starting to do tests with only one try per problem at 11:30 on 02/02.
# at 16:37 i burnt through all my computer time
# ==================================================================================================================== #
#date_string = '2020_02_01_23:32' # real dwave value
#date_string = '2020_02_02_11:34'

date_strings = [ '2020_02_02_11:34', '2020_02_02_12:43', '2020_02_02_13:05', '2020_02_02_14:24', '2020_02_02_18:42', '2020_02_03_22:26', '2020_02_05_22:09', '2020_02_05_22:48']
# ==================================================================================================================== #

dirname = os.path.dirname(os.path.abspath(__file__)) + "/battle_results_qbsolv"
records = []
for date_string in date_strings:
    records_file = dirname + "/records_{}.pickle".format(date_string)

    records += pickle.load(open(records_file, "rb"))

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
        matching = [r for r in records if r.layers == l and r.strategy == s]
        times = [r.running_time for r in matching]
        avg_time = np.mean(times)
        average_times += [avg_time]
    plt.plot(layers, average_times, 'x-', label=s)

plt.yscale("log")
plt.xticks(layers)
plt.title('Total running time vs problem size')
plt.xlabel('Number of layers in problem')
plt.ylabel('Average running time [s]')
plt.legend()
plt.savefig(dirname + '/total_times.png')
plt.show()

# ===================== plot failures ==========================================#

plt.figure()
for s in ['D_WAVE', 'SIMULATED_ANNEALING']:
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
plt.xlabel('Number of layers in problem')
plt.ylabel('Probability [%]')
plt.legend()
plt.savefig(dirname + '/failure_chance.png')
plt.show()


# ======================================= plot QPU times ========================================== #
dwave_scaling = 1000000

plt.figure()
for s in strategies:
    average_times = []
    for l in layers:
        matching = [r for r in records if r.layers == l and r.strategy == s]
        if s == 'BRUTE_FORCE':
            times = [r.running_time for r in matching]
            label = 'x--'
        else:
            times = [r.timing.get('qpu_access_time', False) for r in matching]
            label = 'x-'
        avg_time = np.mean([t for t in times if t])
        if s == 'D_WAVE':
            avg_time = avg_time/dwave_scaling
        average_times += [avg_time]
    plt.plot(layers, average_times, label, label=s)

# time_for_first = average_times[0]
# multiplier = 1.0 / time_for_first
# extrapolated_times = [ t * multiplier for t in average_times]
# plt.plot(layers, extrapolated_times, 'x-', label='extrapolated')
#
# integrated_time = np.sum(extrapolated_times)
# print('integrated average time for runs in range ({},{}) is: {}'.format(layers[0], layers[-1], integrated_time))
# print('you can perform {} runs.'.format(55.0/integrated_time))

plt.yscale("log")

plt.xticks(layers)
plt.title('Total QPU time vs problem size')
plt.xlabel('Number of layers in problem')
plt.ylabel('Average time [s]')
plt.legend()
plt.savefig(dirname + '/quantum_times.png')
plt.show()
