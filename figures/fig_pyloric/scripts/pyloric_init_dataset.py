import numpy as np

from tabpfn_sbi.tasks.pyloric import PyloricTask, results_to_ss_array

task = PyloricTask()


prior = task.get_prior_dist()


simulator = task.get_simulator(device="cuda", voltage_noise=0.5)


thetas = []
ss_stats = []
for i in range(50):
    print(i)
    np.random.seed(i)
    prior_samples = prior.sample((1_000,))
    ss = simulator(prior_samples)
    ss_stats.append(np.array(ss))
    thetas.append(np.array(prior_samples))


ss_stats = np.concatenate(ss_stats, axis=0)
thetas = np.concatenate(thetas, axis=0)

np.save("pyloric_ss_stats.npy", ss_stats)
np.save("pyloric_thetas.npy", thetas)
