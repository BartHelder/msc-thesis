from train import train
import itertools
import multiprocessing as mp
import json

import torch

from params import env_params, ac_params, rls_params, pid_params, path

training_logs,  = train(env_params=env_params,
                      ac_params=ac_params,
                      rls_params=rls_params,
                      pid_params=pid_params,
                      path=path,
                      seed=78,
                      plot_states=True,
                      save_weights=False,
                      plot_rls=False)


num_seeds = 50

taus = (0.01, 0.1, 1.0)
nn_stdevs = (0.1, 0.2, 0.3)
gammas = (0.7, 0.8, 0.9, 0.95)

def mp_learn(*args):

    settings = list(itertools.product(*args))
    print(settings)

    for setting in settings:

        ac_params['lon']['tau_target_critic'] = setting[0]
        ac_params['lon']['nn_stdev_actor'] = setting[1]
        ac_params['lon']['nn_stdev_critic'] = setting[1]
        ac_params['lon']['discount_factor'] = setting[2]

        with mp.Pool(processes=10) as pool:
            results = [pool.apply_async(train, args=(env_params, ac_params, rls_params, pid_params, "", seed, 10, False, False, False, False, False, False))
                       for seed in range(num_seeds)]
            results = [p.get() for p in results]
            succesful_trials = sum(x[0] for x in results)
            average_rms_final_10 = sum(x[1] for x in results) / succesful_trials
            print("Succesful trials: ", succesful_trials, "/", num_seeds)

            final = {'success_rate': succesful_trials/num_seeds,
                     'average_rms': average_rms_final_10}

            with open('results/mar/5/'+str(setting)+'.json', 'w') as f:
                json.dump(final, f)

            pool.close()
            pool.join()
