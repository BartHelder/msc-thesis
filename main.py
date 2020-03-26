from train import train
import itertools
import multiprocessing as mp
import json

from params import env_params_train, ac_params_train, rls_params, pid_params

num_seeds = 40

taus = (0.01, 0.1, 1.0)
nn_stdevs = (0.05, 0.1, 0.2)
gammas = (0.7, 0.8, 0.9, 0.95, 0.99)
lrs_act = (3, 5, 7)
lrs_crit = (3, 5, 7)


def mp_learn(args):
    ac_params = ac_params_train.copy()
    settings = list(itertools.product(*args))

    for setting in settings:

        ac_params['lon']['tau_target_critic'] = setting[0]
        ac_params['lon']['nn_stdev_actor'] = setting[1]
        ac_params['lon']['nn_stdev_critic'] = setting[1]
        ac_params['lon']['discount_factor'] = setting[2]
        ac_params['lon']["learning_rate_actor"] = setting[3]
        ac_params['lon']['learning_rate_critic'] = setting[4]

        with mp.Pool(processes=10) as pool:
            results = [pool.apply_async(train, args=("train", env_params_train, ac_params_train, rls_params, pid_params, "", seed, 10, False, False, False, False, False, False, False, False))
                       for seed in range(num_seeds)]
            results = [p.get() for p in results]
            succesful_trials = [x[0] for x in results]
            average_rms_final_10 = [x[1] for x in results]
            print("Succesful trials: ", sum(succesful_trials), "/", num_seeds)

            final = {'success': succesful_trials,
                     'average_rms': average_rms_final_10}

            with open('results/mar/26/json/'+str(setting)+'.json', 'w') as f:
                json.dump(final, f)

            pool.close()
            pool.join()


mp_learn([taus, nn_stdevs, gammas, lrs_act, lrs_crit])
