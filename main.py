from train import train
import itertools
import multiprocessing as mp
import json

from params import env_params_train, env_params_test1, env_params_test2, ac_params_train, ac_params_test, rls_params, pid_params


def train_test(mode, env_params, ac_params, results_path, seed):
    agents_path = "saved_models/apr/10/" + str(seed) + "/"
    # if mode == "test_1":
    #     _, _ = train(mode="train",
    #                  env_params=env_params_train,
    #                  ac_params=ac_params_train,
    #                  rls_params=rls_params,
    #                  pid_params=pid_params,
    #                  results_path=results_path,
    #                  agents_path=agents_path,
    #                  seed=seed,
    #                  return_logs=False,
    #                  save_logs=False,
    #                  save_weights=False,
    #                  save_agents=True,
    #                  load_agents=False,
    #                  plot_states=False,
    #                  plot_nn_weights=False,
    #                  plot_rls=False)

    results = train(mode=mode,
                    env_params=env_params,
                    ac_params=ac_params,
                    rls_params=rls_params,
                    pid_params=pid_params,
                    results_path=results_path,
                    agents_path=agents_path,
                    seed=seed,
                    return_logs=False,
                    save_logs=True,
                    save_weights=False,
                    save_agents=False,
                    load_agents=True,
                    plot_states=False,
                    plot_nn_weights=False,
                    plot_rls=False)

    return results


def mp_train(settings, num_seeds=100):
    """
    Trains an agent for a variety of hyperparameter combinations for multiple seeds each.
    :param settings: list of hyperparameter combinations to test.
    :param num_seeds:

    Saves results of each hyperparameter combination to a separate json
    """
    ac_params = ac_params_train.copy()

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


def mp_test(mode, env_params, ac_params, num_seeds: int = 100):
    """
    Perform a test with the given parameters
    :param mode: Test to perform: test_1 or test_2
    :param env_params: dict, relevant parameters for environment setup
    :param ac_params: dict, relevant parameters for actor-critic setup
    :param num_seeds: Amount of random seeds tested
    :return:
    """
    results_path = 'results/apr/10/test_1/high/'
    with mp.Pool(processes=10) as pool:
        results = [pool.apply_async(train_test, args=(mode, env_params, ac_params, results_path+str(seed)+"/", seed))
                   for seed in range(num_seeds)]
        results = [p.get() for p in results]
        succesful_trials = [x[0] for x in results]
        print("Succesful trials: ", sum(succesful_trials), "/", num_seeds)

        pool.close()
        pool.join()


if __name__ == "__main__":
    # taus = (0.01, 0.1, 1.0)
    # nn_stdevs = (0.05, 0.1, 0.2)
    # gammas = (0.7, 0.8, 0.9, 0.95, 0.99)
    # lrs_act = (2, 5, 10)
    # lrs_crit = (2, 5, 10)
    # settings = list(itertools.product(*[taus, nn_stdevs, gammas, lrs_act, lrs_crit]))
    # mp_train(settings, num_seeds=40)

    print("Starting expertiment 1: accel-decel")
    mp_test('test_1', env_params_test1, ac_params_test)
    print("Finished experiment 1. ")
    print(" ----------------------------- ")
    # print("Starting experiment 2: OEI landing")
    # mp_test('test_2', env_params_test2, ac_params_test)
