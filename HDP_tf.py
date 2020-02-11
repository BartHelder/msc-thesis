import numpy as np



tf.random.set_seed(3)
cfp = "config_3dof.json"

env = Helicopter3DOF()
env.setup_from_config(task="sinusoid", config_path=cfp)
observation, trim_actions = env.reset(v_initial=20)

rls_kwargs = {'state_size': 7, 'action_size': 2, 'gamma': 1, 'covariance': 10**8, 'constant': False}
RLS = RecursiveLeastSquares(**rls_kwargs)
CollectiveAgent = Agent(cfp, actor=TFActor3DOF, actor_kwargs={'offset': 0, 'action_scaling': np.rad2deg(trim_actions[0])}, control_channel='collective')
CollectiveAgent.set_ds_da(RLS)
CyclicAgent = Agent(cfp, actor=TFActor3DOF, actor_kwargs={'offset': 0, 'action_scaling': 10}, control_channel="cyclic_lon")
CyclicAgent.set_ds_da(RLS)
agents = (CollectiveAgent, CyclicAgent)
stats = []
reward = [None, None]
weight_stats = {'t': [], 'wci': [], 'wco': [], 'wai': [], 'wao': []}
rls_stats = {'t': [0],
             'wa_col': [RLS.gradient_action()[:6, 0].ravel().copy()],
             'wa_cyc': [RLS.gradient_action()[:6, 1].ravel().copy()],
             'ws': [RLS.gradient_state().ravel().copy()]}

excitation = np.zeros((1000, 2))
for j in range(400):
    excitation[j, 0] = -np.sin(np.pi*j/50)
    excitation[j+400, 1] = np.sin(2*np.pi*j/50) * 2

excitation = np.deg2rad(excitation)
excitation_phase = True
done = False
step = 0
while not done:

    # Get new reference
    reference = env.get_ref()

    # Augment state with tracking errors
    augmented_states = (CollectiveAgent.augment_state(observation, reference),
                        CyclicAgent.augment_state(observation, reference))

    # Get actions from actors
    actions = np.array([CollectiveAgent.actor(augmented_states[0]).numpy().squeeze(),
                        CyclicAgent.actor(augmented_states[1]).numpy().squeeze()])

    # In excitation phase add values to actions
    if excitation_phase:
        actions += excitation[step]

    actions = actions + trim_actions
    actions = np.clip(actions, np.deg2rad([0, -15]), np.deg2rad([10, 15]))
    # Take step in the environment
    next_observation, _, done = env.step(actions)

    # Update RLS model
    RLS.update(state=observation, action=actions, next_state=next_observation)

    # Update action gradients
    CollectiveAgent.set_ds_da(RLS)
    CyclicAgent.set_ds_da(RLS)

    # Get rewards, update actor and critic networks
    for agent, count in zip(agents, itertools.count()):
        reward[count] = agent.get_reward(next_observation, reference)
        next_augmented_state = agent.augment_state(next_observation, reference)
        td_target = reward[count] + agent.gamma * agent.critic(next_augmented_state)
        agent.update_networks(td_target, augmented_states[count], n_updates=1)
        if count == 1:
            break

    # Log data
    stats.append({'t': env.t,
                  'x': observation[0],
                  'z': observation[1],
                  'u': observation[2],
                  'w': observation[3],
                  'theta': observation[4],
                  'q': observation[5],
                  'reference': env.get_ref(),
                  'collective': actions[0],
                  'cyclic': actions[1],
                  'r1': reward[0],
                  'r2': reward[1]})

    rls_stats['t'].append(env.t)
    rls_stats['ws'].append(RLS.gradient_state().ravel())
    rls_stats['wa_col'].append(RLS.gradient_action()[:6, 0].ravel())
    rls_stats['wa_cyc'].append(RLS.gradient_action()[:6, 1].ravel())

    if env.t > 16:
        excitation_phase = False
    if env.t > 100:
        done = True

    # Next step..
    observation = next_observation
    step += 1
stats = pd.DataFrame(stats)
plot_stats_3dof(stats)

wa_col = pd.DataFrame(data=rls_stats['wa_col'], index=rls_stats['t'], columns = ['ex', 'z', 'u', 'w', 'pitch', 'q'])
wa_cyc = pd.DataFrame(data=rls_stats['wa_cyc'], index=rls_stats['t'], columns = ['ex', 'z', 'u', 'w', 'pitch', 'q'])
wa_col = wa_col.drop(columns=['ex', 'u', 'pitch', 'q'])
wa_cyc = wa_cyc.drop(columns=['ex', 'z', 'u', 'w'])
ws = pd.DataFrame(data=rls_stats['ws'], index=rls_stats['t'])
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure()
sns.lineplot(data=wa_col, dashes=False, legend='full', palette=sns.color_palette("hls", len(wa_col.columns)))
plt.xlabel('Time [s]')
plt.ylabel('Gradient size [-]')
plt.title('iRLS Collective gradients')
plt.show()

plt.figure()
sns.lineplot(data=wa_cyc, dashes=True, legend='full', palette=sns.color_palette("hls", len(wa_cyc.columns)))
plt.xlabel('Time [s]')
plt.ylabel('Gradient size [-]')
plt.title('iRLS Cyclic gradients')
plt.show()
