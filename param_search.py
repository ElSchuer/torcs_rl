import dqn_agent
import eval
import torcs_env
import torcs

batch_space = [64, 256, 512]
learning_rate_space = [0.001, 0.0005, 0.0001]
eps_min_space = [0.01, 0.1, 0.2]
max_episodes = 150
env_name = 'torcs_env'

eval_inst = eval.RLEvaluation(resume_train=False)
env = torcs_env.TorcsEnvironment(eval_inst=eval_inst, max_episodes=max_episodes)

for batch in batch_space:
    for lr in learning_rate_space:
        for eps_min in eps_min_space:
            model = torcs.get_model(env.action_size, env.state_size)
            agent = dqn_agent.DQNAgent(state_size=env.state_size, action_size=env.action_size,
                                       model=model,
                                       learning_rate=lr, queue_size=500000, batch_size=batch,
                                       decay_rate=0.95, loss='mse')

            agent.enable_target_network(update_steps=10000)
            agent.enable_double_dqn()
            agent.enable_epsilon_greedy(eps_decay=0.999, eps_min=eps_min, eps_start=1.0)
            agent.enable_dueling_dqn(dueling_type='mean')

            env.set_agent(agent)
            env.set_reward_func(torcs.reward_function)

            env.learn(resume_train=False)

            plot_filename = env_name + '_' + 'b' + str(batch) + '_l' + str(lr) + '_e' + str(eps_min) + '.jpeg'
            eval_inst.save_plot('./log/', plot_filename)
            eval_inst.reset()
