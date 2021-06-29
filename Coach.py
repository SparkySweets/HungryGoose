from Utils import gen_epsilon_greedy_policy, get_features
from agents import epsilon_really_greedy_agent

def q_learning(env, estimator, memory, mode, n_episode, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.999):
    total_reward_episode = [0] * n_episode
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(estimator, epsilon, 4)
        trainer = env.train([None, "greedy", "greedy", "greedy"])
        observation = trainer.reset()
        is_done = False
        prev_heads = [-1 for _ in range(4)]
        prev_obs = observation
        state = get_features(observation, env.configuration, prev_heads)
        length = 1
        old_length = 1
        max_length = 1
        steps = 0
        local_memory = []
        memory_flag = False
        old_action = 0
        while not is_done:
            if mode == 'greedy':
                action = epsilon_really_greedy_agent(observation, env.configuration, epsilon)
            else:
                action = policy(state)
            observation, reward, is_done, _ = trainer.step(['NORTH', 'EAST', 'SOUTH', 'WEST'][action])
            #print(observation['index'])
            length = len(observation['geese'][observation['index']])
            if reward > 99:
                reward = reward // 100
            if length == 0:
                max_length = old_length
                modified_reward = -100
                if (action - old_action) % 4 == 2:
                    # stupid death
                    modified_reward = -1000
                else:
                    # heroic death
                    modified_reward = -100
                
            elif length > old_length:
                memory_flag = True
                modified_reward = reward + 10
                max_length = length
            else:
                modified_reward = reward
            old_length = length
            steps += 1
            #print('reward = {}'.format(reward))
            #print('length = {}'.format(len(observation['geese'][0])))    
            total_reward_episode[episode] += modified_reward
            
            prev_heads = [goose[0] if len(goose) > 0 else -1 for goose in prev_obs['geese']]
            #print(observation)
            #print(prev_heads)
            
            next_state = get_features(observation, env.configuration, prev_heads)
            #memory.append((state, action, next_state, reward, is_done))
            #memory.append((state, action, next_state, modified_reward, is_done))
            local_memory.append((state, action, next_state, modified_reward, is_done))
                
            if is_done:
                if memory_flag:
                    for mem in local_memory:
                        memory.append(mem)
                local_memory = []
                memory_flag = False
                break
            estimator.replay(memory, replay_size, gamma)
                
            state = next_state
            prev_obs = observation
            #old_action = action
        if episode % 100 == 0:
            print('Эпизод: {}, полное вознаграждение: {}, максимальная длина: {}, число шагов: {}, epsilon:{}'.format(
            episode, total_reward_episode[episode], max_length, steps, epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)
    return total_reward_episode