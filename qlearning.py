import gym
import numpy as np
env = gym.make("MountainCar-v0", render_mode='human')


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODE = 25000



SHOW_EVERY = 2000


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE;

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAING = EPISODE //2
epsilon_decay_value = epsilon/(END_EPSILON_DECAING-START_EPSILON_DECAYING)

q_table = np.random.uniform(low = -2 , high = 0 , size = (DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = ((state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE)
    return tuple(discrete_state.astype(np.int64))



for epsidode in range(EPISODE):

    

    if epsidode%SHOW_EVERY == 0:
        print(epsidode)
        render = True
    else:
        render = False
    

    observation , info = env.reset()
    discrete_state = get_discrete_state(observation)
    done = False
    
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        
        if render==True :
            env.render()
        if not done :
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action , )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)

            q_table[discrete_state + (action , )] = new_q

        elif new_state[0] >= env.goal_position :
            print(f"we made it in {epsidode}")
            q_table[discrete_state + (action , )] = 0


        discrete_state = new_discrete_state

    if END_EPSILON_DECAING >= epsidode >= START_EPSILON_DECAYING:
        epsilon -=epsilon_decay_value

env.close()