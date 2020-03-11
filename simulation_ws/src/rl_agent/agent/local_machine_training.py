# Import simulation Env

from collections import deque  # Ordered collection with ends
from datetime import datetime  # Help us logging time
import environments
import gym
from gym.envs.registration import register
import os
import numpy as np  # Handle matrices
# Import training required packags
import torch  # Deep Learning library
import torch.nn as nn
import torch.nn.functional as F
from rocket import ignite, images
from drivers import IntrinsicReward, FeatureExtract

from rocket.ignite.types import Transition
from rocket.ignite.layers import init_conv, init_linear
from rocket.images.preprocess import stack_frames
from rocket.utils.exprience import ExperienceBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


def make_batch(env, model, episodes, memory, stacked_frames, stack_size, training,
               possible_actions, state_size, explore_rate):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    # states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []

    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.

    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num = 1
    run_steps = 1
    # Launch a new episode
    state = env.reset()

    # Get a new state
    state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=stack_size)

    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = model(
            torch.as_tensor(state.reshape(1, *state_size)).to(device=device, dtype=torch.float)).detach()
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)
        if np.random.rand() <= explore_rate:
            action = np.random.choice(possible_actions)
        else:
            action = np.random.choice(possible_actions,
                                      p=action_probability_distribution.view(
                                          -1).cpu().numpy())  # select action w.r.t the actions prob
        action = possible_actions[action]
        action_one_hot = np.zeros(len(possible_actions))
        action_one_hot[action] = 1
        # Perform action
        next_state, reward, done, info = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=stack_size)
        memory.store(Transition(state, action_one_hot, reward, next_state), reward)

        # # Store results
        # states.append(state)
        # actions.append(action_one_hot)
        # rewards_of_episode.append(reward)

        if done:
            # The episode ends so no next state
            # next_state = np.zeros((256, 256, 3), dtype=np.int)
            # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=stack_size)

            # Append the rewards_of_batch to reward_of_episode
            # rewards_of_batch.append(rewards_of_episode)

            # Calculate gamma Gt
            # discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode,gamma))

            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if episode_num >= episodes:
                break

            # Reset the transition stores
            rewards_of_episode = []

            # Add episode
            episode_num += 1

            # Start a new episode
            state = env.reset()

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=stack_size)

        else:
            # If not done, the next_state become the current state
            # run_steps += 1

            # print(np.shape(next_state))
            # model.forward.minimize(inputs=[state.reshape(1, *state_size), action_one_hot, reward, next_state.reshape(1, *state_size)], optimizer=opt)
            # model.inverse.minimize(inputs=[state.reshape(1, *state_size), action_one_hot, reward, next_state.reshape(1, *state_size)], optimizer=opt)
            state = next_state

    return episode_num


def make_mini_batch(data, batch_size, mini_batch_size):
    mini_batch = []
    for (start_idx, end_idx) in zip(np.arange(0, batch_size - mini_batch_size, mini_batch_size),
                                    np.arange(mini_batch_size, batch_size, mini_batch_size)):
        mini_batch.append([mini_data[start_idx:end_idx] for mini_data in data])

    return mini_batch


def samples(memory, batch_size, randomness):
    (transitions, weights) = memory.sample(batch_size, randomness=randomness)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    states_mb = batch.state
    actions_mb = batch.action
    rewards_mb = batch.reward
    next_states_mb = batch.next_state
    print(np.shape(states_mb))
    return np.stack(states_mb), np.stack(actions_mb), np.stack(rewards_mb), np.stack(next_states_mb)


if __name__ == "__main__":
    env = gym.make('DeepRacer-v1')
    env = env.unwrapped
    # env = DeepRacerMultiDiscreteEnv()
    ################################
    #
    #   Initialize training hyperparameters
    #
    #################################
    # resize to 256 x 256
    state_size = [3, 4, 128, 128,
                  ]  # Our input is a stack of 4 frames hence 4x160x120x3 (stack size,Width, height, channels)

    action_size = env.action_space.n  # 10 possible actions: turn left, turn right, move forward
    print(str(env.action_space.n))
    print(str(env.observation_space))
    possible_actions = [x for x in range(action_size)]
    stack_size = 4  # Defines how many frames are stacked together

    # TRAINING HYPERPARAMETERS
    learning_rate = 1e-3
    max_episodes = 10
    min_episodes = 5  # Total episodes for sampling
    explore_rate = 0.5
    min_explore_rate = 0.1
    explore_decay_rate = 0.95
    explore_decay_step = 30
    epoch = 0
    batch_size = 2048  # Each 1 is AN EPISODE
    mini_batch_size = 128
    gamma = 0.9  # Discounting rate

    # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    ##################################
    #
    #   End of Hyperparameters
    #
    ##################################

    ################################
    #
    #   Initialize  variables
    #
    #################################

    # Initialize deque with zero-images one array for each input_image
    stacked_frames = deque([np.zeros((3, 128, 128), dtype=np.int) for i in range(stack_size)], maxlen=4)

    # Create Save checkpoint
    save_point = "./ckpt/deepracer/icn+entropy.pth"
    if not os.path.exists(save_point):
        f = open(save_point,"w+")
        f.close()



    # Set up Feature Extraction Layer

    feature_extractor = FeatureExtract().cuda()



    # Set up Proximal Policy Agent + Intrinsic Reward + Inverse Dynamics
    curiosity_agent = IntrinsicReward(action_size, 512, feature_extractor)

    # restore saved model if available
    if not os.stat(save_point).st_size == 0:
        checkpoint = torch.load(save_point, map_location=device)

        curiosity_agent_state_dict =  checkpoint['curiosity_agent_state_dict']
        curiosity_agent_state_dict.pop('fc2.weight')
        curiosity_agent_state_dict.pop('fc2.bias')
        curiosity_agent_state_dict.pop('fc3.weight')
        feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'],strict=False)
        curiosity_agent.load_state_dict(curiosity_agent_state_dict,strict=False)
        # curiosity_agent.optimizer.load_state_dict(checkpoint['curiosity_agent_optimizer_state_dict'],strict=False)
        # epoch = checkpoint['epoch']

    curiosity_agent = curiosity_agent.apply(init_linear)
    curiosity_agent = curiosity_agent.apply(init_conv)
    curiosity_agent = curiosity_agent.to(device)
    curiosity_agent = curiosity_agent.cuda()
    curiosity_agent = curiosity_agent.apply(lambda m: m.cuda())
    # Use Internal Optimizer

    memory = ExperienceBuffer(1500 * 5)

    ################################
    #
    #   End of  variables
    #
    #################################

    #############################
    #
    # Start Training
    #
    #############################


    # Define the Keras TensorBoard callback.

    # Define Summary Metrics

    print("[INFO] START TRAINING")
    epoch += 1
    episodes = 15
    while training:
        # Gather training data

        nb_episodes_mb = make_batch(env, curiosity_agent, episodes, memory, stacked_frames, stack_size,
                                    training, possible_actions, state_size,
                                    explore_rate=min_explore_rate + (explore_rate - min_explore_rate) * np.exp(
                                            -(1 - np.power(explore_decay_rate, epoch / explore_decay_step)) * epoch))

        ### These part is used for analytics
        # Calculate the total reward ot the batch

        # Calculate the mean reward of the batch

        # Calculate the average reward of all training

        # Calculate maximum reward recorded
        # maximumRewardRecorded = np.amax(allRewards)

        print("==========================================")
        print("Epoch: ", epoch)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))

        # Feedforward, gradient and backpropagation

        mean_loss = []
        mean_total_reward = []
        mini_batch_run = 1
        for mini_batch in make_mini_batch((samples(memory, batch_size, randomness=0.5)), batch_size=batch_size,
                                          mini_batch_size=mini_batch_size):
            states_mb, actions_mb, rewards_mb, next_states_mb = mini_batch

            mini_batch = tuple([torch.as_tensor(mb).to(device=device, dtype=torch.float) for mb in
                                [states_mb, actions_mb, rewards_mb, next_states_mb]])

            states_mb, actions_mb, rewards_mb, next_states_mb = mini_batch

            print("mini_batch running {} times ".format(mini_batch_run))

            curiosity_agent.minimize(inputs=mini_batch)

            mean_loss.append(curiosity_agent.loss(mini_batch).detach().cpu().numpy())
            mean_total_reward.append(np.sum(rewards_mb.detach().cpu().numpy()))

            mini_batch_run += 1

        print("Total reward: {}".format(np.mean(mean_total_reward), nb_episodes_mb))
        print("Training Loss: {}".format(np.mean(mean_loss)))

        # write summary to files

        # save checkpoint
        if epoch % 10 == 0:
            torch.save({
                    'epoch':                                epoch,
                    'feature_extractor_state_dict':         feature_extractor.state_dict(),
                    'curiosity_agent_state_dict':           curiosity_agent.state_dict(),
                    'curiosity_agent_optimizer_state_dict': curiosity_agent.optimizer.state_dict(),
            }, save_point)

        epoch += 1
        episodes = int(max_episodes * np.exp(-epoch * gamma / 30)) + min_episodes
