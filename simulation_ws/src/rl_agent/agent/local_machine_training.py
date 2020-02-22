# Import simulation Env

import gym

# Import training required packags
import tensorflow as tf  # Deep Learning library
from tensorflow.keras import layers, initializers
import numpy as np  # Handle matrices

from rocket.ignite.layers import downsample3D, upsample3D
from rocket.ignite.loss import cross_entropy_loss
from rocket.ignite.models import FPN3D
from rocket.image.preprocess import stack_frames
from rocket.ignite.types import Transition

from datetime import datetime  # Help us logging time

from collections import deque  # Ordered collection with ends

from rocket.utils.exprience import ExperienceBuffer

tf.keras.backend.set_floatx('float64')




# Define Out Driver Forward Value Network
class DriveDQN(tf.keras.Model):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, state_size, action_size, learning_rate, feature_extract, name='DriveDQN'):
        super(DriveDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.inputs = tf.keras.layers.Input(shape=(4, 256, 256, 3))
        self.fe = feature_extract
        self.model = tf.keras.Sequential()
        self.model.add(self.fe)

        ## --> [512]
        self.fc1 = layers.Dense(
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=initializers.GlorotUniform)(self.model(self.inputs))
        self.advs = []
        for i in range(action_size):
            self.advs.append(layers.Dense(
                    kernel_initializer=initializers.GlorotUniform,
                    units=512,  # unwrap tuple
                    activation=None)(self.fc1))
        self.fc2 = layers.Dense(
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=initializers.GlorotUniform)(self.model(self.inputs))

        self.value = layers.Dense(
                kernel_initializer=initializers.GlorotUniform,
                units=512,  # unwrap tuple
                activation=None)(self.fc2)
        self.aggeregate = layers.Concatenate()(
                [layers.Dense(
                        units=128,
                        activation=tf.nn.elu,
                        kernel_initializer=initializers.GlorotUniform)(
                        layers.multiply([self.value, adv]))
                        for adv in self.advs])
        self.logits = layers.Dense(kernel_initializer=initializers.GlorotUniform,
                                   units=action_size,  # unwrap tuple
                                   activation=None)(self.aggeregate)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.logits)

    def call(self, state, training=False):
        return tf.nn.softmax(self.model(state, training=training)).numpy()


class TargetDQN(tf.keras.Model):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, state_size, action_size, learning_rate, feature_extract, name='TargetDQN'):
        super(TargetDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.fe = feature_extract

        self.model = tf.keras.Sequential()
        self.fc1 = layers.Dense(
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=initializers.GlorotUniform)


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


def make_batch(env, model, train_opt, episodes, batch_size, memory, stacked_frames, stack_size, training,
               possible_actions, gamma, state_size, explore_rate):
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
    state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)

    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = model(state.reshape(1, *state_size), training=training)

        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)
        if np.random.rand() <= explore_rate:
            action = np.random.choice(possible_actions)
        else:
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]
        action_one_hot = np.zeros(len(possible_actions))
        action_one_hot[action] = 1
        # Perform action
        next_state, reward, done, info = env.step(action)
        state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.store(Transition(state, np.transpose(action_one_hot), reward, next_state), reward)

        # # Store results
        # states.append(state)
        # actions.append(action_one_hot)
        # rewards_of_episode.append(reward)

        if done:
            # The episode ends so no next state
            next_state = np.zeros((160, 120, 3), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)

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
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)


        else:
            # If not done, the next_state become the current state
            # run_steps += 1
            pass

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

    return np.stack(states_mb), np.stack(actions_mb), rewards_mb, np.stack(next_states_mb)


if __name__ == "__main__":
    env = gym.make('DeepRacer-v1')
    env = env.unwrapped
    # env = DeepRacerMultiDiscreteEnv()
    ################################
    #
    #   Initialize training hyperparameters
    #
    #################################

    state_size = [4, 256, 256,
                  3]  # Our input is a stack of 4 frames hence 160x120x3x4 (Width, height, channels*stack_size)
    action_size = env.action_space.n  # 10 possible actions: turn left, turn right, move forward
    print(str(env.action_space.n))
    print(str(env.observation_space))
    possible_actions = [x for x in range(action_size)]
    stack_size = 4  # Defines how many frames are stacked together

    ## TRAINING HYPERPARAMETERS
    learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=50,
            decay_rate=0.95,
            staircase=True)
    episodes = 10  # Total episodes for sampling
    explore_rate = 0.5
    min_explore_rate = 0.1
    explore_decay_rate = 0.95
    explore_decay_step = 30

    batch_size = 512  # Each 1 is AN EPISODE
    mini_batch_size = 128
    gamma = 0.9  # Discounting rate

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
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
    stacked_frames = deque([np.zeros((256, 256, 3), dtype=np.int) for i in range(stack_size)], maxlen=4)
    # Set up optimizer
    agent = DriveDQN(state_size, action_size, learning_rate)
    train_opt = tf.keras.optimizers.Adam(lr_schedule)
    # memory = Memory(2000 * int(episodes/2))
    memory = ExperienceBuffer(2500)

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

    # Create Save checkpoint

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=train_opt, model=agent)
    manager = tf.train.CheckpointManager(checkpoint, directory="./ckpt/model", max_to_keep=5)
    # Define the Keras TensorBoard callback.
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/deepracer"
    rewards_log_dir = log_dir + "/rewards/" + current_time
    loss_log_dir = log_dir + "/loss/" + current_time
    rewards_summary_writer = tf.summary.create_file_writer(rewards_log_dir)
    loss_summary_writer = tf.summary.create_file_writer(loss_log_dir)

    # Define Summary Metrics

    # restore saved model if available
    checkpoint.restore(manager.latest_checkpoint)
    print("[INFO] START TRAINING")
    epoch = 1
    # allRewards=[]
    # mean_reward_total=[]
    # average_reward = []
    while training:
        # Gather training data
        nb_episodes_mb = make_batch(env, agent, train_opt, episodes, batch_size, memory, stacked_frames, stack_size,
                                    training, possible_actions, gamma, state_size,
                                    explore_rate=min_explore_rate + (explore_rate - min_explore_rate) * np.exp(
                                            -(1 - np.power(explore_decay_rate, epoch / explore_decay_step)) * epoch))

        ### These part is used for analytics
        # Calculate the total reward ot the batch

        # allRewards.append(total_reward_of_that_batch)

        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        # mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        # mean_reward_total.append(mean_reward_of_that_batch)

        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        # average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

        # Calculate maximum reward recorded 
        # maximumRewardRecorded = np.amax(allRewards)

        print("==========================================")
        print("Epoch: ", epoch)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))

        # print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        # print("Average Reward of all training: {}".format(average_reward_of_all_training))
        # print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

        # Feedforward, gradient and backpropagation

        mean_loss = []
        mean_total_reward = []

        for mini_batch in make_mini_batch((samples(memory, batch_size, randomness=0.5)), batch_size=batch_size,
                                          mini_batch_size=mini_batch_size):
            states_mb, actions_mb, rewards_mb = mini_batch
            loss_ = lambda: cross_entropy_loss(agent, states_mb, actions_mb, rewards_mb, training=training)
            vars_ = lambda: agent.trainable_variables
            train_opt.minimize(loss_, vars_)
            mean_loss.append(loss_())
            mean_total_reward.append(np.sum(rewards_mb))

        #             weights = model.get_weights()  # Retrieves the state of the model.
        #             model.set_weights(weights)  # Sets the state of the model.

        print("Total reward: {}".format(np.mean(mean_total_reward), nb_episodes_mb))
        print("Training Loss: {}".format(np.mean(mean_loss)))

        # write summary to files
        with loss_summary_writer.as_default():
            tf.summary.scalar('train_loss', np.mean(mean_loss), step=epoch)
        with rewards_summary_writer.as_default():
            tf.summary.scalar('total_reward_of_that_batch', np.mean(mean_total_reward), step=epoch)

        # Save Model
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 10 == 0:
            save_path = manager.save()

        epoch += 1
