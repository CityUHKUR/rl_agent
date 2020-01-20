# Import simulation Env

from markov.environments.deepracer_env import DeepRacerEnv
from gym.envs.registration import register
import gym

# Import training required packags
import tensorflow as tf      # Deep Learning library
from tensorflow.keras import layers,initializers
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends


# Define my own for action space and reward target for environment
class DeepRacerMultiDiscreteEnv(DeepRacerEnv):
    def __init__(self):
        DeepRacerEnv.__init__(self)

        # actions -> straight, left, right
        self.action_space = spaces.Discrete(10)

    def step(self, action):

        # Convert discrete to continuous
        if action == 0:  # straight
            throttle = 0.3  # 0.5
            steering_angle = 0
        elif action == 1:
            throttle = 0.7
            steering_angle = 0
        elif action == 2:
            throttle = 1.0
            steering_angle = 0
        elif action == 3:
            throttle = 0.1
            steering_angle = 1
        elif action == 4:  # move left
            throttle = 0.1
            steering_angle = -1
        elif action == 5:  # move left
            throttle = 0.3
            steering_angle = 0.75
        elif action == 6:  # move left
            throttle = 0.3
            steering_angle = -0.75
        elif action == 7:  # move right
            throttle = 0.5
            steering_angle = 0.5
        else:  # action == 4
            throttle = 0.5
            steering_angle = -0.5

        continous_action = [steering_angle, throttle]

        return super().step(continous_action)

"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """
def preprocess_frame(frame,noise=False):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]

    
    # Normalize Pixel Values
    normalized_frame = frame/255.0
    ## TRAINING_IMAGE_SIZE = (160, 120)
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [160,120,3])

    # Add noise if desired
    
    return preprocessed_frame



def stack_frames(stacked_frames, state, is_new_episode, stack_size=4):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((160,120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        for i in range(stack_size):
            stacked_frames.append(frame)

        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=-1)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=-1) 
    
    return stacked_state, stacked_frames

def discount_and_normalize_rewards(episode_rewards,gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


class DriveDQN(tf.keras.Model):
    def __init__(self, state_size, action_size, learning_rate, name='DriveDQN'):
        super(DriveDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.model = tf.keras.Sequential()
        self.conv1 = layers.Conv3D(input_shape=(160,120,3,4),
                                     filters = 32,
                                     kernel_size = [8,8,3],
                                     strides = [3,4,4],
                                     padding = "valid",
                                      kernel_initializer=initializers.GlorotUniform)
        
        self.conv1_batchnorm = layers.BatchNormalization(
                                                         trainable = True,
                                               epsilon = 1e-5)
        self.conv1_out = layers.ELU()
        ## --> [20, 20, 32]

        self.conv2 = layers.Conv3D(
                             filters = 64,
                             kernel_size = [4,4,1],
                             strides = [3,2,2],
                             padding = "valid",
                            kernel_initializer=initializers.GlorotUniform)

        self.conv2_batchnorm = layers.BatchNormalization(
                                               trainable = True,
                                               epsilon = 1e-5)

        self.conv2_out = layers.ELU()
        ## --> [9, 9, 64]

        self.conv3 = layers.Conv3D(
                             filters = 128,
                             kernel_size = [4,4,1],
                             strides = [3,2,2],
                             padding = "valid",
                            kernel_initializer=initializers.GlorotUniform)

        self.conv3_batchnorm = layers.BatchNormalization(
                                               trainable = True,
                                               epsilon = 1e-5)

        self.conv3_out = layers.ELU()
        ## --> [3, 3, 128]


        self.flatten = layers.Flatten()
        ## --> [1152]


        self.fc = layers.Dense(
                               units = 512,
                              activation = tf.nn.elu,
                                   kernel_initializer=initializers.GlorotUniform)

        self.logits = layers.Dense(
                                   kernel_initializer=initializers.GlorotUniform,
                                      units = action_size, # unwrap tuple
                                    activation=None)
        self.chain = [self.conv1,self.conv1_batchnorm,  self.conv1_out,self.conv2,self.conv2_batchnorm,self.conv2_out,self.conv3,self.conv3_batchnorm,self.conv3_out,self.flatten,self.fc,self.logits]
        for layer in self.chain:
            self.model.add(layer)
        

        
    def call(self,state,training=False):
        return tf.nn.softmax(self.model(state,training=training)).numpy()
                

def loss(model,state,actions,discounted_episode_rewards_,training=False) :
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model.model(state,training=training), labels = actions) * discounted_episode_rewards_) 
        
def make_batch(env, model, batch_size, stacked_frames, stack_size, training, possible_actions,gamma,state_size):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num  = 1
    
    # Launch a new episode
    state = env.reset()
        
    # Get a new state
    state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)

    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = model(state.reshape(1,*state_size),training=training)
        
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        #30% chance that we take action a2)
        action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]
        action_one_hot = np.zeros(len(possible_actions))
        action_one_hot[action]=1
        # Perform action
        next_state, reward, done, info = env.step(action)
        state, stacked_frames = stack_frames(stacked_frames, next_state, False)


        # Store results
        states.append(state)
        actions.append(action_one_hot)
        rewards_of_episode.append(reward)
        
        if done:
            # The episode ends so no next state
            next_state = np.zeros((160, 120, 3), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
            
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode,gamma))
           
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if episode_num > batch_size:
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
            pass
                         
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num


if __name__ == "__main__":
    env = gym.make('DeepRacer-v1')
    env = env.unwrapped
    # env = DeepRacerMultiDiscreteEnv()
    ################################
    #
    #   Initialize training hyperparameters
    #
    #################################

    state_size = [160, 120, 3,4] # Our input is a stack of 4 frames hence 160x120x3x4 (Width, height, channels*stack_size) 
    action_size = env.action_space.n # 10 possible actions: turn left, turn right, move forward
    print(str(env.action_space.n))
    print(str(env.observation_space))
    possible_actions = [x for x in range(action_size)]
    stack_size = 4 # Defines how many frames are stacked together

    ## TRAINING HYPERPARAMETERS
    learning_rate = 0.001
    num_epochs = 500 # Total epochs for training 

    batch_size = 10 # Each 1 is AN EPISODE # YOU CAN CHANGE TO 50 if you have GPU
    gamma = 0.9 # Discounting rate

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

# Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((160,120,3), dtype=np.int) for i in range(stack_size)], maxlen=4) 
    # Set up optimizer
    agent = DriveDQN(state_size,action_size,learning_rate)
    train_opt = tf.keras.optimizers.RMSprop(learning_rate)


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

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),optimizer=train_opt, model=agent)
    manager = tf.train.CheckpointManager(checkpoint, directory="./ckpt/model", max_to_keep=5)

    print("[INFO] START TRAINING")
    epoch = 1
    allRewards=[]
    mean_reward_total=[]
    average_reward = []
    while training:
        # Gather training data
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(env,agent,batch_size, stacked_frames, stack_size, training, possible_actions,gamma,state_size)

        ### These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)

        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)

        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

        # Calculate maximum reward recorded 
        maximumRewardRecorded = np.amax(allRewards)

        print("==========================================")
        print("Epoch: ", epoch )
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

        # Feedforward, gradient and backpropagation
        loss_ = lambda : loss(agent,states_mb,actions_mb,discounted_rewards_mb,training=training)
        vars_ = lambda : agent.trainable_variables
        train_opt.minimize(loss_,vars_)
        print("Training Loss: {}".format(loss_()))


        # Save Model
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 100 == 0:
            save_path = manager.save()

        epoch += 1