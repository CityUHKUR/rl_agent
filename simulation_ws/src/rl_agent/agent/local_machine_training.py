# Import simulation Env

from agent.environments.deepracer_env import DeepRacerEnv
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
tf.keras.backend.set_floatx('float64')


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
        stacked_frames = deque([np.zeros((160,120,3), dtype=np.float64) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        for i in range(stack_size):
            stacked_frames.append(frame)

        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0) 
    
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
        
        self.inputs = tf.keras.layers.Input(shape=(4,160,120,3))

        self.model = tf.keras.Sequential()
        self.conv1 = layers.Conv3D(
                                    filters = 32,
                                     kernel_size = [4,4,4],
                                     strides = [1,4,4],
                                     padding = "valid",
                                      kernel_initializer=initializers.GlorotUniform)
        
        self.conv1_batchnorm = layers.BatchNormalization(
                                                         trainable = True,
                                               epsilon = 1e-5)
        self.conv1_out = layers.ELU()
        ## --> [20, 20, 32]

        self.conv2 = layers.Conv3D(
                             filters = 64,
                             kernel_size = [1,3,3],
                             strides = [1,2,2],
                             padding = "valid",
                            kernel_initializer=initializers.GlorotUniform)

        self.conv2_batchnorm = layers.BatchNormalization(
                                               trainable = True,
                                               epsilon = 1e-5)

        self.conv2_out = layers.ELU()
        ## --> [9, 9, 64]

        self.conv3 = layers.Conv3D(
                             filters = 128,
                             kernel_size = [1,2,2],
                             strides = [1,2,2],
                             padding = "valid",
                            kernel_initializer=initializers.GlorotUniform)

        self.conv3_batchnorm = layers.BatchNormalization(
                                               trainable = True,
                                               epsilon = 1e-5)

        self.conv3_out = layers.ELU()
        ## --> [3, 3, 128]


        self.flatten = layers.Flatten()
        ## --> [1152]

        self.chain = [self.conv1,self.conv1_batchnorm,  self.conv1_out,self.conv2,self.conv2_batchnorm,self.conv2_out,self.conv3,self.conv3_batchnorm,self.conv3_out,self.flatten]
        for layer in self.chain:
            self.model.add(layer)
        # self.fc1 = layers.Dense(
        #                        units = 512,
        #                       activation = tf.nn.elu,
        #                            kernel_initializer=initializers.GlorotUniform)(self.model(self.inputs))
        self.advs = []
        for i in range(action_size):
            self.advs.append(layers.Dense(
                                   kernel_initializer=initializers.GlorotUniform,
                                      units = 512, # unwrap tuple
                                    activation=None)(self.model(self.inputs)))
        # self.fc2 = layers.Dense(
        #                        units = 512,
        #                       activation = tf.nn.elu,
        #                            kernel_initializer=initializers.GlorotUniform)(self.model(self.inputs))

        
        self.value = layers.Dense(
                                   kernel_initializer=initializers.GlorotUniform,
                                      units = 512, # unwrap tuple
                                    activation=None)(self.model(self.inputs))
        self.aggeregate = layers.Concatenate()([layers.multiply([self.value,adv]) for adv in self.advs])
        self.logits = layers.Dense(kernel_initializer=initializers.GlorotUniform,
                                      units = action_size, # unwrap tuple
                                    activation=None)(self.aggeregate)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.logits)

        
        

        
    def call(self,state,training=False):
        return tf.nn.softmax(self.model(state,training=training)).numpy()
                

def loss(model,states,actions,discounted_episode_rewards_,training=False) :
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model.model(states,training=training), labels = actions) * discounted_episode_rewards_) 

class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
        
def make_batch(env, model, train_opt, episodes, batch_size, memory ,stacked_frames, stack_size, training, possible_actions, gamma, state_size, mini_update=False):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    # states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num  = 1
    run_steps = 1
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
        memory.store((state,action_one_hot,reward,next_state))


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
            if episode_num >= episodes and run_steps >= batch_size :
                break
                
            # Reset the transition stores
            rewards_of_episode = []
            
            # Add episode
            episode_num += 1
            
            # Start a new episode
            state = env.reset()

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)

            # if run_steps >= batch_size and mini_update :
            #     loss_ = lambda : loss(model,np.stack(states),np.stack(actions),np.concatenate(discounted_rewards),training=training)
            #     vars_ = lambda : model.trainable_variables
            #     train_opt.minimize(loss_,vars_)
            #     states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
            #     run_steps = 0
            
                
        else:
            # If not done, the next_state become the current state
            run_steps += 1
            pass



        (b_idx,mb,b_weights) = memory.sample(batch_size)
        states_mb = np.array([each[0][0] for each in mb], ndmin=3)
        actions_mb = np.array([each[0][1] for each in mb])
        rewards_mb = np.array([each[0][2] for each in mb]) 
        next_states_mb = np.array([each[0][3] for each in mb], ndmin=3)

    return b_idx, np.stack(states_mb), np.stack(actions_mb), rewards_mb , episode_num


if __name__ == "__main__":
    env = gym.make('DeepRacer-v1')
    env = env.unwrapped
    # env = DeepRacerMultiDiscreteEnv()
    ################################
    #
    #   Initialize training hyperparameters
    #
    #################################

    state_size = [4,160, 120, 3] # Our input is a stack of 4 frames hence 160x120x3x4 (Width, height, channels*stack_size) 
    action_size = env.action_space.n # 10 possible actions: turn left, turn right, move forward
    print(str(env.action_space.n))
    print(str(env.observation_space))
    possible_actions = [x for x in range(action_size)]
    stack_size = 4 # Defines how many frames are stacked together

    ## TRAINING HYPERPARAMETERS
    learning_rate = 0.00001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=20,
    decay_rate=0.95,
    staircase=True)
    episodes = 5 # Total episodes for sampling

    batch_size = 512 # Each 1 is AN EPISODE 
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
    train_opt = tf.keras.optimizers.Adagrad(lr_schedule)
    memory = Memory(batch_size*int(episodes/2))

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
    # restore saved model if available
    checkpoint.restore(manager.latest_checkpoint)
    print("[INFO] START TRAINING")
    epoch = 1
    # allRewards=[]
    # mean_reward_total=[]
    # average_reward = []
    while training:
        # Gather training data
        tree_idx, states_mb, actions_mb, rewards_mb, nb_episodes_mb = make_batch(env,agent,train_opt,episodes,batch_size, memory, stacked_frames, stack_size, training, possible_actions,gamma,state_size,True)

        ### These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_mb)
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
        print("Epoch: ", epoch )
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        # print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        # print("Average Reward of all training: {}".format(average_reward_of_all_training))
        # print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

        # Feedforward, gradient and backpropagation
        loss_ = lambda : loss(agent,states_mb,actions_mb,rewards_mb,training=training)
        vars_ = lambda : agent.trainable_variables
        train_opt.minimize(loss_,vars_)
        # memory.batch_update(tree_idx,rewards_mb)
        print("Training Loss: {}".format(loss_()))


        # Save Model
        checkpoint.step.assign_add(1)
        if int(checkpoint.step) % 10 == 0:
            save_path = manager.save()

        epoch += 1
