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


class FeatureExtract(tf.keras.Model):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, state_size, learning_rate, name='featureExtract'):
        super(FeatureExtract, self).__init__()
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.model = tf.keras.Sequential()
        self.model.add(FPN3D())
        down = [downsample3D(32, [4, 4, 4], [4, 2, 2]),
                downsample3D(64, [1, 4, 4], [1, 2, 2]),
                downsample3D(128, [1, 4, 4], [1, 2, 2])]
        for layer in down:
            self.model.add(layer)
            self.model.add(tf.keras.layers.AveragePooling3D((1, 4, 4)))

        self.model.add(tf.keras.layers.Flatten())

    def call(self, inputs, training=None, mask=None):
        return self.model(input)


class Target(tf.keras.Model):
    """
        1. Feature Extract Layers
        2. Dense Connected Value Network
        3. Action Values
        4. Softmaxed Policy Gradient
    """

    def __init__(self, state_size, action_size, feature_extract, name='TargetDQN'):
        super(Target, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extract = feature_extract

        self.state = tf.keras.Input(shape=state_size)
        self.model = tf.keras.Sequential()
        self.advs = []
        for i in range(action_size):
            self.advs.append(layers.Dense(
                    kernel_initializer=initializers.GlorotUniform,
                    units=512,  # unwrap tuple
                    activation=None)(self.feature_extract(self.state)))

        self.value = layers.Dense(
                kernel_initializer=initializers.GlorotUniform,
                units=512,  # unwrap tuple
                activation=None)(self.feature_extract(self.state))
        self.aggregate = layers.concatenate(
                        [self.value, *self.advs],
                        axis=1)

        self.fc1 = layers.Dense(
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=initializers.GlorotUniform)

        self.model.add(self.aggregate, self.fc1)
        self.model = tf.keras.Model(inputs=[self.state],outputs=self.model)

    def call(self, state, training=None, mask=None):
        return self.model(state)

    def loss(self, state, next_state, reward):
        return tf.math.reduce_mean(
                tf.keras.losses.logcosh(
                        self.feature_extract(next_state),
                        self.call(state))
                * reward)


class Forward(tf.keras.Model):

    def __init__(self, state_size, action_size, feature_extract):
        super(Forward, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.action = tf.keras.Input(shape=self.action_size)
        self.state = tf.keras.Input(shape=self.state_size)
        self.model = tf.keras.Sequential()

        self.fc1 = layers.Dense(512,
                                activation=tf.nn.elu(),
                                kernel_initializer=initializers.GlorotNormal)
        self.connect = layers.concatenate(
                [self.fc1(self.action),
                 self.feature_extract(self.state)],
                axis=1)
        self.fc2 = layers.Dense(512,
                                activation=tf.nn.elu(),
                                kernel_initializer=initializers.GlorotUniform
                                )(self.connect)
        self.chain = [self.fc1,
                      self.connect,
                      self.fc2]

        for layer in self.chain:
            self.model.add(layer)

        self.model = tf.keras.models.Model(
                inputs=[self.state, self.action],
                outputs=self.model)

    def call(self, inputs, training=False, mask=None):
        state, action = inputs
        return self.model([state, action])

    def loss(self, state, action, next_state):
        prediction = self.call(state, action, training=True)
        actual = self.feature_extract(next_state)
        tf.keras.losses.logcosh(actual, prediction)


class Inverse(tf.keras.Model):
    def __init__(self, state_size, action_size, feature_extract):
        super(Inverse, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.state = tf.keras.Input(shape=self.state_size)
        self.next_state = tf.keras.Input(shape=self.state_size)
        self.state_diff = tf.keras.layers.subtract([self.feature_extract(s) for s in [self.next_state, self.state]])
        self.model = tf.keras.Sequential
        self.fc1 = layers.Dense(512,
                                activation=tf.nn.elu(),
                                kernel_initializer=initializers.GlorotNormal)

        self.fc2 = layers.Dense(int(512 / self.action_size),
                                activation=tf.nn.elu(),
                                kernel_initializer=initializers.GlorotNormal)

        self.fc3 = layers.Dense(self.action_size,
                                activation=tf.nn.elu(),
                                kernel_initializer=initializers.GlorotNormal)

        self.chain = [self.state_diff, self.fc1, self.fc2, self.fc3]
        for layer in self.chain:
            self.model.add(layer)
        self.model = tf.keras.models.Model(inputs=[self.state, self.next_state], outputs=[self.model])

    def call(self, inputs, training=None, mask=None):
        state, next_state = inputs
        return tf.nn.softmax(self.model(inputs=[state, next_state]))

    def loss(self, state, next_state, action):
        prediction = self.call(inputs=[state, next_state])
        return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=action))


class IntrinsicReward(tf.keras.Model):
    def __init__(self, state_size, action_size, feature_extract):
        super(IntrinsicReward, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.action = tf.keras.Input(shape=self.action_size)
        self.state = tf.keras.Input(shape=self.state_size)
        self.next_state = tf.keras.Input(shape=self.state_size)
        self.forward = Forward(self.state_size, self.action_size, self.feature_extract)
        self.inverse = Inverse(self.state_size, self.action_size, self.feature_extract)
        self.target = Target(self.state_size, self.feature_extract)

    def call(self,inputs,training=False,mask=None):
        state=inputs
        expected_state = self.target(state)
        expected_action = self.inverse([state,expected_state])
