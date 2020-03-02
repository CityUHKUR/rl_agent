# Import simulation Env


# Import training required packags
import tensorflow as tf  # Deep Learning library
from tensorflow.keras import layers, initializers
import numpy as np  # Handle matrices

from rocket.ignite.layers import downsample3D, upsample3D
from rocket.ignite.loss import cross_entropy_loss
from rocket.ignite.models import FPN3D
from rocket.images.preprocess import stack_frames
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

    def __init__(self, state_size, name='featureExtract'):
        super(FeatureExtract, self).__init__()
        self.state_size = state_size

        self.model = tf.keras.Sequential()
        self.model.add(FPN3D())
        down_stack = [downsample3D(32, [4, 4, 4], [4, 2, 2]),
                downsample3D(64, [1, 4, 4], [1, 2, 2]),
                downsample3D(128, [1, 4, 4], [1, 2, 2])]
        pooling = [tf.keras.layers.AveragePooling3D((1, 4, 4)),
                   tf.keras.layers.AveragePooling3D((1, 4, 4)),
                   None]
        for (down,pool) in zip(down_stack,pooling):
            self.model.add(down)
            if pool is not None:
                self.model.add(tf.keras.layers.AveragePooling3D((1, 4, 4)))
        self.model.add(tf.keras.layers.Flatten())

    def call(self, states, training=None, mask=None):
        return self.model(states, training=training, mask=mask)


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

        self.model = self.fc1(self.aggregate)

        self.model = tf.keras.Model(inputs=[self.state], outputs=self.model)

    def call(self, states, training=None, mask=None):
        return self.model(states, training=training, mask=mask)

    def loss(self, inputs):
        states, _, rewards, next_states = inputs
        return tf.math.reduce_mean(
                tf.keras.losses.logcosh(
                        self.feature_extract(next_states),
                        self.call(states))
                * rewards)

    def minimize(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            _loss = self.loss(inputs)
            grads = tape.gradient(_loss, self.model.trainable_variables)
        optimizer.apply_gradients(grads, self.model.trainable_variables)


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
                                activation=layers.ELU(),
                                kernel_initializer=initializers.GlorotNormal)
        self.connect = layers.concatenate(
                [self.fc1(self.action),
                 self.feature_extract(self.state)],
                axis=1)
        self.fc2 = layers.Dense(512,
                                activation=layers.ELU(),
                                kernel_initializer=initializers.GlorotUniform
                                )(self.connect)

        self.model = tf.keras.Model(
                inputs=[self.state, self.action],
                outputs=self.fc2)

    def call(self, inputs, training=False, mask=None):
        states, actions, _, _ = inputs
        return self.model([states, actions], training=training, mask=mask)

    def loss(self, inputs):
        states, actions, _, next_states = inputs
        prediction = self.call([states, actions, None, None], training=True)
        actual = self.feature_extract(next_states)
        return tf.keras.losses.logcosh(actual, prediction)

    def minimize(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            _loss = self.loss(inputs)
            grads = tape.gradient(_loss, self.model.trainable_variables)
        optimizer.apply_gradients(grads, self.model.trainable_variables)


class Inverse(tf.keras.Model):
    def __init__(self, state_size, action_size, feature_extract):
        super(Inverse, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extract = feature_extract
        self.feature_shape = (512)
        self.state = tf.keras.Input(shape=self.feature_shape)
        self.next_state = tf.keras.Input(shape=self.feature_shape)
        self.state_diff = tf.keras.layers.subtract([self.next_state, self.state])
        self.model = self.state_diff
        self.fc1 = layers.Dense(512,
                                activation=layers.ELU(),
                                kernel_initializer=initializers.GlorotNormal)

        self.fc2 = layers.Dense(int(512 / self.action_size),
                                activation=layers.ELU(),
                                kernel_initializer=initializers.GlorotNormal)

        self.fc3 = layers.Dense(self.action_size,
                                activation=layers.ELU(),
                                kernel_initializer=initializers.GlorotNormal)

        self.chain = [self.fc1, self.fc2, self.fc3]
        for layer in self.chain:
            self.model = layer(self.model)
        self.model = tf.keras.models.Model(inputs=[self.state, self.next_state], outputs=[self.model])

    def call(self, inputs, training=None, mask=None):
        states, _, _, next_states = inputs
        return tf.nn.softmax(self.model(inputs=[states, next_states], training=training, mask=mask))

    def loss(self, inputs):
        states, actions, _, next_states = inputs
        prediction = self.call(inputs=[self.feature_extract(states), actions, None, self.feature_extract(next_states)])
        return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=actions))

    def minimize(self, inputs, optimizer):
        states, actions, _, next_states = inputs
        with tf.GradientTape() as tape:
            _loss = self.loss(inputs)
            grads = tape.gradient(_loss, self.model.trainable_variables)
        optimizer.apply_gradients(grads, self.model.trainable_variables)


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
        self.target = Target(self.state_size, self.action_size, self.feature_extract)

    def call(self, inputs, training=False, mask=None):
        if len(inputs) >= 4:
            states, _, _, _ = inputs
        else:
            states = inputs
        expected_state = self.target(states, training=training, mask=mask)
        expected_action = self.inverse([self.feature_extract(states), None, None, expected_state], training=training, mask=mask)
        return expected_action.numpy()

    def loss(self, inputs):
        state, actions, rewards, next_states = inputs
        curiosity = self.forward.loss(inputs)
        return tf.math.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.call(inputs, training=True),
                        labels=actions) *
                (rewards + curiosity))

    # def updates(self):
    #     self.target.set_weights(self.driver.get_weights())

    def minimize(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            _loss = self.loss(inputs)
            grads = tape.gradient(_loss,
                                  [self.forward.trainable_variables, self.inverse.trainable_variables, self.target])
        optimizer.apply_gradients(grads,
                                  [self.forward.trainable_variables, self.inverse.trainable_variables, self.target])
        models = [self.forward, self.inverse, self.target]
        for model in models:
            model.minimize(inputs, optimizer)
