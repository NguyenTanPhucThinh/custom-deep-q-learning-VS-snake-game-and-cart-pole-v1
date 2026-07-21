import random
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MSE

class DQNAgent:
    def __init__(self, state_size, num_actions, memory_size=100_000, gamma=0.995, alpha=1e-3):
        self.state_size = state_size
        self.num_actions = num_actions
        self.memory_buffer = deque(maxlen=memory_size)
        self.gamma = gamma
        self.alpha = alpha
        
        self.q_network = self._build_model()
        self.target_q_network = self._build_model()
        self.target_q_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.state_size),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=self.num_actions, activation='linear')
        ])
        return model
        
    def store_experience(self, state, action, reward, next_state, done):
        self.memory_buffer.append(self.experience(state, action, reward, next_state, done))
        
    def get_action(self, state, epsilon):
        rand = np.random.random()
        if rand < epsilon:
            # EXPLORE
            action = np.random.randint(0, self.num_actions)
        else:
            # EXPLOIT
            state_input = np.expand_dims(state, axis=0)
            q_values = self.q_network(state_input)
            action = np.argmax(q_values.numpy().squeeze())
        return action
        
    def get_experiences(self, batch_size=64):
        # We limit batch size to min(batch_size, len(memory))
        batch_size = min(batch_size, len(self.memory_buffer))
        idx = np.random.choice(len(self.memory_buffer), size=batch_size, replace=False)
        batch = [self.memory_buffer[i] for i in idx]

        states      = np.array([b.state      for b in batch], dtype=np.float32)
        actions     = np.array([b.action     for b in batch], dtype=np.int32)
        rewards     = np.array([b.reward     for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones       = np.array([b.done       for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones
        
    @tf.function
    def compute_loss(self, experiences, gamma):
        states, actions, rewards, next_states, done_vals = experiences
        max_qsa = tf.reduce_max(self.target_q_network(next_states), axis=-1)
        y_targets = rewards + max_qsa * gamma * (1 - done_vals)
        
        q_values = self.q_network(states)
        indices = tf.stack([tf.range(tf.shape(q_values)[0]), tf.cast(actions, tf.int32)], axis=1)
        q_values_selected = tf.gather_nd(q_values, indices)
        
        loss = MSE(y_targets, q_values_selected)
        return loss

    @tf.function
    def agent_learn(self, experiences, tau=0.001):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, self.gamma)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # update target network
        self.update_target_network_soft(tau)
        return loss
        
    def update_target_network_soft(self, tau=0.001):
        for target_weights, q_net_weights in zip(self.target_q_network.weights, self.q_network.weights):
            target_weights.assign(tau * q_net_weights + (1.0 - tau) * target_weights)

    def decay_epsilon(self, epsilon, min_eps=0.01, decay_rate=0.995):
        return max(min_eps, epsilon * decay_rate)

    def save_model(self, path):
        self.q_network.save(path)

    def load_model(self, path):
        self.q_network = tf.keras.models.load_model(path)
        # Re-build target network
        self.target_q_network.set_weights(self.q_network.get_weights())
