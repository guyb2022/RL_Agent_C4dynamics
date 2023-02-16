import numpy as np
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
tf.get_logger().setLevel(logging.ERROR)

class C4dAgent:
    def __init__(self, state_size, action_size, agent_name):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.agent_name = agent_name

    def remember(self, state, action, reward, next_state, done):
        """
        Store the moves of an episode
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose the best action (action, amount)
        for the given state using an epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon and self.epsilon > 0:
            # Explore: choose a random action
            action = np.random.randint(self.action_size-1)
            return action
        else:
            # Exploit: choose the action with the highest action-value
            state = np.array(state).reshape((1, -1))
            q_values = self.agent_model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            return action

    def replay(self, batch_size):
        """
        Sample a batch of transitions from the memory and update the model.
        Parameters:
        batch_size (int): The number of samples to take from the memory.

        """
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        done = np.array([self.memory[i][4] for i in minibatch])
        # Create callbacks for early stopping and model checkpointing
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        checkpointer = ModelCheckpoint(filepath=self.agent_best_weights,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')
        # Update the model for each sample in the minibatch
        for i in range(0, batch_size):
            target = rewards[i]
            if not done[i]:
                action = int(actions[i])
                # Compute the target action-value using the SARSA algorithm
                next_action = self.act(next_states[i])
                try:
                    target += self.gamma * self.agent_model.predict(next_states[i], verbose=0)[0][next_action]
                except Exception as e:
                    print("An error occurred in REPLAY:", e)
                    target += 0
                target_f = np.zeros((1, self.action_size))
                target_f = self.agent_model.predict(states[i], verbose=0)
                action_one_hot = to_categorical(action, num_classes=self.action_size)
                target_f[0] = action_one_hot * target

                self.agent_model.fit(states[i], target_f, epochs=1, verbose=0,
                                    callbacks=[early_stop, checkpointer])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _build_agent_model(self):
        """
        Load the model if exist otherwise
        create the neural network model
        """
        with tf.device('/device:GPU:0'):
            if os.path.exists(self.agent_model_file):
                print("Found model file, loading...........................")
                return self.load_agent_model()
            else:
                # if model doesnt found create a new one
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(64, input_dim=self.state_size,
                                                activation='relu'))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(64, activation='relu'))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(64, activation='relu'))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
                model.compile(loss='mse',
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
                return model

    def load_agent_model(self):
        """
        Load an agent from memory if exist
        Also load best known weights and compile
        """
        model = tf.keras.models.load_model(self.agent_model_file)
        if os.path.exists(self.agent_best_weights):
            # Load weights file
            model.load_weights(self.agent_best_weights)
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def load_best_weights(self):
        """
        Laod best known weights
        """
        # check if weights file exsist
        if os.path.exists(self.agent_best_weights):
            #print("Found best Agent weights, loading...................")
            # Load weights file
            self.agent_model.load_weights(self.agent_best_weights)

    def save(self):
        """
        Save the model to the given file.
        """
        self.agent_model.save(self.agent_model_file)

    def save_model_weights(self):
        """
        Save the model weights to the given file.
        """
        self.agent_model.save_weights(self.agent_best_weights)


