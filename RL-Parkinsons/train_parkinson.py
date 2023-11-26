import os
from dqn import TrainDQN
from data import get_train_test_val, load_csv
from tensorflow.keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

episodes = 100_000
warmup_steps = 170_000
memory_length = warmup_steps
batch_size = 32
collect_steps_per_episode = 2000
collect_every = 500

target_update_period = 800
target_update_tau = 1
n_step_update = 1

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]

learning_rate = 0.00025
gamma = 0.0
min_epsilon = 0.5
decay_episodes = episodes // 10

X_train, y_train, X_test, y_test = load_csv("./data/train_data.csv", "./data/test_data.csv", "status", [], normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, val_frac=0.2)

model = TrainDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, layers)
model.q_net.summary()
model.train(X_val, y_val, "F1")

print({k: round(v, 6) for k, v in stats.items()})
