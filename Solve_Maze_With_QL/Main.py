import json
import matplotlib.pyplot as plt
import time
from mazeQL import MazeQL

# Maze layout:  X = wall,  P = player start,  E = goal
MAZE = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X PXXXXXX          XXXXXX",
    "X XXXXXXX      XX  XXXXXX",
    "X      XX      XX  XXXXXX",
    "X      XX      XX    XXXX",
    "XXXXX  XX XXX  XXX  XXXXX",
    "XXXXX  XX XXX  XXX  XXXXX",
    "XXXXX  XX XXX  XXX  XXXXX",
    "XXXXX     XXX  XXX  XXXXX",
    "XXXXX     XXX  XXX  XXXXX",
    "XXXXX     XXX  XXXXXXXXXX",
    "XXXXXXXXX  XX          XX",
    "XXXXXXX    XX           X",
    "XXXXXXXX  XXXXXXXXXXXXX X",
    "XXX                 XXX X",
    "XXX                 XXX X",
    "XXX  XXXXXXXXXXXX   XXX X",
    "XXX  XX   XXXXXXX   XXXXX",
    "XXX  XX   XXXXXXX   XXXXX",
    "XXX  XX  EXXX     XXXXXXX",
    "X    XX XXXXX     XXXXXXX",
    "X    XX XXXXXXXX  XXXXXXX",
    "X       XXXXXXX   XXXXXXX",
    "XXX     XXXXXXXX  XXXXXXX",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]

# Q-table: one entry per maze cell, each entry is [Q(left), Q(right), Q(down), Q(up)]
num_states = sum(len(row) for row in MAZE)
Q = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_states)]

# Uncomment to load a pre-trained Q-table:
# with open("Q.json") as f:
#     Q = json.load(f)

NUM_EPISODES         = 100   # total episodes to train on
ep_start             = 1     # exploration rate at the start
ep_final             = 0.01  # exploration rate at the end
exploration_fraction = 0.9   # fraction of episodes over which ep decays linearly
alpha                = 0.2   # learning rate
gamma                = 1.0   # discount factor
step_delay           = 0.01  # pause between agent steps (seconds) — for visualization
episode_pause        = 0.1   # pause between episodes (seconds) — avoids flickering windows
random_start         = True  # if True, agent starts at a random walkable cell each episode

decay_episodes = int(NUM_EPISODES * exploration_fraction)
episode_times = []

for episode in range(NUM_EPISODES):
    # Linear decay: ep goes from ep_start to ep_final over the first decay_episodes,
    # then stays at ep_final for the remaining episodes.
    if episode < decay_episodes:
        ep = ep_start + (ep_final - ep_start) * episode / decay_episodes
    else:
        ep = ep_final

    print(f"Episode {episode + 1}/{NUM_EPISODES}  (ep={ep:.4f})")
    start = time.time()
    Q = MazeQL(MAZE, Q, ep, delay=step_delay, alpha=alpha, gamma=gamma, random_start=random_start)
    episode_times.append(time.time() - start)
    time.sleep(episode_pause)

# Uncomment to save the trained Q-table:
with open("Q.json", "w") as f:
    json.dump(Q, f)

plt.figure(figsize=(8, 5))
plt.plot(episode_times, marker="o")
plt.xlabel("Episode")
plt.ylabel("Time (seconds)")
plt.title("Time to complete each episode")
plt.tight_layout()
plt.show()
