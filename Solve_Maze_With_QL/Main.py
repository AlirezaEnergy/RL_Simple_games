import matplotlib.pyplot as plt
from mazeQL import MazeQL
import time
import pandas as pd

### create the maze
"""
    X: wall
    P: player
    E: end (where the final reward is)
"""

Maze =[
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
"XXXXXXXXXXXXXXXXXXXXXXXXX"]

# initialize the Q table
Q = []
for i in range(len(Maze)):
    for j in range(len(Maze[i])):
        Q.append([0,0,0,0])

### uncomment to use the pre-trained agent
# QDF1 = pd.read_excel('Q.xlsx')
# print(QDF1.columns)
# QDF1.drop('Unnamed: 0', inplace = True, axis = 1)
# Q = QDF1.to_numpy().tolist()

### initiate the episodes
ep = 0.05
delay = 0

t = []
for i in range(10):
    print(f"Episode: {str(i).zfill(3)}")
    start_time = time.time()
    Q = MazeQL(Maze, Q, ep, delay)
    t.append(time.time()-start_time)
    ep = ep*0.9995
    time.sleep(0.5)

### plot time required to complete an episode
plt.figure(figsize = (8,10), dpi = 150)
plt.plot(t)

### uncomment to save the Q-table
# QDF = pd.DataFrame(Q)
# QDF.to_excel('Q.xlsx')
# QDF1 = pd.read_excel('Q.xlsx')
# print(QDF1.columns)
# QDF1.drop('Unnamed: 0', inplace = True, axis = 1)
# Q1 = QDF1.to_numpy().tolist()

