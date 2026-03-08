import turtle
import time
import numpy as np


def MazeQL(maze, Q, ep, delay, alpha=0.1, gamma=1.0):
    """
    Run one episode of Q-learning on the maze.
    Returns the updated Q-table.

    Maze legend:  X = wall,  P = player start,  E = goal
    Actions:      0=left, 1=right, 2=down, 3=up
    alpha:        learning rate
    gamma:        discount factor
    """

    CELL = 24  # pixel size of each maze cell

    # --- Turtle window setup ---
    wn = turtle.Screen()
    wn.bgcolor("black")
    wn.title("Maze RL")
    wn.setup(700, 700)
    wn.tracer(0)  # disable auto-update for speed

    # --- Turtle classes ---
    class Pen(turtle.Turtle):
        """Used to stamp walls onto the screen."""
        def __init__(self):
            super().__init__()
            self.shape("square")
            self.color("white")
            self.penup()
            self.speed(0)

    class Player(turtle.Turtle):
        def __init__(self, x, y):
            super().__init__()
            self.shape("square")
            self.color("blue")
            self.penup()
            self.speed(0)
            self.goto(x, y)

        def move(self, dx, dy):
            """Move by (dx, dy) pixels, but only if the destination is not a wall."""
            new_x = round(self.xcor()) + dx
            new_y = round(self.ycor()) + dy
            if (new_x, new_y) not in walls:
                self.goto(new_x, new_y)

    # --- Parse the maze ---
    walls = set()           # wall positions as (screen_x, screen_y)
    state_map = {}          # (screen_x, screen_y) -> state index
    player_start = None
    goal_pos = None
    goal_state = None

    pen = Pen()
    state_idx = 0
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            sx = -288 + col * CELL
            sy =  288 - row * CELL
            state_map[(sx, sy)] = state_idx
            char = maze[row][col]

            if char == "X":
                walls.add((sx, sy))
                pen.goto(sx, sy)
                pen.stamp()

            elif char == "P":
                player_start = (sx, sy)

            elif char == "E":
                goal_pos = (sx, sy)
                goal_state = state_idx

            state_idx += 1

    pen.hideturtle()

    # Draw the goal marker
    goal_marker = turtle.Turtle()
    goal_marker.shape("circle")
    goal_marker.color("gold")
    goal_marker.penup()
    goal_marker.goto(*goal_pos)

    # Create the player
    player = Player(*player_start)

    # Map action index to (dx, dy) movement
    ACTIONS = {
        0: (-CELL, 0),   # left
        1: ( CELL, 0),   # right
        2: (0, -CELL),   # down
        3: (0,  CELL),   # up
    }

    def pick_action(q_values):
        """Epsilon-greedy: explore randomly with probability ep, else pick best action."""
        if np.random.rand() < ep:
            return np.random.randint(len(ACTIONS))
        return int(np.argmax(q_values))

    # --- Q-learning loop (one episode) ---
    while True:
        s = state_map[(round(player.xcor()), round(player.ycor()))]

        a = pick_action(Q[s])
        dx, dy = ACTIONS[a]
        player.move(dx, dy)

        wn.update()
        time.sleep(delay)

        sp = state_map[(round(player.xcor()), round(player.ycor()))]

        R = 10 if sp == goal_state else -1

        # Q-learning update
        Q[s][a] = Q[s][a] + alpha * (R + gamma * max(Q[sp]) - Q[s][a])

        if sp == goal_state:
            break

    # Close the window cleanly so the next episode can reopen it
    try:
        wn.bye()
        turtle.TurtleScreen._RUNNING = True
    except turtle.Terminator:
        pass

    return Q
