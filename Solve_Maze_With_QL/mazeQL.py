import turtle
from pynput.keyboard import Key, Controller
import time
import numpy as np

def MazeQL(level_1,Q,ep,delay):
    keyboard = Controller()
    
    wn = turtle.Screen()
    wn.bgcolor("black")
    wn.title("A Maze Game")
    wn.setup(700,700)
    wn.tracer(0)
    
    # create Pen class
    class Pen(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape("square")
            self.color("white")
            self.penup()
            self.speed(0)
    
    class Player(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape("square")
            self.color("blue")
            self.penup()
            self.speed(0)
            self.gold = 0
    
        def go_up(self):
            move_to_x = player.xcor()
            move_to_y = player.ycor() + 24
            
            if (move_to_x, move_to_y) not in walls:
                self.goto(move_to_x, move_to_y)
                
        def go_down(self):
            move_to_x = player.xcor()
            move_to_y = player.ycor() - 24
            
            if (move_to_x, move_to_y) not in walls:
                self.goto(move_to_x, move_to_y)
            
        def go_left(self):
            move_to_x = player.xcor() - 24
            move_to_y = player.ycor()
            
            if (move_to_x, move_to_y) not in walls:
                self.goto(move_to_x, move_to_y)
            
        def go_right(self):
            move_to_x = player.xcor() + 24
            move_to_y = player.ycor()
            
            if (move_to_x, move_to_y) not in walls:
                self.goto(move_to_x, move_to_y)
            
        def is_collision(self, other):
            a = self.xcor() - other.xcor()
            b = self.ycor() - other.ycor()
            dist = ( a**2 + b**2 )**0.5
            
            if dist < 5:
                return True
            else:
                return False
    
    class Treasure(turtle.Turtle):
        def __init__(self, x, y):
            turtle.Turtle.__init__(self)
            self.shape("circle")
            self.color("gold")
            self.penup()
            self.speed(0)
            self.gold = 100
            self.goto(x,y)
        
        def destroy(self):
            self.goto(2000, 2000)
            self.hideturtle()
    
    levels = [""]
    
    # Add treatures list
    treasures = []
    
    # Add maze to maze list
    levels.append(level_1)
    
    # Create level setup function
    States = {}
    def setup_maze(level):
        counter = 0
        cordinates = []
        for y in range(len(level)):
            cordinates.append([])
            for x in range(len(level[y])):
                character = level[y][x]
                
                screen_x = -288 + (x * 24)
                screen_y = 288 - (y * 24)
                
                States[(screen_x, screen_y)] = counter
                cordinates[y].append(counter)
                
                if character == "X":
                    pen.goto(screen_x, screen_y)
                    pen.stamp()
                    walls.append((screen_x,screen_y))
                
                if character == "P":
                    player.goto(screen_x, screen_y)
                
                if character == "E":
                    treasures.append(Treasure(screen_x, screen_y))
                    FinalState = counter
                counter += 1
        return FinalState, cordinates
    
    def EGAS(QS,ep):
        NumActions = len(QS)                       # determine the number of available Actions in state S
        Actions = list(range(NumActions))          # list the available actions is state S
        Qmax = max(QS)                             # find "one of" greedy actions in state S
        NumGreedy = 0                              # initialize number of greedy actions in state S
        GreedyActions = []                         # create list of greedy actions in state S
        for i in range(len(QS)):                   # find all greedy actions in state S
            if QS[i] == Qmax:
                GreedyActions.append(i)
                NumGreedy = NumGreedy + 1
        
        NonGreedyActions = []                         # create list of non-greedy actions
        for i in range(len(Actions)):
            if Actions[i] not in GreedyActions:
                NonGreedyActions.append(Actions[i])
        NumNonGreedy = len(NonGreedyActions)
        
        rnd = np.random.rand()
        if rnd >= ep: # choose one of greedy actions with equal probability
            rndGreedy = np.random.randint(NumGreedy)
            a = GreedyActions[rndGreedy]
        else: # choose one of non-greedy actions
            if NumActions == NumGreedy: # if all actions are greedy (this can happen in the beginning), choose one of them
                rndGreedy = np.random.randint(NumGreedy)
                a = Actions[rndGreedy]
            else: # if there are non-greedy actions, choose one of them with equal probability
                rndNonGreedy = np.random.randint(NumNonGreedy)
                a = NonGreedyActions[rndNonGreedy]
        return a
    
    
    # defining the keypress simulator function
    def PressKey(action):
        if action == 0:
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            decoded_action = "Left"
            
        elif action == 1:
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            decoded_action = "Right"
            
        elif action == 2:
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            decoded_action = "Down"
            
        else:
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            decoded_action = "Up"
        return decoded_action
    
    # Create an instance of pen class
    pen = Pen()
    player = Player()
    
    # Create wall cordinates list
    walls = []
    
    # Setup the level
    FinalState, cordinates = setup_maze(levels[1])
    
    # Keyboard Binding
    turtle.listen()
    turtle.onkeypress(player.go_left, "Left")
    turtle.onkeypress(player.go_right, "Right")
    turtle.onkeypress(player.go_up, "Up")
    turtle.onkeypress(player.go_down, "Down")
    
    # Turn off screen update
    #wn.tracer(0)
    
    # Main game loop
    KeepGoing = True
    while KeepGoing:
        
        for treasure in treasures:
            if player.is_collision(treasure):
                player.gold += treasure.gold
                print(f"Player gold: {player.gold}")
                treasure.destroy()
                treasures.remove(treasure)
        
        S = States[(player.xcor(), player.ycor())] #gives back an integer as the state of the agent
        
        ### renadom agent
        #action = randint(0,3)
        
        ### QL agent
        a = EGAS(Q[S],ep)
        decoded_action = PressKey(a) # get a number and do the appropriate action 
                                     #(the returned value is just for the purpose of ptinting)
        
        wn.update()
        time.sleep(delay)
        
        Sp = States[(player.xcor(), player.ycor())] #gives back an integer as the state of the agent
        
        if Sp != FinalState:
            R = -1
        else:
            R = 10
        
        #print(f'S: {S}, a: {decoded_action}, Sp: {Sp}')
        #print(f'Q(s,a) before update: {Q[S][a]}')    
        
        Q[S][a] = Q[S][a] + 0.1 * (R + 1 * max(Q[Sp]) - Q[S][a])
        
        #print(f'Q(s,a) after update: {Q[S][a]}')
        
        if Sp == FinalState:
            #print('The agent successfully reached to the end of the maze!!')
            KeepGoing = False
            #wn.bye()
            #wn.exitonclick()
            try:
                wn.bye()
                turtle.TurtleScreen._RUNNING = True
            except turtle.Terminator:
                wn.bye()
            #time.sleep(delay)
            #print('end')
    return Q
        





















