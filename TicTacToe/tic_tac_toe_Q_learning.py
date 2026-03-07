import numpy as np
import json

###################################################################
def win_check(arr, char):
    '''
    reference: https://www.geeksforgeeks.org/validity-of-a-given-tic-tac-toe-board-configuration/
    Returns true if char wins. Char can be either
    1 or 2, the game starts with 1
    arr is the current board

    X => 1
    O => 2

    input: win_check(arr,1)
    output: True or False
    '''
    # Check all possible winning combinations
    matches = [[0, 1, 2],[3, 4, 5],[6, 7, 8],[0, 3, 6],[1, 4, 7],[2, 5, 8],[0, 4, 8],[2, 4, 6]]

    for i in range(8):
        if(arr[matches[i][0]] == char and
            arr[matches[i][1]] == char and
            arr[matches[i][2]] == char):
            return True
    return False
###################################################################
def is_valid(arr):
    '''
    Returns true if the board is reachable when either player can go first.
    Either player may start, so counts can differ by at most 1 in either direction.

    input: is_valid(arr)
    output: True or False
    '''
    xcount = arr.count(1)
    ocount = arr.count(2)

    # Turns alternate and either player can start, so counts differ by at most 1
    if abs(xcount - ocount) > 1:
        return False

    x_wins = win_check(arr, 1)
    o_wins = win_check(arr, 2)

    # Both players cannot win simultaneously
    if x_wins and o_wins:
        return False

    # The winner must have made the last move, so their count >= opponent's count
    if x_wins and xcount < ocount:
        return False
    if o_wins and ocount < xcount:
        return False

    return True
###################################################################
def EGAS(QS, ep):
    '''
        epsilon greedy action selection

        input: EGAS([0,1,-1],0.1)
        output: 0 or 1 or 2
    '''
    NumActions = len(QS)
    Actions = list(range(NumActions))
    Qmax = max(QS)

    GreedyActions = [i for i in range(NumActions) if QS[i] == Qmax]
    NonGreedyActions = [i for i in Actions if i not in GreedyActions]
    NumGreedy = len(GreedyActions)
    NumNonGreedy = len(NonGreedyActions)

    rnd = np.random.rand()
    if rnd >= ep:  # choose a greedy action
        return GreedyActions[np.random.randint(NumGreedy)]
    else:          # choose a non-greedy action
        if NumNonGreedy == 0:  # all actions are greedy (e.g. all Q=0 at init)
            return GreedyActions[np.random.randint(NumGreedy)]
        return NonGreedyActions[np.random.randint(NumNonGreedy)]
###################################################################
def get_possible_actions(board):
    '''
        input: get_possible_actions([0,1,0,2,0,0,0,0,0])
        output: [0,2,4,5,6,7,8]
    '''
    return [i for i, j in enumerate(board) if j == 0]
###################################################################
def get_current_state(board):
    '''
        O(1) lookup using a pre-built dictionary.
        input: get_current_state([0,0,0,0,0,0,0,0,0])
        output: 0
    '''
    return board_to_index[tuple(board)]
###################################################################
def create_valid_boards():
    '''
        reference: https://stackoverflow.com/questions/61508393/generate-all-possible-board-positions-of-tic-tac-toe
        input: -
        output: a list of valid boards
    '''
    boards = []
    for i in range(19683):
        c = i
        temp_board = []
        for _ in range(9):
            temp_board.append(c % 3)
            c //= 3
        if is_valid(temp_board):
            boards.append(temp_board)
    return boards
###################################################################
def tied(board, possible_actions):
    '''
        Checks if the game is tied or not
    '''
    return possible_actions == []
###################################################################
def Reward(board, winning_reward, punishment):
    end = False
    player_1_won = win_check(board, 1)
    player_2_won = win_check(board, 2)
    if player_1_won:
        r1, r2 = winning_reward, punishment
        end = True
    elif player_2_won:
        r2, r1 = winning_reward, punishment
        end = True
    else:
        r1, r2 = punishment, punishment
    return r1, r2, end



boards = create_valid_boards()
board_to_index = {tuple(b): i for i, b in enumerate(boards)}



Q1 = np.zeros((len(boards), 9))  # Q table for player 1
Q2 = np.zeros((len(boards), 9))  # Q table for player 2

ep_start         = 1.00           # starting exploration rate
ep_end           = 0.005          # final exploration rate
ep_time_fraction = 0.9            # fraction of episodes over which epsilon decays linearly

alpha_start          = 0.5        # starting learning rate
alpha_end            = 0.001      # final learning rate
alpha_time_fraction  = 0.95       # fraction of episodes over which alpha decays linearly

gamma = 1                         # discount factor

step_reward    = 0                # per-step reward (no cost for game length)
loss_reward    = -1               # reward assigned to loser's last move
winning_reward = 2                # reward for winning
tie_reward     = 1                # reward for a tied game (equal to winning)

max_episode = 1000000
print_every = 5000                # print summary every N episodes



wins = {1: 0, 2: 0, 'tie': 0}

for episode in range(max_episode):
    # Linear epsilon schedule (Stable Baselines style)
    progress = min(episode / (ep_time_fraction * max_episode), 1.0)
    ep = ep_start + (ep_end - ep_start) * progress

    # Linear alpha schedule — high early (fast learning), low late (stable convergence)
    alpha_progress = min(episode / (alpha_time_fraction * max_episode), 1.0)
    alpha = alpha_start + (alpha_end - alpha_start) * alpha_progress

    board = [0] * 9
    current_player = int(np.random.choice([1, 2]))

    # pending stores each player's (S, a) that hasn't been Q-updated yet.
    # We delay each player's non-terminal update by one full round so the
    # bootstrap uses Spp (the state after the opponent also moves) — the state
    # the current player actually acts from next, not Sp which the opponent acts from.
    pending = {1: None, 2: None}

    while True:
        S = get_current_state(board)
        possible_actions = get_possible_actions(board)
        opponent = 2 if current_player == 1 else 1

        # --- Action selection ---
        Q = Q1 if current_player == 1 else Q2
        QpS = np.array([Q[S][a] for a in possible_actions])
        a = possible_actions[EGAS(QpS, ep)]

        # --- Apply action ---
        board[a] = current_player
        Sp = get_current_state(board)
        possible_actions_sp = get_possible_actions(board)
        _, _, end = Reward(board, winning_reward, loss_reward)

        if end:
            # Winner update — terminal, no bootstrap
            Qw = Q1 if current_player == 1 else Q2
            Qw[S][a] += alpha * (winning_reward - Qw[S][a])

            # Loser's pending move now terminates with punishment
            if pending[opponent] is not None:
                pS, pa = pending[opponent]
                Ql = Q1 if opponent == 1 else Q2
                Ql[pS][pa] += alpha * (loss_reward - Ql[pS][pa])

            wins[current_player] += 1
            break

        elif tied(board, possible_actions_sp):
            # Current player's move caused the tie
            Qc = Q1 if current_player == 1 else Q2
            Qc[S][a] += alpha * (tie_reward - Qc[S][a])

            # Opponent's pending move also gets tie reward
            if pending[opponent] is not None:
                pS, pa = pending[opponent]
                Qo = Q1 if opponent == 1 else Q2
                Qo[pS][pa] += alpha * (tie_reward - Qo[pS][pa])

            wins['tie'] += 1
            break

        else:
            # Non-terminal: apply the opponent's pending update now, bootstrapping
            # from Sp — the state the opponent acts from next (correct two-step lookahead)
            if pending[opponent] is not None:
                pS, pa = pending[opponent]
                Qo = Q1 if opponent == 1 else Q2
                max_Q_Sp = max(Qo[Sp][a_sp] for a_sp in possible_actions_sp)
                Qo[pS][pa] += alpha * (step_reward + gamma * max_Q_Sp - Qo[pS][pa])

            # Store current player's move — will be updated next time they act
            pending[current_player] = (S, a)

        current_player = opponent

    if (episode + 1) % print_every == 0:
        start = episode + 1 - print_every
        print(f"Episodes {start+1:>5}-{episode+1:<5} | P1 wins: {wins[1]:>4} | P2 wins: {wins[2]:>4} | Ties: {wins['tie']:>4} | ep: {ep:.4f} | alpha: {alpha:.4f}")
        wins = {1: 0, 2: 0, 'tie': 0}

# ---------------------------------------------------------------------------
# Export Q-tables as JS files for the browser UI
# ---------------------------------------------------------------------------

def export_q_table(Q, boards, var_name, filename):
    q_dict = {
        ','.join(map(str, b)): [round(float(v), 6) for v in Q[i]]
        for i, b in enumerate(boards)
    }
    with open(filename, 'w') as f:
        f.write(f'const {var_name} = ')
        json.dump(q_dict, f)
        f.write(';')
    print(f"Saved {filename}")

export_q_table(Q1, boards, 'Q1_TABLE', 'q1.js')
export_q_table(Q2, boards, 'Q2_TABLE', 'q2.js')