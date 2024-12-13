import numpy as np
import random
import copy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 定数の定義
BOARD_SIZE = 8
BLACK = 1
WHITE = -1
EMPTY = 0
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2  # ε-greedy法の探索率

class Othello:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.board[3][3], self.board[4][4] = WHITE, WHITE
        self.board[3][4], self.board[4][3] = BLACK, BLACK
        self.current_player = BLACK

    def get_valid_moves(self):
        valid_moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y, x] == EMPTY and self.can_flip(x, y):
                    valid_moves.append((x, y))
        print(f"合法手: {valid_moves}")
        return valid_moves

    def can_flip(self, x, y):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            flip_found = False
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == -self.current_player:
                flip_found = True
                nx += dx
                ny += dy
            if flip_found and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == self.current_player:
                return True
        return False

    def make_move(self, x, y):
        if not self.can_flip(x, y):
            return False
        self.board[y, x] = self.current_player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            flips = []
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == -self.current_player:
                flips.append((nx, ny))
                nx += dx
                ny += dy
            if flips and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == self.current_player:
                for fx, fy in flips:
                    self.board[fy, fx] = self.current_player
        self.current_player = -self.current_player
        return True

    def is_game_over(self):
        current_valid_moves = self.get_valid_moves()
        self.current_player = -self.current_player  # 相手に切り替え
        opponent_valid_moves = self.get_valid_moves()
        self.current_player = -self.current_player  # 元に戻す
        return len(current_valid_moves) == 0 and len(opponent_valid_moves) == 0

    def get_winner(self):
        black_score = np.sum(self.board == BLACK)
        white_score = np.sum(self.board == WHITE)
        return BLACK if black_score > white_score else WHITE if white_score > black_score else 0

    def print_board(self):
        for row in self.board:
            print(" ".join([f" {'●' if cell == BLACK else '○' if cell == WHITE else '・'} " for cell in row]))
        print()  # 空行を追加して見やすく

# DQNAgentなどのクラスの定義はそのまま
class DQNAgent:
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(BOARD_SIZE * BOARD_SIZE, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.target_model = copy.deepcopy(self.model)
        self.memory = []
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
    
    def get_state(self, board):
        return np.array(board)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.get_valid_moves(state))
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        valid_q_values = [q for i, q in enumerate(q_values[0]) if self.is_legal_move(state, i)]
        return np.argmax(valid_q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        print(f"メモリサイズ: {len(self.memory)}")
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)# この行のインデントを調整
        targets[range(self.batch_size), actions] = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)
        self.model.fit(states, targets, epochs=1, verbose=0)
        print(f"入力の形状: {states.shape}")
        print(f"ターゲットの形状: {targets.shape}")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def choose_action(self, state, valid_moves):
        if np.random.rand() < self.epsilon:  # ε-greedyでランダムな行動を選択
            return random.choice(valid_moves)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        # 合法手のみを考慮して最大値の行動を選択
        valid_q_values = [(q_values[move[1] * BOARD_SIZE + move[0]], move) for move in valid_moves]
        return max(valid_q_values, key=lambda x: x[0])[1]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
# 強化学習による学習
def train_agent(episodes=1):
    agent = DQNAgent()
    for episode in range(episodes):
        print(f"エピソード {episode}/{episodes} 開始")
        game = Othello()
        state = agent.get_state(game.board)
        while not game.is_game_over():
            print(f"現在のターン: {game.current_player}, ボード:")
            game.print_board()  # 盤面を表示
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = -game.current_player
                continue
            action = agent.choose_action(state, valid_moves)
            game.make_move(*action)
            reward = 1 if game.is_game_over() and game.get_winner() == game.current_player else 0
            next_state = agent.get_state(game.board)
            done = game.is_game_over()
            agent.remember(state, action[1] * BOARD_SIZE + action[0], reward, next_state, done)
            state = next_state
        agent.replay()
        if episode % 10 == 0:
            agent.update_target_model()
        if episode % 100 == 0:
            print(f"エピソード {episode}/{episodes} 完了")
    return agent

# 実行
if __name__ == "__main__":
    print("学習開始！")
    trained_agent = train_agent()
    print("学習完了！")


# import numpy as np
# import random
# import copy
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten

# # 定数の定義
# BOARD_SIZE = 8
# BLACK = 1
# WHITE = -1
# EMPTY = 0
# LEARNING_RATE = 0.1
# DISCOUNT_FACTOR = 0.9
# EPSILON = 0.2  # ε-greedy法の探索率

# class Othello:
#     def __init__(self):
#         self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
#         self.board[3][3], self.board[4][4] = WHITE, WHITE
#         self.board[3][4], self.board[4][3] = BLACK, BLACK
#         self.current_player = BLACK

#     def get_valid_moves(self):
#         valid_moves = []
#         for y in range(BOARD_SIZE):
#             for x in range(BOARD_SIZE):
#                 if self.board[y, x] == EMPTY and self.can_flip(x, y):
#                     valid_moves.append((x, y))
#         print(f"合法手: {valid_moves}")
#         return valid_moves

#     def can_flip(self, x, y):
#         directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             flip_found = False
#             while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == -self.current_player:
#                 flip_found = True
#                 nx += dx
#                 ny += dy
#             if flip_found and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == self.current_player:
#                 return True
#         return False

#     def make_move(self, x, y):
#         if not self.can_flip(x, y):
#             return False
#         self.board[y, x] = self.current_player
#         directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             flips = []
#             while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == -self.current_player:
#                 flips.append((nx, ny))
#                 nx += dx
#                 ny += dy
#             if flips and 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny, nx] == self.current_player:
#                 for fx, fy in flips:
#                     self.board[fy, fx] = self.current_player
#         self.current_player = -self.current_player
#         return True

#     def is_game_over(self):
#         # 現在のプレイヤーと相手プレイヤーの合法手を確認
#         current_valid_moves = self.get_valid_moves()
#         self.current_player = -self.current_player  # 相手に切り替え
#         opponent_valid_moves = self.get_valid_moves()
#         self.current_player = -self.current_player  # 元に戻す
#         # 両者に合法手がなければゲーム終了
#         return len(current_valid_moves) == 0 and len(opponent_valid_moves) == 0


#     def get_winner(self):
#         black_score = np.sum(self.board == BLACK)
#         white_score = np.sum(self.board == WHITE)
#         return BLACK if black_score > white_score else WHITE if white_score > black_score else 0

# # class QLearningAgent:
# #     def __init__(self):
# #         self.q_table = {}

# #     def get_state(self, board):
# #         return str(board.flatten())

# #     def choose_action(self, state, valid_moves):
# #         if random.random() < EPSILON:
# #             return random.choice(valid_moves)
# #         q_values = [self.q_table.get((state, move), 0) for move in valid_moves]
# #         return valid_moves[np.argmax(q_values)]

# #     def update_q_table(self, state, action, reward, next_state, next_valid_moves):
# #         max_q_next = max([self.q_table.get((next_state, move), 0) for move in next_valid_moves], default=0)
# #         old_value = self.q_table.get((state, action), 0)
# #         self.q_table[(state, action)] = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q_next - old_value)

# # # 強化学習による学習
# # def train_agent(episodes=1000):
# #     agent = QLearningAgent()
# #     for episode in range(episodes):
# #         print(f"エピソード {episode}/{episodes} ")
# #         game = Othello()
# #         state = agent.get_state(game.board)
# #         while not game.is_game_over():
# #             valid_moves = game.get_valid_moves()
# #             if not valid_moves:
# #                 game.current_player = -game.current_player
# #                 continue
# #             action = agent.choose_action(state, valid_moves)
# #             game.make_move(*action)
# #             reward = 1 if game.is_game_over() and game.get_winner() == game.current_player else 0
# #             next_state = agent.get_state(game.board)
# #             next_valid_moves = game.get_valid_moves()
# #             agent.update_q_table(state, action, reward, next_state, next_valid_moves)
# #             state = next_state
# #         if episode % 100 == 0:
# #             print(f"エピソード {episode}/{episodes} 完了")
# #     return agent
    
# class DQNAgent:
#     def __init__(self):
#         self.model = Sequential([
#             Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE)),
#             Dense(128, activation='relu'),
#             Dense(128, activation='relu'),
#             Dense(BOARD_SIZE * BOARD_SIZE, activation='linear')
#         ])
#         self.model.compile(optimizer='adam', loss='mse')
#         self.target_model = copy.deepcopy(self.model)
#         self.memory = []
#         self.gamma = 0.9
#         self.epsilon = 1.0
#         self.epsilon_min = 0.1
#         self.epsilon_decay = 0.995
#         self.batch_size = 32
    
#     def get_state(self, board):
#         return np.array(board)

#     def act(self, state):
#         if np.random.rand() < self.epsilon:
#             return random.choice(self.get_valid_moves(state))
#         q_values = self.model.predict(state[np.newaxis, :], verbose=0)
#         valid_q_values = [q for i, q in enumerate(q_values[0]) if self.is_legal_move(state, i)]
#         return np.argmax(valid_q_values)

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#         if len(self.memory) > 10000:
#             self.memory.pop(0)

#     def replay(self):
#         print(f"メモリサイズ: {len(self.memory)}")
#         if len(self.memory) < self.batch_size:
#             return
#         minibatch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*minibatch)
#         states = np.array(states)
#         actions = np.array(actions)
#         rewards = np.array(rewards)
#         next_states = np.array(next_states)
#         dones = np.array(dones)
        
#         targets = self.model.predict(states)
#         next_q_values = self.target_model.predict(next_states)# この行のインデントを調整
#         targets[range(self.batch_size), actions] = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)
#         self.model.fit(states, targets, epochs=1, verbose=0)
#         print(f"入力の形状: {states.shape}")
#         print(f"ターゲットの形状: {targets.shape}")
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
    
#     def choose_action(self, state, valid_moves):
#         if np.random.rand() < self.epsilon:  # ε-greedyでランダムな行動を選択
#             return random.choice(valid_moves)
#         q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
#         # 合法手のみを考慮して最大値の行動を選択
#         valid_q_values = [(q_values[move[1] * BOARD_SIZE + move[0]], move) for move in valid_moves]
#         return max(valid_q_values, key=lambda x: x[0])[1]

#     def update_target_model(self):
#         self.target_model.set_weights(self.model.get_weights())

# # 強化学習による学習
# def train_agent(episodes=1):
#     agent = DQNAgent()
#     for episode in range(episodes):
#         print(f"エピソード {episode}/{episodes} 開始")
#         game = Othello()
#         state = agent.get_state(game.board)
#         while not game.is_game_over():
#             print(f"現在のターン: {game.current_player}, ボード:\n{game.board}")
#             valid_moves = game.get_valid_moves()
#             if not valid_moves:
#                 game.current_player = -game.current_player
#                 continue
#             action = agent.choose_action(state, valid_moves)
#             game.make_move(*action)
#             reward = 1 if game.is_game_over() and game.get_winner() == game.current_player else 0
#             next_state = agent.get_state(game.board)
#             done = game.is_game_over()
#             # 経験を記録
#             agent.remember(state, action[1] * BOARD_SIZE + action[0], reward, next_state, done)
#             state = next_state
#         # 経験を再学習
#         agent.replay()
#         # ターゲットモデルを更新（適宜頻度を調整可能）
#         if episode % 10 == 0:
#             agent.update_target_model()
#         if episode % 100 == 0:
#             print(f"エピソード {episode}/{episodes} 完了")
#     return agent

# # 実行
# if __name__ == "__main__":
#     print("学習開始！")
#     trained_agent = train_agent()
#     print("学習完了！")
