import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# モデルのロード
model = keras.models.load_model('saved_model/my_model5000@200kai#3')

# 盤面の状態（0: 空き, 1: 黒, -1: 白）
def init_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] = 1   # 黒
    return board

# 石を置けるか確認する関数
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False

    current_player = 1 if is_black_turn else -1
    opponent_player = -1 if is_black_turn else 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for direction in directions:
        x, y = row + direction[0], col + direction[1]
        has_opponent_between = False

        while 0 <= x < 8 and 0 <= y < 8:
            if board[x, y] == opponent_player:
                has_opponent_between = True
            elif board[x, y] == current_player:
                if has_opponent_between:
                    return True
                else:
                    break
            else:
                break
            x += direction[0]
            y += direction[1]

    return False

# モデルAIによる手を選ぶ関数
def model_move(board, model, is_black_turn):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    if valid_moves:
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        board_input[0, 0] = (board == 1).astype('int8')  # 黒の盤面
        board_input[0, 1] = (board == -1).astype('int8')  # 白の盤面
        predictions = model.predict(board_input)[0]
        
        best_move = max(valid_moves, key=lambda pos: predictions[pos[0] * 8 + pos[1]])
        return best_move
    return None

# ランダムAIによる手を選ぶ関数
def random_move(board, is_black_turn):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    return random.choice(valid_moves) if valid_moves else None

# 盤面に石を置き、石を反転する関数
def put(board, row, col, is_black_turn):
    player = 1 if is_black_turn else -1
    board[row, col] = player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for dx, dy in directions:
        x, y = row + dx, col + dy
        stones_to_flip = []
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == -player:
            stones_to_flip.append((x, y))
            x += dx
            y += dy
        if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            for fx, fy in stones_to_flip:
                board[fx, fy] = player

                
# ゲーム(モデル黒の先攻)の進行を管理する関数
def play_game(model):
    board = init_board()
    is_black_turn = True

    while True:
        #print_board(board)

        # 現在のプレイヤーの合法手をチェック
        if is_black_turn:
            move = model_move(board, is_black_turn)
        else:
            move = random_move(board, model, is_black_turn)

        if move:
            put(board, move[0], move[1], is_black_turn)
        else:
            if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                break  # 両者ともに合法手がない場合は終了
            print("パス")
        
        is_black_turn = not is_black_turn  # ターンを切り替え

    return board

# 盤面を表示する関数
def print_board(board):
    for row in board:
        print(" ".join(["●" if x == 1 else "○" if x == -1 else "." for x in row]))
    print()

# 勝者判定のために石の数をカウントする関数
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

# モデルAI（黒）と人間プレイヤー（白）の対戦

def print_valid_moves(board, is_black_turn):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    print("現在、置ける位置は以下です:")
    for move in valid_moves:
        print(f"({move[0]},{move[1]})")

def human_move(board):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, False)]
    if not valid_moves:
        print("置ける場所がありません。あなたはパスします。")
        return None
    while True:
        print_valid_moves(board, False)  # 白の番で置ける位置を表示
        try:
            move = input("あなたの番です。石を置きたい位置を入力してください(例:3,4): ")
            row, col = map(int, move.split(','))
            if can_put(board, row, col, False):  # 白の番
                return row, col
            else:
                print("その位置には石を置けません。再度入力してください。")
        except (ValueError, IndexError):
            print("無効な入力です。再度入力してください(例:3,4)")


# ゲーム進行を管理する関数
def play_game(model):
    board = init_board()
    is_black_turn = True  # モデルAIが黒（先攻）

    while True:
        print_board(board)

        # 現在のプレイヤーの合法手をチェック
        if is_black_turn:
            print("AI（黒）のターンです...")
            move = model_move(board, model, is_black_turn)
        else :
            print("あなた（白）のターンです...")
            move = human_move(board)

        if move:
            put(board, move[0], move[1], is_black_turn)
        else:
            if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                print("両者ともに合法手がないため、ゲーム終了")
                break  # 両者ともに合法手がない場合は終了
            print("パス")

        is_black_turn = not is_black_turn  # ターンを切り替え

    return board


# 勝者判定のために石の数をカウントする関数
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

# ゲーム開始と結果表示
final_board = play_game(model)
black, white = count_stones(final_board)

# 結果表示
print(f"最終スコア: 黒 = {black}, 白 = {white}")
if black > white:
    print("AI（黒）の勝利!")
elif white > black:
    print("あなた（白）の勝利!")
else:
    print("引き分け!")

