import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# CSVファイルの読み込み (ヘッダーなし)
csv_data = pd.read_csv("10.csv", header=None)

# 正規表現を使って2文字ずつ切り出す
extract_one_hand = csv_data[0].str.extractall('(..)')


# Indexを再構成して、1行1手の表にする
# 試合の切り替わり判定のためgame_idも残しておく
one_hand_df = extract_one_hand.reset_index().rename(columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"})

# 列の値を数字に変換するdictonaryを作る
def left_build_conv_table():
  left_table = ["a","b","c","d","e","f","g","h"]
  left_conv_table = {}
  n = 1
  
  for t in left_table:
    left_conv_table[t] = n
    n = n + 1

  return left_conv_table

left_conv_table = left_build_conv_table()

# dictionaryを使って列の値を数字に変換する
def left_convert_colmn_str(col_str):
  return left_conv_table[col_str]  

# 1手を数値に変換する
def convert_move(v):
  l = left_convert_colmn_str(v[:1]) # 列の値を変換する
  r = int(v[1:]) # 行の値を変換する
  return np.array([l - 1, r - 1], dtype='int8')

one_hand_df["move"] = one_hand_df.apply(lambda x: convert_move(x["move_str"]), axis=1)

# 盤面の中にあるかどうかを確認する
def is_in_board(cur):
  return cur >= 0 and cur <= 7

# ある方向(direction）に対して石を置き、可能なら敵の石を反転させる
def put_for_one_move(board_a, board_b, move_row, move_col, direction):
  board_a[move_row][move_col] = 1

  tmp_a = board_a.copy()
  tmp_b = board_b.copy()
  cur_row = move_row
  cur_col = move_col

  cur_row += direction[0]
  cur_col += direction[1]
  reverse_cnt = 0
  while is_in_board(cur_row) and is_in_board(cur_col):
    if tmp_b[cur_row][cur_col] == 1: # 反転させる
      tmp_a[cur_row][cur_col] = 1
      tmp_b[cur_row][cur_col] = 0
      cur_row += direction[0]
      cur_col += direction[1]
      reverse_cnt += 1
    elif tmp_a[cur_row][cur_col] == 1:
      return tmp_a, tmp_b, reverse_cnt
    else:
      return board_a, board_b, reverse_cnt
  return board_a, board_b, reverse_cnt

# 方向の定義（配列の要素は←、↖、↑、➚、→、➘、↓、↙に対応している）
directions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]

# ある位置に石を置く。すべての方向に対して可能なら敵の石を反転させる
def put(board_a, board_b ,move_row, move_col):
  tmp_a = board_a.copy()
  tmp_b = board_b.copy()
  global directions
  reverse_cnt_amount = 0
  for d in directions:
    board_a ,board_b, reverse_cnt = put_for_one_move(board_a, board_b, move_row, move_col, d)
    reverse_cnt_amount += reverse_cnt

  return board_a , board_b, reverse_cnt_amount

# 盤面の位置に石がないことを確認する
def is_none_state(board_a, board_b, cur_row, cur_col):
  return board_a[cur_row][cur_col] == 0 and board_b[cur_row][cur_col] == 0

# 盤面に石が置けるかを確認する（ルールでは敵の石を反転できるような位置にしか石を置けない）  
def can_put(board_a, board_b, cur_row, cur_col):
  copy_board_a = board_a.copy()
  copy_board_b = board_b.copy()
  _,  _, reverse_cnt_amount = put(copy_board_a, copy_board_b, cur_row, cur_col)
  return reverse_cnt_amount > 0

# パスする必要のある盤面かを確認する
def is_pass(is_black_turn, board_black, board_white):
  if is_black_turn:
    own = board_black
    opponent = board_white
  else:
    own = board_white
    opponent = board_black
  for cur_row in range(8):
      for cur_col in range(8):
        if is_none_state(own, opponent, cur_row, cur_col) and can_put(own, opponent, cur_row, cur_col):
          return False
  return True

# 変数の初期化
b_tournamentId = -1 # トーナメント番号
board_black = [] # 黒にとっての盤面の状態（１試合保存用）
board_white = [] # 白にとっての盤面の状態（１試合保存用）
boards_black = [] # 黒にとっての盤面の状態（全トーナメント保存用）
boards_white = [] # 白にとっての盤面の状態（全トーナメント保存用）
moves_black = [] # 黒の打ち手（全トーナメント保存用）
moves_white = [] # 白の打ち手（全トーナメント保存用）
is_black_turn = True # True = 黒の番、 False = 白の番
# ターン（黒の番 or 白の番）を切り変える
def switch_turn(is_black_turn):
  return is_black_turn == False # ターンを切り替え

# 棋譜のデータを１つ読み、学習用データを作成する関数
def process_tournament(df):
  global is_black_turn
  global b_tournamentId
  global boards_white
  global boards_black
  global board_white
  global board_black
  global moves_white
  global moves_black
  if df["tournamentId"] != b_tournamentId:
    # トーナメントが切り替わったら盤面を初期状態にする
    b_tournamentId = df["tournamentId"]
    board_black = np.zeros(shape=(8,8), dtype='int8')
    board_black[3][4] = 1
    board_black[4][3] = 1
    board_white = np.zeros(shape=(8,8), dtype='int8')
    board_white[3][3] = 1
    board_white[4][4] = 1
    is_black_turn = True
  else:
    # ターンを切り替える
    is_black_turn = switch_turn(is_black_turn)
    if is_pass(is_black_turn, board_black, board_white): # パスすべき状態か確認する
      is_black_turn = switch_turn(is_black_turn) #パスすべき状態の場合はターンを切り替える

  # 黒の番なら黒の盤面の状態を保存する、白の番なら白の盤面の状態を保存する
  if is_black_turn:
    boards_black.append(np.array([board_black.copy(), board_white.copy()], dtype='int8'))
  else:
    boards_white.append(np.array([board_white.copy(), board_black.copy()], dtype='int8'))
  
  # 打ち手を取得する
  move = df["move"]
  move_one_hot = np.zeros(shape=(8,8), dtype='int8')
  move_one_hot[move[1]][move[0]] = 1

  # 黒の番なら自分→敵の配列の並びをを黒→白にして打ち手をセットする。白の番なら白→黒の順にして打ち手をセットする
  if is_black_turn:
    moves_black.append(move_one_hot)
    board_black, board_white, _ = put(board_black, board_white, move[1], move[0])
  else:
    moves_white.append(move_one_hot)
    board_white, board_black, _ = put(board_white, board_black, move[1], move[0])

# 棋譜データを学習データに展開する
one_hand_df.apply(lambda x: process_tournament(x), axis= 1)

x_train = np.concatenate([boards_black, boards_white])
y_train = np.concatenate([moves_black, moves_white])  
# 教師データは8x8の2次元データになっているので、64要素の1次元データにreshapeする
y_train_reshape = y_train.reshape(-1,64)

class Bias(layers.Layer):
    def __init__(self, input_shape):
        super(Bias, self).__init__()
        self.W = tf.Variable(initial_value=tf.zeros(input_shape[1:]), trainable=True)

    def call(self, inputs):
        return inputs + self.W  

model = keras.Sequential()
model.add(layers.Permute((2, 3, 1), input_shape=(2, 8, 8)))  # 入力データの次元を変換
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層1
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層2
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層3
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層4
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層5
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層6
model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))  # Conv2D層7
model.add(layers.Conv2D(1, kernel_size=1, use_bias=False))  # 出力用のConv2D層
model.add(layers.Flatten())  # 平坦化
model.add(layers.Dropout(0.5))  # ドロップアウト層（50%）
model.add(Bias((1, 64)))  # カスタムバイアス層
model.add(layers.Activation('softmax'))  # 出力層（Softmax）

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # 監視する指標（検証データの損失）
    patience=5,          # 指標が改善しないエポック数
    restore_best_weights=True  # 最良の重みを復元
)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# # Tensor Boardコールバックの設定
# tb_cb = keras.callbacks.TensorBoard(log_dir='model_log/relu_1.100', histogram_freq=1, write_graph=True)
tb_cb = keras.callbacks.TensorBoard(log_dir='model_log/relu_1.200.1', histogram_freq=1, write_graph=True)

start_time = time.time()  # 訓練開始時間を記録


model.fit(x_train, y_train_reshape,epochs=200,batch_size=32,validation_split=0.2,callbacks=[tb_cb, early_stopping])  # EarlyStoppingを追加
model.save('saved_model/my_model5000@200.7kai#3')

end_time = time.time()  # 訓練終了時間を記録
training_time = end_time - start_time  # 訓練にかかった時間を計算
print(f'訓練が完了しました。モデルを保存しました。訓練時間: {training_time:.2f}秒')


# # オセロの初期の盤面データを与える
# board_data = np.array([[
# [[0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,1,0,0,0],
#  [0,0,0,1,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0]],

# [[0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,1,0,0,0,0],
#  [0,0,0,0,1,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0]]]],dtype=np.int8)

# # 打ち手を予想する
# model.predict(board_data)  
# # 出力結果
# # array([[2.93197723e-11, 1.42428486e-10, 7.34781472e-11, 2.39318716e-08,
# #         1.31301447e-09, 1.50736756e-08, 9.80145964e-10, 2.52176102e-09,
# #         3.33402395e-09, 2.05685264e-08, 2.49693510e-09, 3.53782520e-12,
# #         8.09815548e-10, 6.63711930e-08, 2.62752469e-08, 5.35828759e-09,
# #         4.46924164e-10, 7.42555386e-08, 2.38477658e-11, 3.76452749e-06,
# #         6.29236463e-12, 4.04659602e-07, 2.37438894e-06, 1.51068477e-10,
# #         1.81150719e-11, 4.47054616e-10, 3.75479488e-07, 2.84151619e-14,
# #         3.70454689e-09, 1.66316525e-07, 1.27947108e-09, 3.30583454e-08,
# #         5.33877942e-10, 5.14411222e-11, 8.31681668e-11, 6.85821679e-13,
# #         1.05046523e-08, 9.99991417e-01, 3.23126500e-07, 1.72151644e-07,
# #         1.01420455e-10, 3.35642431e-10, 2.22305030e-12, 5.21605148e-10,
# #         5.75579229e-08, 9.84997257e-08, 3.62535673e-07, 4.41284129e-08,
# #         2.43385506e-10, 1.96498547e-11, 1.13820758e-11, 3.01558468e-14,
# #         3.58017758e-08, 8.61415117e-09, 1.17988044e-07, 1.36784823e-08,
# #         1.19627297e-09, 2.55619081e-10, 9.82019244e-10, 2.45560993e-12,
# #         2.43100295e-09, 8.31343083e-09, 4.34338648e-10, 2.09913722e-08]],
# #       dtype=float32)
# np.argmax(model.predict(board_data))

# 出力結果
# 37
