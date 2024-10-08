import copy
import json
import os
import random
import gc
import numpy as np
import multiprocessing as mp

init_game_board = [
    -8, -6, -4, -2, -1, -3, -5, -7, -9,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,-10,  0,  0,  0,  0,  0,-11,  0,
   -12,  0,-13,  0,-14,  0,-15,  0,-16,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
    12,  0, 13,  0, 14,  0, 15,  0, 16,
     0, 10,  0,  0,  0,  0,  0, 11,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     8,  6,  4,  2,  1,  3,  5,  7,  9,
]

[red,black] = [1,0]

def piece_to_name(piece):
    _abs_piece = abs(piece) - 1
    _list = ["帅","士","士","象","象","马","马","车","车","炮","炮","兵","兵","兵","兵","兵"]
    assert _abs_piece < 16
    return _list[_abs_piece]

def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    return _move_id2move_action, _move_action2move_id

move_id2move_action,move_action2move_id = get_all_legal_moves()

def get_filepaths(directory,extension="cbf"):
    filepaths = []
    gc.disable()
    for root,dirs,files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root,_file))
        # if len(filepaths) > 3000:
        #     break
    gc.enable()
    print("size of filepaths = ",len(filepaths))
    return filepaths

#def make_move(board,action)

def convert_data(cbf_root_path,dump_path):
    paths = get_filepaths(cbf_root_path)
    for idx,path in enumerate(paths):
        _dump_path = os.path.join(dump_path, f"folder_{idx}")
        if not os.path.exists(_dump_path):
            os.mkdir(_dump_path)
        with open(path,"r",encoding="utf-8") as f:
            lines = f.readlines()
            game_board = copy.deepcopy(init_game_board)
            action_sequance = []
            for line in lines:
                if "Move value" in line and "00-00" not in line:
                    convert_line = line.strip()
                    convert_line = convert_line.replace("<Move value=\"","")
                    convert_line = convert_line.replace(" ", "")
                    convert_line = convert_line.replace("/>", "")
                    split_list = convert_line.split("-")
                    from_x,from_y = int(split_list[0][0]),int(split_list[0][1])
                    to_x,to_y = int(split_list[1][0]),int(split_list[1][1])
                    action = {
                        "fx" : from_x,
                        "fy" : from_y,
                        "tx" : to_x,
                        "ty" : to_y
                    }
                    action_sequance.append(action)
            #
            ply = red
            for action in action_sequance:
                from_i = action['fy'] * 9 + action['fx']
                to_i = action['ty'] * 9 + action['tx']
                # convert data
                a_np_board = np.reshape(np.array(game_board,dtype=np.int64),newshape=(10,9))
                b_np_board = np.fliplr(a_np_board)
                c_np_board = -np.flipud(b_np_board)
                d_np_board = np.fliplr(c_np_board)
                #
                a_action = copy.deepcopy(action)
                #
                b_action = copy.deepcopy(action)
                b_action['fx'] = 8 - b_action['fx']
                b_action['tx'] = 8 - b_action['tx']
                #
                c_action = copy.deepcopy(b_action)
                c_action['fy'] = 9 - c_action['fy']
                c_action['ty'] = 9 - c_action['ty']
                #
                d_action = copy.deepcopy(c_action)
                d_action['fx'] = 8 - d_action['fx']
                d_action['tx'] = 8 - d_action['tx']
                #
                a_ply = copy.deepcopy(ply)
                b_ply = copy.deepcopy(ply)
                c_ply = 1 - ply
                d_ply = 1 - ply
                assert c_ply == 1 or c_ply == 0
                #
                boards = [a_np_board.tolist(),b_np_board.tolist(),c_np_board.tolist(),d_np_board.tolist()]
                actions = [a_action,b_action,c_action,d_action]
                plys = [a_ply,b_ply,c_ply,d_ply]
                #
                for i in range(4):
                    random_id = random.randint(1, 100000000000000000000000)
                    with open(os.path.join(_dump_path, f"{random_id}.json"), "w+", encoding="utf-8") as f:
                        _dict = {
                            "board" : boards[i],
                            "action" : actions[i],
                            "ply" : plys[i]
                        }
                        json.dump(_dict,f)
                # make move
                game_board[to_i] = game_board[from_i]
                game_board[from_i] = 0
                ply = 1 - ply
                assert ply == 0 or ply == 1

def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and toX in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        return True
    return False


# 不能走到自己的棋子位置
def check_obstruct(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True

def convert_to_state_list(board):
    state_list = np.full(shape=(10,9),fill_value="一一")

    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p:
                k = "红" if p > 0 else "黑"
                t = piece_to_name(p)
                state_list[y][x] = k + t
    #print(state_list)
    return state_list


def get_legal_moves(state_list, current_player_color):
    """
    ====
      将
    车
    ====
    ====
      将
      车
    ====
    ====
    将
      车
    ====
    ====
    将
    车
    ====
    ====
      将
    车
    ====
    这个时候，车就不能再往右走抓帅
    接下来不能走的动作是(1011)，因为将会盘面与state_deque[-4]重复
    """

    moves = []  # 用来存放所有合法的走子方法
    face_to_face = False  # 将军面对面

    # 记录将军的位置信息
    k_x = None
    k_y = None
    K_x = None
    K_y = None

    # state_list是以列表形式表示的, len(state_list) == 10, len(state_list[0]) == 9
    # 遍历移动初始位置
    for y in range(10):
        for x in range(9):
            # 只有是棋子才可以移动
            if state_list[y][x] == '一一':
                pass
            else:
                if state_list[y][x] == '黑车' and current_player_color == '黑':  # 黑车的合法走子
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        # 前面是先前位置，后面是移动后的位置
                        # 这里通过中断for循环实现了车的走子，车不能越过子
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)

                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '红' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)

                elif state_list[y][x] == '红车' and current_player_color == '红':  # 红车的合法走子
                    toY = y
                    for toX in range(x - 1, -1, -1):
                        # 前面是先前位置，后面是移动后的位置
                        # 这里通过中断for循环实现了，车不能越过子
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)

                    toX = x
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if state_list[toY][toX] != '一一':
                            if '黑' in state_list[toY][toX]:
                                moves.append(m)
                            break
                        moves.append(m)

                # 黑马的合理走法
                elif state_list[y][x] == '黑马' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                    and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                    and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                moves.append(m)

                # 红马的合理走法
                elif state_list[y][x] == '红马' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        for j in range(-1, 3, 2):
                            toY = y + 2 * i
                            toX = x + 1 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                    and state_list[toY - i][x] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                moves.append(m)
                            toY = y + 1 * i
                            toX = x + 2 * j
                            if check_bounds(toY, toX) \
                                    and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                    and state_list[y][toX - j] == '一一':
                                m = str(y) + str(x) + str(toY) + str(toX)
                                moves.append(m)

                # 黑象的合理走法
                elif state_list[y][x] == '黑象' and current_player_color == '黑':
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 5 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 5 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

                # 红象的合理走法
                elif state_list[y][x] == '红象' and current_player_color == '红':
                    for i in range(-2, 3, 4):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 4 and state_list[y + i // 2][x + i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) \
                                and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 4 and state_list[y + i // 2][x - i // 2] == '一一':
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

                # 黑士的合理走法
                elif state_list[y][x] == '黑士' and current_player_color == '黑':
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑') \
                                and toY >= 7 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

                # 红士的合理走法
                elif state_list[y][x] == '红士' and current_player_color == '红':
                    for i in range(-1, 3, 2):
                        toY = y + i
                        toX = x + i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toY = y + i
                        toX = x - i
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红') \
                                and toY <= 2 and 3 <= toX <= 5:
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

                # 黑帅的合理走法
                elif state_list[y][x] == '黑帅':
                    k_x = x
                    k_y = y
                    if current_player_color == '黑':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                toY = y + i * sign
                                toX = x + j * sign

                                if check_bounds(toY, toX) and check_obstruct(
                                        state_list[toY][toX], current_player_color='黑') and toY >= 7 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    moves.append(m)

                # 红帅的合理走法
                elif state_list[y][x] == '红帅':
                    K_x = x
                    K_y = y
                    if current_player_color == '红':
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 - i
                                toY = y + i * sign
                                toX = x + j * sign

                                if check_bounds(toY, toX) and check_obstruct(
                                        state_list[toY][toX], current_player_color='红') and toY <= 2 and 3 <= toX <= 5:
                                    m = str(y) + str(x) + str(toY) + str(toX)
                                    moves.append(m)

                # 黑炮的合理走法
                elif state_list[y][x] == '黑炮' and current_player_color == '黑':
                    toY = y
                    hits = False
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    moves.append(m)
                                break
                    hits = False
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    moves.append(m)
                                break

                    toX = x
                    hits = False
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    moves.append(m)
                                break
                    hits = False
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '红' in state_list[toY][toX]:
                                    moves.append(m)
                                break

                # 红炮的合理走法
                elif state_list[y][x] == '红炮' and current_player_color == '红':
                    toY = y
                    hits = False
                    for toX in range(x - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    moves.append(m)
                                break
                    hits = False
                    for toX in range(x + 1, 9):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    moves.append(m)
                                break

                    toX = x
                    hits = False
                    for toY in range(y - 1, -1, -1):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    moves.append(m)
                                break
                    hits = False
                    for toY in range(y + 1, 10):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        if hits is False:
                            if state_list[toY][toX] != '一一':
                                hits = True
                            else:
                                moves.append(m)
                        else:
                            if state_list[toY][toX] != '一一':
                                if '黑' in state_list[toY][toX]:
                                    moves.append(m)
                                break

                # 黑兵的合法走子
                elif state_list[y][x] == '黑兵' and current_player_color == '黑':
                    toY = y - 1
                    toX = x
                    if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        moves.append(m)
                    # 小兵过河
                    if y < 5:
                        toY = y
                        toX = x + 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toX = x - 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='黑'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

                # 红兵的合法走子
                elif state_list[y][x] == '红兵' and current_player_color == '红':
                    toY = y + 1
                    toX = x
                    if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                        m = str(y) + str(x) + str(toY) + str(toX)
                        moves.append(m)
                    # 小兵过河
                    if y > 4:
                        toY = y
                        toX = x + 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)
                        toX = x - 1
                        if check_bounds(toY, toX) and check_obstruct(state_list[toY][toX], current_player_color='红'):
                            m = str(y) + str(x) + str(toY) + str(toX)
                            moves.append(m)

    if K_x is not None and k_x is not None and K_x == k_x:
        face_to_face = True
        for i in range(K_y + 1, k_y, 1):
            if state_list[i][K_x] != '一一':
                face_to_face = False

    if face_to_face is True:
        if current_player_color == '黑':
            m = str(k_y) + str(k_x) + str(K_y) + str(K_x)
            moves.append(m)
        else:
            m = str(K_y) + str(K_x) + str(k_y) + str(k_x)
            moves.append(m)

    moves_id = []
    for move in moves:
        if move in move_action2move_id:
            moves_id.append(move_action2move_id[move])
    return moves_id

def get_moves():
    board = copy.deepcopy(init_game_board)
    current_player_color = "红"
    state_list = convert_to_state_list(board)
    move_ids = get_legal_moves(state_list,current_player_color)
    print(move_ids)
    return


def expand_json_data(dump_path):
    filepaths = get_filepaths(dump_path,extension="json")
    for idx,path in enumerate(filepaths):
        global data
        data = None
        with open(path,"r+",encoding="utf-8") as f:
            try:
                data = json.load(f)
                if 'legal_move_ids' not in data:
                    board = data['board']
                    state_list = convert_to_state_list(board)
                    ply_name = "红" if data['ply'] == 1 else "黑"
                    legal_move_ids = get_legal_moves(state_list, ply_name)
                    data['legal_move_ids'] = legal_move_ids
                else:
                    continue
            except:
                pass
        if data is not None:
            with open(path, "w+", encoding="utf-8") as f:
                json.dump(data,f)
        else:
            os.remove(path)
        print(f"{idx + 1}/{len(filepaths)}")
    return

def single_expand_json_data(path):
    with open(path, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if 'legal_move_ids' not in data:
                board = data['board']
                state_list = convert_to_state_list(board)
                ply_name = "红" if data['ply'] == 1 else "黑"
                legal_move_ids = get_legal_moves(state_list, ply_name)
                data['legal_move_ids'] = legal_move_ids
        except:
            pass
    if data is not None:
        with open(path, "w+", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        os.remove(path)
    return
def parallel_expand_json_data(dump_path):
    pool = mp.Pool(mp.cpu_count())
    pool.map(single_expand_json_data,get_filepaths(dump_path,"json"))

if __name__ == "__main__":
    cbf_root_path = "data\\cbf"
    dump_path = "data\\converted_data"
    #convert_data(cbf_root_path=cbf_root_path,dump_path=dump_path)
    parallel_expand_json_data(dump_path=dump_path)