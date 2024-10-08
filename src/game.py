import copy

import torch
import numpy as np
from model import block,net
from get_data import get_all_legal_moves,piece_to_type,piece_to_name
from convert_data import convert_to_state_list
from convert_data import get_legal_moves
from torchsummary import summary

torch.backends.cudnn.benchmark = True

init_game_board = [
     0,  0, -4, -2, -1, -3, -5,  0,  0,
     0,  0,  0,  0, -7, -8,  0,  0,  0,
     0,  9, -6,  0,  0,  0,  0,  0,  0,
   -12,  0,-13,  0,-14,  0,  0,  0,-16,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  8,  0,  0,  0,  0,  0,
    12,  0, 13,  0, 14,  0, 15,  0, 16,
     0,  0,  6,  0,  0,  0,-11,  0,  0,
     0, 11,  0,  0,  0,  0,  0, -9,  0,
     0,  0,  4,  2,  1,  3,  5,  0,  0,
]

init_game_board = np.reshape(np.array(init_game_board,dtype=np.int64),newshape=(10,9))

[red,black] = [1,0]

move_id2move_action, move_action2move_id = get_all_legal_moves()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_input(board,ply):
    legal_move_ids = get_legal_moves(convert_to_state_list(board),"红" if ply == 1 else "黑")
    input_dat = np.full(fill_value=0, shape=(15, 10, 9), dtype=np.int64)
    #
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p:
                now_t = piece_to_type(p)
                if p < 0:
                    now_t += 7
                input_dat[now_t][y][x] = 1
    #
    output_mask = np.full(fill_value=0, shape=(2086), dtype=np.float32)
    for mode_id in legal_move_ids:
        output_mask[mode_id] = 1
    #
    fill_value = 1 if ply == 1 else 0
    input_dat[14] = np.full(fill_value=fill_value, shape=(10, 9))
    #
    input_dat = torch.as_tensor(input_dat, dtype=torch.float32).to(device)
    output_mask = torch.as_tensor(output_mask, dtype=torch.float32).to(device)
    return [input_dat,output_mask]

def self_game():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "models\\epoch_1_tcc_130.pth"
    model_params = torch.load(model_path)
    converted_model_params = {}

    for layer_name,weights in model_params.items():
        new_layer_name = layer_name.replace("module.","")
        converted_model_params[new_layer_name] = weights
    model = net(in_channels=15,out_channels=128, num_classes=2086).to(device)
    model.load_state_dict(converted_model_params)
    model = model.eval()
    summary(model,input_size=(15,10,9))
    #
    #move_id2move_action,move_action2move_id = get_all_legal_moves()
    #
    board = copy.deepcopy(init_game_board)
    ply = red
    #
    x,mask = get_input(board,ply)
    #
    x = torch.as_tensor(x,dtype=torch.float32).to(device)
    x = torch.unsqueeze(x,dim=0)
    #
    mask = torch.as_tensor(mask, dtype=torch.float32).to(device)
    mask = torch.unsqueeze(mask, dim=0)
    #
    y = model(x)
    #
    y = torch.softmax(y,dim=1)
    y = torch.mul(y, mask)
    #
    arg_pool = torch.topk(y,3)
    arg_vls,arg_ids = arg_pool
    print(arg_vls,arg_ids)
    arg_idx = int(arg_ids[0][0])
    #
    arg_move = move_id2move_action[arg_idx]
    print(arg_move)
    #
    print(init_game_board)
    #
    fy,fx,ty,tx = int(arg_move[0]),int(arg_move[1]),int(arg_move[2]),int(arg_move[3])
    assert init_game_board[fy][fx]
    init_game_board[ty][tx] = init_game_board[fy][fx]
    init_game_board[fy][fx] = 0
    #
    print()
    print(init_game_board)



if __name__ == "__main__":
    self_game()