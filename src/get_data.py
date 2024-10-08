import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from convert_data import get_filepaths

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

def piece_to_type(piece):
    _abs_piece = abs(piece) - 1
    _list = [0,1,1,2,2,3,3,4,4,5,5,6,6,6,6,6]
    assert _abs_piece < 16
    return _list[_abs_piece]

def piece_to_name(piece):
    _abs_piece = abs(piece) - 1
    _list = ["帅","士","士","象","象","马","马","车","车","炮","炮","兵","兵","兵","兵","兵"]
    assert _abs_piece < 16
    return _list[_abs_piece]
class CustomIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.i = 0
        self.load_file_nums = 5
        self.num_files = len(file_list)
        self.move_id2move_action, self.move_action2move_id = get_all_legal_moves()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def split_file_list(self, num_workers):
        files_per_worker = self.num_files // num_workers
        split_points = [files_per_worker * i for i in range(num_workers)] + [self.num_files]
        return [(split_points[i], split_points[i + 1]) for i in range(num_workers)]

    def read_file(self,file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            data = json.load(f)
            board = data['board']
            action = data['action']
            ply = data['ply']
            legal_move_ids = data['legal_move_ids']
            # print(ply)
            assert ply == 0 or ply == 1
            #
            fx = action['fx']
            fy = action['fy']
            tx = action['tx']
            ty = action['ty']
            input_dat = np.full(fill_value=0,shape=(15, 10, 9), dtype=np.int64)
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
            output_mask = np.full(fill_value=0,shape=(2086),dtype=np.float32)
            for mode_id in legal_move_ids:
                output_mask[mode_id] = 1
            #
            fill_value = 1 if ply == 1 else 0
            input_dat[14] = np.full(fill_value=fill_value, shape=(10, 9))
            #
            assert board[fy][fx] != 0
            #move_p = board[fy][fx] + 16
            #action_idx = move_p * 90 + ty * 10 + tx
            comb_action = str(fy) + str(fx) + str(ty) + str(tx)
            action_idx = self.move_action2move_id[comb_action]
            yield ([input_dat,output_mask], action_idx)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        files_range = self.split_file_list(num_workers)[worker_id]
        tmp_list = self.file_list[files_range[0]: files_range[1]]
        random.shuffle(tmp_list)
        self.file_list[files_range[0]: files_range[1]] = tmp_list
        for i in range(files_range[0], files_range[1], self.load_file_nums):
            filepaths = self.file_list[i: min(i + self.load_file_nums, files_range[1])]
            for file_path in filepaths:
                try:
                    yield from self.read_file(file_path)
                except Exception as e:
                    for arg in e.args:
                        print(arg)
                    continue

# Step 4: Create a DataLoader object
def create_data_loader(file_list, batch_size, num_workers):
    dataset = CustomIterableDataset(file_list)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,prefetch_factor=2)
    # dataPrefetcher = DataPrefetcher(data_loader, device)
    return data_loader

if __name__ == "__main__":
    filepaths = get_filepaths("data\\converted_data","json")
    print(f"data size is {len(filepaths)}")
    t = CustomIterableDataset(file_list=filepaths)
    res = t.read_file(file_path=filepaths[6])
    print(res)


