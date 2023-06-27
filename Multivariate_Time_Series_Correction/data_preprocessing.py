import torch
import numpy as np
import pickle
import gzip
from sklearn.model_selection import train_test_split

def minmax_scaler(data):
    numerator=data-np.min(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    return numerator/(denominator+1e-7)

def data_preprocessing(train_input_file, test_input_file):
  #데이터 파일을 불러옴
  
  with gzip.open(train_input_file) as FI:
        train = pickle.load(FI)
  
  train = train[240:, :] ## 초반 4시간 제거(논문 참고)

  with gzip.open(test_input_file) as FI:
        test = pickle.load(FI)

  y = np.where(test[:,-1]==1)

  test = test[:,:-1]

  total = np.concatenate([train, test])
  total=minmax_scaler(total)

  train_final = total[:len(train),:]
  test_final = total[len(train):,:]
  # train_final = minmax_scaler(train_final)
  # test_final = minmax_scaler(test_final)
  anomal_idx = y[0]
    
  return train_final, test_final, anomal_idx

def Split_data(data):
  interval_n = int(len(data)/10)
  train_data = data[0:interval_n*7] # 학습 데이터
  validate_data = data[interval_n*7:] # 검증 데이터

  return train_data, validate_data

def make_data_idx(data, window_size):
  input_idx = []
  for idx in range(window_size-1, len(data)):
    input_idx.append(list(range(idx - window_size+1, idx+1)))

  return input_idx

class Get_Dataset:

    def __init__(self, data, Windowsize):
      
      self.input_ids = make_data_idx(data, Windowsize)

      self.var_data = np.array(data)
      self.var_data = torch.FloatTensor(self.var_data)

    def __len__(self):

        # write your codes here
        return len(self.input_ids)

    def __getitem__(self, idx):
      temp_input_ids = self.input_ids[idx]
      input_values = self.var_data[temp_input_ids]

      return input_values

def build_dataset(sequences, n_steps_in, n_steps_out):
    dataX = []
    dataY = []
    for i in range(len(sequences)):
        # if i %  == 0:
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, :]
        dataX.append(seq_x)
        dataY.append(seq_y)
    
    x_train, x_val, y_train, y_val = train_test_split(np.array(dataX), np.array(dataY), test_size=0.3, shuffle=True, random_state=42)

    return x_train, x_val, y_train, y_val

def data_preprocessing_test(train_input_file, test_input_file, start_idx, seq_length):
  #데이터 파일을 불러옴
  
  with gzip.open(train_input_file) as FI:
        train = pickle.load(FI)

  train = train[240:, :] ## 초반 4시간 제거(논문 참고)

  with gzip.open(test_input_file) as FI:
        test = pickle.load(FI)

  y = np.where(test[:,-1]==1)

  test = test[:,:-1]

  total = np.concatenate([train, test])
  total=minmax_scaler(total)

  train_final = total[:len(train),:]
  test_final = total[len(train):,:]

  test_data = test_final[start_idx-seq_length:, :]
  anomal_idx = y[0]
    
  return train_final, test_data, anomal_idx

def build_dataset_multi(sequences, n_steps_in, n_steps_out):
    dataX = []
    dataY = []
    for i in range(len(sequences)):
        if i % n_steps_in ==0:
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out-1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, :]
            dataX.append(seq_x)
            dataY.append(seq_y)

    return np.array(dataX), np.array(dataY)

def build_dataset_uni(sequences, n_steps_in):
    dataX = []
    dataY = []
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        dataX.append(seq_x)
        dataY.append(seq_y)
    
    return np.array(dataX), np.array(dataY)