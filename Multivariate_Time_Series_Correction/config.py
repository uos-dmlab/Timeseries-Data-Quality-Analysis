import easydict
import torch

args = easydict.EasyDict({
    "train_data" : '../data/hai/hai_normal_60.pickle',
    "test_data" : '../data/hai/hai_attack_60.pickle',
    "Mode": 'Train', # Train, Test
    "batch_size": 16, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "input_size": 14, ## 입력 차원 설정
    "latent_size2":100, ## Hidden 차원 설정
    "latent_size": 100, ## Hidden 차원 설정
    "output_size": 14, ## 출력 차원 설정
    "window_size" : 10, ## sequence Lenght
    "num_layers": 2,     ## LSTM layer 갯수 설정
    "learning_rate" : 0.001, ## learning rate 설정
    "max_iter" : 20000, ## 총 반복 횟수 설정
    'early_stop' : True,  ## valid loss가 작아지지 않으면 early stop 조건 설정
    "num" : 0, ##window size에서 몇번째 index의 데이터를 활용할지
    "VAE" : 'models/VAE_model.pt',
    "Discriminator" : 'models/discriminator_model.pt',
    "Generator" : 'models/generator_model.pt',
    "seq_length" : 100,
    "n_steps_out" : 1,
})
