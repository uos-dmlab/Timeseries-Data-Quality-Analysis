import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import tensorflow as tf
from tensorflow.keras import backend as K
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Layer, Permute, multiply, Dropout, Reshape, Bidirectional, ReLU, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from tqdm.notebook import trange, tqdm
import seaborn as sns

import pickle
import gzip

## 인코더
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0, bidirectional=False)
        self.hidden2mean = nn.Linear(self.hidden_size*self.num_layers, self.hidden_size)
        self.hidden2logv = nn.Linear(self.hidden_size*self.num_layers, self.hidden_size)

    def encode(self,x):
        mu = self.hidden2mean(x)
        log_var = self.hidden2logv(x)
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        z = torch.randn([self.num_layers, self.batch_size, self.hidden_size]).to(args.device)
        z = z * std + mu
        return z

    def forward(self, x):
        self.batch_size, _, _ = x.size()
        _, (hidden,cell) = self.lstm(x)  
        
        if self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(self.batch_size, self.hidden_size*self.num_layers)
        else:
            hidden = hidden.squeeze()

        mu, logvar = self.encode(hidden)
        z = self.reparametrize(mu,logvar)
        
        return mu, logvar, (z, cell)

## Discriminator
class Discriminator(nn.Module):

    def __init__(self, batch_size,input_size, hidden_size, num_layers):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.0, bidirectional=False)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        output = self.fc(outputs)
        num_dims = len(output.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        output = torch.mean(output, dim=reduction_dims)
        return output

## Generator
class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        # self.fc = nn.Sequential(nn.Linear(hidden_size, output_size)) ### nn.ReLU

    def forward(self, x, latent):
        output, (_,_) = self.lstm(x, latent)  
        prediction = self.fc(output)
        return prediction

def loss_function_VAE(reconstruct_output, batch_data, mu, log_var):
    recon_x = reconstruct_output
    x = batch_data
    mu = mu
    logvar = log_var

    Le = F.mse_loss(recon_x, x)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return Le + KLD

def loss_function_discriminator(window_size, output_discriminator_real, output_discriminator_generated, output_discriminator_noise, real_samples_labels, generated_samples_labels):
    loss_discriminator_real = 0
    loss_discriminator_generated = 0

    output_discriminator_generated = torch.stack(output_discriminator_generated, dim=1)
    output_discriminator_real = torch.stack(output_discriminator_real, dim=1)
    output_discriminator_noise = torch.stack(output_discriminator_noise, dim=1)

    output_discriminator_real = torch.clamp(output_discriminator_real, 1e-40, 1.0)
    loss_discriminator_real = -torch.log(output_discriminator_real)

    output_discriminator_generated = torch.clamp(output_discriminator_generated, 1e-40, 1.0)  
    loss_discriminator_generated = -torch.log(1-output_discriminator_generated)
    
    output_discriminator_noise = torch.clamp(output_discriminator_noise, 1e-40, 1.0)  
    loss_discriminator_noise = -torch.log(1-output_discriminator_noise)

    batch_loss = loss_discriminator_real + loss_discriminator_generated + 0.1*loss_discriminator_noise

    return torch.mean(batch_loss)

def loss_function_Generator(reconstruct_output, batch_data, window_size, output_discriminator_generated, output_discriminator_noise, real_samples_labels):
    recon_x = reconstruct_output
    x = batch_data
    
    Le = F.mse_loss(recon_x, x)

    loss_Generator = 0

    output_discriminator_generated = torch.stack(output_discriminator_generated, dim=1)
    output_discriminator_noise = torch.stack(output_discriminator_noise, dim=1)

    output_discriminator_generated = torch.clamp(output_discriminator_generated, 1e-40, 1.0)
    output_discriminator_noise = torch.clamp(output_discriminator_noise, 1e-40, 1.0)
    
    loss_generated = -torch.log(output_discriminator_generated)
    loss_noise = -torch.log(output_discriminator_noise)
    
    loss_Generator = loss_generated + 0.1*loss_noise + Le

    return torch.mean(loss_Generator)

def Model_initialize(input_dim, latent_dim2 ,latent_dim, window_size, num_layers, batch_size):

    discriminator = Discriminator(batch_size=args.batch_size,
                                  input_size=input_dim,
                                  hidden_size=latent_dim,
                                  num_layers=num_layers,)

    generator = Generator(input_size=input_dim,
                          output_size=input_dim,
                          hidden_size=latent_dim2,
                          num_layers=num_layers,
                         )
    encoder = Encoder(input_size=input_dim,
                      hidden_size=latent_dim2,
                      num_layers=num_layers,
                     )
    
    return encoder, discriminator, generator

def run(args, model_VAE, model_discriminator, model_generator, train_loader, test_loader):

    # optimizer 설정
    optimizer_VAE = torch.optim.Adam(model_VAE.parameters(), lr=args.learning_rate)
    optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr=args.learning_rate)
    optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=args.learning_rate)

    ## 반복 횟수 Setting
    epochs = tqdm(range(args.max_iter//len(train_loader)+1))

    ## 학습하기
    count = 0
    best_loss_VAE = 100000000
    best_loss_GAN = 100000000

    for epoch in epochs:

        model_VAE.train()
        model_discriminator.train()
        model_generator.train()

        optimizer_VAE.zero_grad()
        optimizer_discriminator.zero_grad()
        optimizer_generator.zero_grad()
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")

        for i, batch_data in train_iterator:
            batch_data = batch_data.to(args.device)
            batch_size, sequence_length, var_length = batch_data.size()

            real_samples_labels = torch.ones((batch_size, 1))
            generated_samples_labels = torch.zeros((batch_size, 1))

            mu, log_var, encoder_latent = model_VAE(batch_data)

            inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
            output_discriminator_real = []
            output_discriminator_generated = []
            output_discriminator_noise = []
            reconstruct_output = []

            temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch_data.device)
            hidden = encoder_latent
            
            ########################################################################################################
            z_noise = Variable(torch.zeros((hidden[1].size()[0], hidden[1].size()[1], hidden[1].size()[2]))).cuda() # feature_sizehi
            hidden_noise = Variable(torch.randn((hidden[1].size()[0], hidden[1].size()[1], hidden[1].size()[2]))).cuda() # feature_sizehi

            for t in range(sequence_length):
                temp_input = model_generator(temp_input, hidden)
                temp_output_discriminator_generated = model_discriminator(temp_input)
                temp_output_discriminator_real = model_discriminator(batch_data[:,t,:].unsqueeze(1))
                
                temp_noise = model_generator(temp_input, [z_noise, hidden_noise])
                temp_output_discriminator_noise = model_discriminator(temp_input)

                output_discriminator_real.append(temp_output_discriminator_real)
                output_discriminator_generated.append(temp_output_discriminator_generated)
                output_discriminator_noise.append(temp_output_discriminator_noise)
                reconstruct_output.append(temp_input)

            reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]

            if count > args.max_iter:
                return model_VAE, model_discriminator, model_generator
            count += 1

            batch_data = batch_data.to(args.device)

            loss_VAE = loss_function_VAE(reconstruct_output, batch_data, mu, log_var)
            loss_discriminator = loss_function_discriminator(args.window_size, output_discriminator_real, output_discriminator_generated, output_discriminator_noise, real_samples_labels, generated_samples_labels)
            loss_Generator = loss_function_Generator(reconstruct_output, batch_data, args.window_size, output_discriminator_generated, output_discriminator_noise, real_samples_labels)

            # Backward and optimize
            loss_VAE.backward(retain_graph=True)
            loss_discriminator.backward(retain_graph=True)
            loss_Generator.backward(retain_graph=True)

            optimizer_VAE.step()
            optimizer_discriminator.step()
            optimizer_generator.step()

            optimizer_VAE.zero_grad()
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()

            train_iterator.set_postfix({
            "loss_VAE": float(loss_VAE),"loss_discriminator": float(loss_discriminator),"loss_Generator": float(loss_Generator)
            })

        model_VAE.eval()
        model_discriminator.eval()
        model_generator.eval()

        eval_loss_VAE = 0
        eval_loss_discriminator = 0
        eval_loss_Generator = 0

        eval_loss = 0

        test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
        with torch.no_grad():
            for i, batch_data in test_iterator:
                batch_data = batch_data.to(args.device)
                batch_size, sequence_length, var_length = batch_data.size()

                real_samples_labels = torch.ones((batch_size, 1))
                generated_samples_labels = torch.zeros((batch_size, 1))

                mu, log_var, encoder_latent = model_VAE(batch_data)

                inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
                output_discriminator_real = []
                output_discriminator_generated = []
                output_discriminator_noise = []
                reconstruct_output = []

                temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch_data.device)
                hidden = encoder_latent
                
                ########################################################################################################
                z_noise = Variable(torch.zeros((hidden[1].size()[0], hidden[1].size()[1], hidden[1].size()[2]))).cuda()
                hidden_noise = Variable(torch.randn((hidden[1].size()[0], hidden[1].size()[1], hidden[1].size()[2]))).cuda()

                for t in range(sequence_length):
                    temp_input = model_generator(temp_input, hidden)
                    temp_output_discriminator_generated = model_discriminator(temp_input)
                    temp_output_discriminator_real = model_discriminator(batch_data[:,t,:].unsqueeze(1))
                    
                    temp_noise = model_generator(temp_input, [z_noise, hidden_noise])
                    temp_output_discriminator_noise = model_discriminator(temp_input)

                    output_discriminator_real.append(temp_output_discriminator_real)
                    output_discriminator_generated.append(temp_output_discriminator_generated)
                    output_discriminator_noise.append(temp_output_discriminator_noise)

                    reconstruct_output.append(temp_input)

                reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]

                batch_data = batch_data.to(args.device)

                loss_VAE = loss_function_VAE(reconstruct_output, batch_data, mu, log_var)
                loss_discriminator = loss_function_discriminator(args.window_size, output_discriminator_real, output_discriminator_generated, output_discriminator_noise, real_samples_labels, generated_samples_labels)
                
                loss_Generator = loss_function_Generator(reconstruct_output, batch_data, args.window_size, output_discriminator_generated, output_discriminator_noise, real_samples_labels)
                
                eval_loss_VAE += loss_VAE.mean().item()
                eval_loss_discriminator += loss_discriminator.mean().item()
                eval_loss_Generator += loss_Generator.mean().item()

                test_iterator.set_postfix({
                "eval_loss_VAE": float(loss_VAE),"eval_loss_discriminator": float(loss_discriminator),"eval_loss_Generator": float(loss_Generator),
                })
        eval_loss_VAE = eval_loss_VAE / len(test_loader)
        eval_loss_discriminator = eval_loss_discriminator / len(test_loader)
        eval_loss_Generator = eval_loss_Generator / len(test_loader)
        epochs.set_postfix({
        "Evaluation Score_VAE": float(loss_VAE), "Evaluation Score_discriminator": float(loss_discriminator), "Evaluation Score_Generator": float(loss_Generator),
        })

        if (eval_loss_VAE < best_loss_VAE and eval_loss_discriminator+eval_loss_Generator < best_loss_GAN):

            best_loss_VAE = eval_loss_VAE
            best_loss_GAN = eval_loss_discriminator + eval_loss_Generator

        else:
            if args.early_stop:
                print('early stop condition   best_loss_VAE[{}]  best_loss_GAN[{}]  eval_loss_VAE[{}]  eval_loss_discriminator[{}]  eval_loss_Generator[{}]'
                      .format(best_loss_VAE, best_loss_GAN, eval_loss_VAE,eval_loss_discriminator,eval_loss_Generator))
                return model_VAE, model_discriminator, model_generator

        torch.save(model_VAE, args.VAE)
        torch.save(model_discriminator, args.Discriminator)
        torch.save(model_generator, args.Generator)
        
    return model_VAE, model_discriminator, model_generator

def discriminate(args, model_discriminator, test_loader):
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
    output_discriminate = []
    with torch.no_grad():
        for i, batch_data in test_iterator:
            batch_data = batch_data.to(args.device)
            batch_size, sequence_length, var_length = batch_data.size()
            temp_output_discriminate = model_discriminator(batch_data)
            # temp_output_discriminate = temp_output_discriminate >= torch.FloatTensor([0.5]).to(args.device)
            for i in range(len(temp_output_discriminate)):
                output_discriminate.append(temp_output_discriminate[i])

    return output_discriminate

def reconstruction(args, model_VAE, model_generator, test_loader, num):
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
    loss_list = []
    predictions = []
    true_data = []
    with torch.no_grad():
        for i, batch_data in test_iterator:
            batch_data = batch_data.to(args.device)    
            batch_size, sequence_length, var_length = batch_data.size()
            mu, log_var, encoder_latent = model_VAE(batch_data)
            inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
            reconstruct_output = []

            temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch_data.device)
            hidden = encoder_latent
            for t in range(sequence_length):
                temp_input = model_generator(temp_input, hidden)
                reconstruct_output.append(temp_input)

            reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]
            batch_data = batch_data.to(args.device)

            for k in range(len(reconstruct_output)):
                predictions.append(reconstruct_output[k][num])
                true_data.append(batch_data[k][num])


    for i in range(len(predictions)):
        Temp_loss_list=abs(predictions[i]-true_data[i]).mean() #50
        loss_list.append(Temp_loss_list)

    return loss_list,predictions,true_data

def Get_Threshold(args, model_VAE, model_generator, model_discriminator, test_loader, num, gamma):
    output_discriminate = discriminate(args, model_discriminator, test_loader)
    loss_list,predictions,true_data = reconstruction(args, model_VAE, model_generator, test_loader, num)
    Res= []
    for i in range(len(loss_list)):
        Res.append(loss_list[i].mean())

    Score = []
    for i in range(len(output_discriminate)):
        Score.append(gamma*(Res[i])+(1-gamma)*output_discriminate[i])
        
    return Score,predictions,true_data

def get_Accuracy(Score, anomal_idx, THRESHOLD, num):
    anomaly_index_true = anomal_idx
    anomaly_index_predict = []
    correct_anomaly = []
    for i in range(0,len(Score)):
        if Score[i] >= THRESHOLD:
            anomaly_index_predict.append(i)

    for i in range(0,len(anomaly_index_predict)):
        k = anomaly_index_predict[i]
        if k in anomaly_index_true:
            correct_anomaly.append(k)

    print(f'Correct anomaly predictions: {len(correct_anomaly)}/{len(anomaly_index_true)}')
    print("Accuracy: {:.2f}%".format(len(correct_anomaly)/len(anomaly_index_true)*100))
    
    return np.array(correct_anomaly) # 

def compare_plot(predictions, true_data, correct_anomaly, num):
    predict=[]
    true=[]
    correct_anomaly_predict=[]
    
    for i in range(len(predictions)): 
        predict.append(predictions[i][num])
        true.append(true_data[i][num])
  
    for i in correct_anomaly:
        correct_anomaly_predict.append(predict[i])
    
    # predict = np.array(predict)
    predict = torch.stack(predict)
    # predict = predict.cpu().numpy()
    # true = np.array(true)
    true = torch.stack(true)
    
    correct_anomaly_predict = torch.stack(correct_anomaly_predict)
    
    Detect_anomaly_predict = correct_anomaly_predict
    predict_ano = correct_anomaly
    
    total_x = len(predict)
    x = np.arange(0,total_x,1)
    plt.figure(figsize=(16, 6))
    plt.plot(x,predict.cpu(),'b',label='Generated value')
    plt.plot(x,true.cpu(),'r',label='True value')
    plt.plot(predict_ano, Detect_anomaly_predict.cpu(),'*g',label='Detected anomaly')
    plt.legend(loc='upper right', fontsize="15")
    plt.ylim([-0.2, 1.5])
    plt.show()

# prediction model
def attention_3d_block(inputs, seq_length):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs) 
    a = Dense(seq_length, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul  = multiply([inputs, a_probs])
    
    return output_attention_mul

def proposed_prediction(x_train, seq_length, n_steps_out):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))

    # Correction of variables
    info = K.mean(tfp.stats.correlation(inputs, sample_axis=1, keepdims=True), axis=1)
    info_2 = tf.where(tf.math.is_nan(info), tf.zeros_like(info), info)
    info_4 = Flatten()(info_2)
    info_5 = Dense(x_train.shape[2]*x_train.shape[2])(info_4)

    # Prediction
    hidden_layer1 = Bidirectional(LSTM(16, return_sequences=True))(inputs)
    hidden_layer2 = attention_3d_block(hidden_layer1, seq_length)
    hidden_layer3 = Flatten()(hidden_layer2)

    Concat_Layer = tf.concat([hidden_layer3, info_5], axis=-1)
    hidden_layer4 = Dense(128)(Concat_Layer)

    hidden_layer5 = Dense(n_steps_out*x_train.shape[2], activation="sigmoid")(hidden_layer4) 
    outcome = Reshape((n_steps_out, x_train.shape[2]))(hidden_layer5)

    proposed_model = Model(inputs=inputs, outputs=outcome)
    
    return proposed_model