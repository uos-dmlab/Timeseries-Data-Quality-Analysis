from config import *
from tqdm.notebook import trange, tqdm
import torch

def Data_Correction(Score, predict_anomaly, THRESHOLD, args, model_VAE, model_generator, predicted_data, real_data, predict_value, test_loader, num):
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="correcting")
    correct_anomaly = []
    predictions_v1 = []
    true_data_v1 = []
    predict_anomaly.sort()
    
    q = torch.tensor([0.1, 0.9]).to(args.device)
    lower = []
    upper = []
    for i in range(len(real_data[0])):
        lower.append(torch.quantile(torch.stack(real_data)[:,i], q)[0])
        upper.append(torch.quantile(torch.stack(real_data)[:,i], q)[1])

    with torch.no_grad():
        for i, batch_data in test_iterator:
            if i in predict_anomaly: #predict_anomaly[300:4130]
                batch_data = batch_data.to(args.device)
                raw_data = torch.Tensor.clone(batch_data)
                batch_data[:,:,:] = predict_value[i] # 해당 변수 predict_value[i]
                
                batch_size, sequence_length, var_length = batch_data.size()
                
                mu, log_var, encoder_latent = model_VAE(batch_data)
                inv_idx = torch.arange(sequence_length - 1, -1, -1).long()

                temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch_data.device)
                hidden = encoder_latent
                
                reconstruct_output = []
                for t in range(sequence_length):
                    temp_input = model_generator(temp_input, hidden)
                    reconstruct_output.append(temp_input)

                reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]
                
                batch_data = batch_data.to(args.device)
                
                for k in range(len(reconstruct_output)):
                    re_dt = reconstruct_output[k][num]
                    for j in range(len(predictions_v1[0])):
                        re_dt[j][re_dt[j] < lower[j]] = lower[j]
                        re_dt[j][re_dt[j] > upper[j]] = upper[j]
                    predictions_v1.append(re_dt)
                    true_data_v1.append(raw_data[k][num])

            else:
                predictions_v1.append(real_data[i])
                true_data_v1.append(real_data[i])
        
        predictions_v1 = torch.stack(predictions_v1)
        true_data_v1 = torch.stack(true_data_v1)

    return predictions_v1, true_data_v1