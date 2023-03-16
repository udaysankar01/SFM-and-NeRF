import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Network import NeRF
import json
import os
from skimage import io
import math
from PIL import Image
from torchvision.transforms.functional import to_tensor
import cv2
import torchvision.utils as vutils
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def get_focal_length(camera_angle_x, image_width=100):
    return 0.5 * image_width / math.tan(camera_angle_x / 2)



def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = torch.broadcast_to(torch.from_numpy(c2w[:3,-1]), rays_d.shape)
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    
    return rays_o, rays_d

def get_rays_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
        
    camera_angle_x = data['camera_angle_x']
    focal_length = get_focal_length(camera_angle_x)
    
    rays_o_list = []
    rays_d_list = []
    
    for frame in data['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        c2w = transform_matrix[:3, :4]
        rays_o, rays_d = get_rays(100, 100, focal_length, c2w)
    
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
        
    rays_o = torch.stack(rays_o_list, dim=0)
    rays_d = torch.stack(rays_d_list, dim=0)
    
    return rays_o, rays_d



def get_all_data(json_file):

    with open(json_file, 'r') as f:
        json_obj = json.load(f)

    # extract the file paths from the JSON object
    file_paths = [frame['file_path'] for frame in json_obj['frames']]

    # read the images into a list
    images = []
    

    #print(images_tensor.shape)
    concatenated_data = None
    for file_path in file_paths:
        image = Image.open(file_path + '.png').convert('RGB')


        tensor = to_tensor(image)
        tensor = tensor.permute(1, 2, 0)

        tensor=tensor.view(-1,3)
        if concatenated_data is None:
            concatenated_data = tensor
        else:
            concatenated_data = torch.cat((concatenated_data, tensor), dim=0)



    ray_o,ray_d=get_rays_from_json(json_file)


    ray_o = ray_o.reshape(-1, 3)
    ray_d = ray_d.reshape(-1, 3)

    return concatenated_data, ray_o, ray_d







def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=1, nb_bins=192):

        #evenly distributed points on the ray, with number of points nb_bins, the upper 'hf' and lower 'hn' bound for each ray
        t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)


         # Perturb sampling along each ray, splitting of t into lower and upper has been done to have same perturbation for each point.
        mid = (t[:, :-1] + t[:, 1:]) / 2
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        t = lower + (upper - lower) * torch.rand(t.shape, device=device)  # [batch_size, nb_bins]
        delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1) #differences between points, and one large number that is insignificant



 	#Compute colors, density at each point
        x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
        ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
        x=x.float()
        ray_directions=ray_directions.float()
        colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
        colors = colors.reshape(x.shape)
        sigma = sigma.reshape(x.shape[:-1])
        
        
        
        #compute accumulated transmittance, weights for each point
        alpha = 1 - torch.exp(-sigma * delta)
        accumulated_transmittance = torch.cumprod(1 - alpha, 1)
        accumulated_transmittance = torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=device),accumulated_transmittance[:, :-1]), dim=-1)
        weights = accumulated_transmittance.unsqueeze(2) * alpha.unsqueeze(2)
        
        
        #Computing final color
        c = (weights * colors).sum(dim=1)  
        weight_sum = weights.sum(-1).sum(-1)  
        #print(c + 1 - weight_sum.unsqueeze(-1))
        return c + 1 - weight_sum.unsqueeze(-1)







def train(images_tensor_train, ray_o_train, ray_d_train, images_tensor_val, ray_o_val, ray_d_val,
          nerf_model, optimizer, scheduler, hn=0, hf=1, nb_epochs=1000, nb_bins=192, H=100, W=100, 
          device=device, CheckPointPath="", batch_size=100):
    training_loss = []
    validation_loss = []
    last_100_batches_px_values = []
    

    n_batches = len(ray_o_train) // batch_size
    print(n_batches)
    
    train_batches = [(ray_o_train[i*batch_size:(i+1)*batch_size], ray_d_train[i*batch_size:(i+1)*batch_size], images_tensor_train[i*batch_size:(i+1)*batch_size]) for i in range(n_batches)]
    
    for epoch_cur in tqdm(range(nb_epochs)):
        epoch_training_loss = []
        last_100_batches=[]
        epoch_validation_loss = []
        batch_num=0
        
        
        
        for batch in train_batches:
            ray_origins = batch[0].to(device)
            ray_directions = batch[1].to(device)
            ground_truth_px_values = batch[2].to(device)

            rpx = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            
            train_loss = ((ground_truth_px_values - rpx) ** 2).sum()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_training_loss.append(train_loss.item())



        training_loss.append(sum(epoch_training_loss) / len(epoch_training_loss))
        print("epoch loss")
        print(sum(epoch_training_loss) / len(epoch_training_loss))


        scheduler.step()


        
        SaveName = CheckPointPath + str(epoch_cur) + "_model.ckpt"
        torch.save(
            {
                "epoch": epoch_cur,
                "model_state_dict": nerf_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


        with torch.no_grad():
            for i in range(100):
                val_ray_origins = ray_o_val[i*100*100:(i+1)*100*100]
                val_ray_directions = ray_d_val[i*100*100:(i+1)*100*100]
                test_image=None
                for j in range(0, H*W, batch_size):
                    val_ray_origins_batch = val_ray_origins[j:j+batch_size].to(device)
                    val_ray_directions_batch = val_ray_directions[j:j+batch_size].to(device)
                    test_px = render_rays(nerf_model, val_ray_origins_batch, val_ray_directions_batch, hn=hn, hf=hf, nb_bins=nb_bins)
                    if test_image is None:
                        test_image = test_px
                    else:
                        test_image = torch.cat((test_image, test_px), dim=0)
                test_image=test_image.cpu().numpy().reshape((H, W, 3))
                cv2.imwrite(f"./Images/{i}.png", test_image)   


        
        plt.figure()
        plt.plot(training_loss, '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training'])
        plt.title('Loss vs. Epoch')
        plt.savefig('LossCurve.png')


    return training_loss





def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.is_available())

    



    model = NeRF(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



    json_file_train = './transforms_train.json'

    images_tensor_train,ray_o_train,ray_d_train=get_all_data(json_file_train)
    print(torch.max(ray_o_train))
    print(torch.max(ray_d_train))
    print(images_tensor_train)


    json_file_val='./transforms_val.json'
    images_tensor_val,ray_o_val,ray_d_val=get_all_data(json_file_val)




    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    CheckPointPath='./Checkpoints/'

    batch_size = 100

  
    train(images_tensor_train, ray_o_train, ray_d_train, images_tensor_val, ray_o_val, ray_d_val,model, model_optimizer, scheduler, hn=0,hf=1,nb_epochs=1000, nb_bins=192, H=100, W=100,device=device, CheckPointPath='./Checkpoints/', batch_size=100)
    



if __name__ == "__main__":
    main()





