import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from siren_pytorch import SirenNet
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageDataset(Dataset):

    def __init__(self, img, include_frac=1, start_idx=0, device='cpu', seed=10):
        # img is a 2D or 3D numpy array
        self.img = img
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.dimvec = np.array([self.img.shape]).astype(np.float32)
        self.include_frac = include_frac
        
        if len(self.img.shape) == 2:
            self.img = np.expand_dims(self.img, axis=-1)

        all_coords = np.array([[row, col] for col in range(self.width) for row in range(self.height)])
        random_perm = np.random.RandomState(seed=seed).permutation(len(all_coords))

        include_perm = random_perm[start_idx:int(len(all_coords) * include_frac + start_idx)]
        
        self.include_coords = all_coords[include_perm]
        self.norm_include_coords = self.include_coords / self.dimvec # normalize coords to the range [-1, 1]
        print(np.min(self.norm_include_coords, axis=0), np.max(self.norm_include_coords, axis=0))
        self.include_vals = self.img[self.include_coords[:, 0], self.include_coords[:, 1]]
        self.norm_include_coords = torch.tensor(self.norm_include_coords, dtype=torch.float32).to(device)
        self.include_vals = torch.tensor(self.include_vals, dtype=torch.float32).to(device)
        
        
        
    def __len__(self):
        return self.include_vals.shape[0]


    def __getitem__(self, i):
        # row = i // self.width
        # col = i % self.width
        # return torch.tensor([row / self.height, col / self.width], dtype=torch.float32), self.img[row, col, :]

        return self.norm_include_coords[i], self.include_vals[i]
    

    def plot(self, title='Training data given to model', filename='training_density_map.png'):
        img = np.zeros((self.height, self.width))
        for coord, val in zip(self.include_coords, self.include_vals):
            img[coord[0], coord[1]] = val
        plt.imshow(img)
        plt.title(f'{title} - {self.include_frac * 100}% of data')
        plt.colorbar()
        plt.savefig(filename)
        plt.cla()
        plt.clf()


class ImageHoldoutDataset(ImageDataset):
    def __init__(self, img, ranges, include_frac=1, start_idx=0, device='cpu', seed=10):
        super().__init__(img, include_frac, start_idx, device, seed)
        self.ranges = torch.tensor([ranges], dtype=torch.float32)
        self.step_area = torch.prod(self.ranges / self.dimvec)

    
    def __getitem__(self, i):
        # row = i // self.width
        # col = i % self.width
        # return torch.tensor([row / self.height, col / self.width], dtype=torch.float32), self.img[row, col, :]

        return self.norm_include_coords[i], self.step_area * self.include_vals[i]


def training_loop(siren, img_loader, epochs=10, lr=0.001):
    objective = nn.MSELoss()
    optimizer = torch.optim.Adam(params=siren.parameters(), lr=lr)
    
    loss_history = []
    print('training')
    for e in tqdm(range(epochs)):
        for batch_in, batch_out in img_loader:
            pred = siren(batch_in)
            loss = objective(pred, batch_out)
            loss_e = objective(pred, batch_out)
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_history


def training_loop_holdout(siren, img_loader, holdout_loader, epochs=10, lr=0.001):
    objective = nn.MSELoss()
    optimizer = torch.optim.Adam(params=siren.parameters(), lr=lr)
    
    loss_history = []
    print('training with holdout')
    for e in tqdm(range(epochs)):
        for (batch_in, batch_out), (batch_holdout_in, batch_holdout_out) in zip(img_loader, holdout_loader):
            pred_p = siren(batch_in)
            pred_e = siren(batch_holdout_in)

            loss_p = objective(pred_p, batch_out)
            loss_e = objective(torch.sum(pred_e), torch.sum(batch_holdout_out))

            loss = loss_p + loss_e
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return loss_history

def train_siren(siren, img, ranges=None, epochs=10, batch_size=16, lr=0.001, include_frac=1, holdout_frac=0, device='cpu'):
    
    img_dset = ImageDataset(img, include_frac=include_frac, device=device)
    img_dset.plot()
    img_loader = DataLoader(img_dset, batch_size=batch_size, shuffle=True)

    if holdout_frac > 0:
        holdout_dset = ImageHoldoutDataset(img, ranges, include_frac=holdout_frac, start_idx=len(img_dset), device=device)
        img_dset.plot(title='Training data given to holdout', filename='training_holdout_density_map.png')
        holdout_loader = DataLoader(holdout_dset, batch_size=batch_size, shuffle=True)
        loss_history = training_loop_holdout(siren, img_loader, holdout_loader, epochs=epochs, lr=lr)
    else:
        loss_history = training_loop(siren, img_loader, epochs=epochs, lr=lr)
    
    
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.title('siren training loss')
    plt.savefig('training_loss.png')
    plt.cla()
    plt.clf()
    


def gen_gaussians(means, vars, num_points=50):
    gaussians = []
    for mean, var in zip(means, vars):
        gaussians.append(np.random.multivariate_normal(
            mean=mean, cov=np.diag(var), size=(num_points,))
        )
    return np.concatenate(gaussians, axis=0)


def density_map(data, mins, maxes, step_size, window_rad):
    dmap = np.zeros((np.flip(maxes - mins) / step_size).astype(int))
    i_x = 0
    for x in tqdm(np.arange(mins[0], maxes[0] - step_size, step_size)):
        i_y = 0
        for y in np.arange(mins[1], maxes[1] - step_size, step_size):
            count = 0
            for i in range(len(data)):
                dist = np.sqrt(np.sum((data[i] - np.array([x, y])) ** 2))
                if dist <= window_rad:
                # if np.abs(x - data[i, 0]) < window_rad and np.abs(y - data[i, 1]) < window_rad:
                    dmap[i_y, i_x] = dmap[i_y, i_x] + 1
                    dmap[i_y, i_x] = dmap[i_y, i_x] + 1
            i_y += 1
        i_x += 1

    return np.flip(dmap, axis=0)


def prelim1():
    means = [[0, 0], [-2, 4]][0:2]
    vars = [[1, 1], [1, 1]][0:2]
    data = gen_gaussians(means, vars, num_points=500)
    mins = np.min(data, axis=0) # for example: xmin, ymin
    maxes = np.max(data, axis=0) # for example: xmax, ymax
    ranges = np.max(maxes - mins, axis=0) # for example: xmax, ymax
    min_range = np.amin(ranges)

    print(data.shape)
    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.xlim((np.amin(mins), np.amax(mins) + np.amax(ranges)))
    plt.ylim((np.amin(mins), np.amax(mins) + np.amax(ranges)))
    plt.title('Sample of points from which to learn the PDF')
    plt.savefig('data.png')
    plt.cla()
    plt.clf()

    min_steps = 40
    step_size = min_range / min_steps

    window_rads = [step_size * (i + 1) for i in range(2, 3, 1)]
    # min_steps = [5, 10, 20, 40]
    dmaps = [
        density_map(data, mins, maxes, step_size, rad)
        for rad in window_rads
    ]
    dmaps = [dmap / (np.sum(dmap) * (step_size ** 2)) for dmap in dmaps]
    
    for i, (dmap, rad) in enumerate(zip(dmaps, window_rads)):
        # plt.subplot(len(dmaps), 1, i + 1)
        plt.imshow(dmap)
        plt.colorbar()
        plt.title(f'Probability Density - window_sidelength={rad * 2}')
        plt.savefig(f'density_map_{i + 1}.png', dpi=150)
        plt.cla()
        plt.clf()

    # dmaps = np.stack(dmaps, axis=0)
    # mean_dmap = np.mean(dmaps, axis=0)
    # plt.imshow(mean_dmap)
    # plt.colorbar()
    # plt.savefig('mean_density_map.png', dpi=150)


    siren = SirenNet(
        dim_in = 2,                        # input dimension, ex. 2d coor
        dim_hidden = 64,                  # hidden dimension
        dim_out = 1,                       # output dimension, ex. rgb value
        num_layers = 5,                    # number of layers
        final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
        w0_initial = 20.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
    )

    train_img = dmaps[0]
    # train_siren(siren, train_img, ranges=ranges, epochs=10, batch_size=16, include_frac=0.5, holdout_frac=0.5, device='cpu')
    train_siren(siren, train_img, epochs=10, batch_size=16, include_frac=1, device='cpu')

    with torch.no_grad():
        siren.eval()
        superres_factor = 10
        new_height = train_img.shape[0] * superres_factor
        new_width = train_img.shape[1] * superres_factor
        out_img = np.zeros((new_height, new_width))
        row_stepsize = (1 / new_height)
        col_stepsize = (1 / new_width)

        print('predicting')
        # print(out_img.shape, len(np.arange(-1, 1, row_stepsize)), len(np.arange(-1, 1, col_stepsize)))

        row_i = 0
        for row in tqdm(np.arange(0, 1, row_stepsize)):
            col_i = 0
            for col in np.arange(0, 1, col_stepsize):
                data_in = torch.tensor([[row, col]], dtype=torch.float32)
                data_out = siren(data_in)[0].item()
                out_img[row_i, col_i] = data_out
                col_i += 1
            row_i += 1
        
        plt.imshow(out_img)
        plt.colorbar()
        plt.title(f'Predicted Probability Density - super resolution x{superres_factor}')
        # plt.show()
        plt.savefig('predicted_density_map.png')






if __name__ == '__main__':
    prelim1()


