import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs
import torch.nn as nn
import matplotlib.pyplot as plt

import shutil
import argparse

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        for layer in self.net:
            out = layer(out)
        return out

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class PINN_cavity:
    rho = 1
    mu = 0.02
    
    # cavity
    lid_velocity = 1
    L = 0.1
    nu = 0.05
    Re = lid_velocity * L/nu

    def __init__(self, ub, lb, Re=100) -> None:
        self.net = DNN(dim_in=2, dim_out=3, n_layer=4, n_node=40, ub=ub, lb=lb).to(device)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": [], "f0": [], "f1": [], "f2": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def bc_loss(self, xy, uv_bnd):
        u, v = self.predict(xy)[0:2]
        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + torch.mean(
            torch.square(v - uv_bnd[:, 1:2])
        )
        return mse_bc

    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        p_out = grad(p.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]

        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        p_x = p_out[:, 0:1]
        p_y = p_out[:, 1:2]

        u_xx = grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
        v_xx = grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]

        # continuity equation
        f0 = u_x + v_y

        # navier-stokes equation
        f1 = u*u_x + v*u_y - (1/self.Re) * (u_xx + u_yy) + p_x
        f2 = u*v_x + v*v_y - (1/self.Re) * (v_xx + v_yy) + p_y

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_pde = mse_f0 + mse_f1 + mse_f2

        return mse_pde, mse_f0.detach().cpu().item(), mse_f1.detach().cpu().item(), mse_f2.detach().cpu().item()

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(self.xy_bnd, self.uv_bnd)
        #mse_outlet = self.outlet_loss(xy_outlet)
        mse_pde, mse_f0, mse_f1, mse_f2 = self.pde_loss(self.xy_col)
        #loss = mse_bc + mse_outlet + mse_pde
        loss = mse_bc + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        #self.losses["outlet"].append(mse_outlet.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["f0"].append(mse_f0)
        self.losses["f1"].append(mse_f1)
        self.losses["f2"].append(mse_f2)
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss
    
    def assign_dataset(self,xy_bnd,xy_col,uv_bnd):
        self.xy_bnd = xy_bnd
        self.xy_col = xy_col
        self.uv_bnd = uv_bnd



def plotLoss(losses_dict, info=["BC", "PDE", "f0", "f1", "f2"]):
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(5), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()

def getData_cavity(N_b,N_w,N_s,N_c,N_r):
    inlet_x = np.random.uniform(x_min, x_max, (N_b, 1))
    inlet_y = np.ones((N_b, 1))*y_max
    inlet_u = np.ones((N_b, 1))
    inlet_v = np.zeros((N_b, 1))
    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    lwall_xy = np.random.uniform([x_min, y_min], [x_min, y_max], (N_w, 2))
    rwall_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_w, 2))
    bwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    lwall_uv = np.zeros((N_w, 2))
    rwall_uv = np.zeros((N_w, 2))
    bwall_uv = np.zeros((N_w, 2))

    # cylinder surface, u=v=0
    # theta = np.linspace(0.0, 2 * np.pi, N_s)
    # cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    # cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    # cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
    # cyl_uv = np.zeros((N_s, 2))

    # all boundary except outlet
    xy_bnd = np.concatenate([inlet_xy, lwall_xy, rwall_xy, bwall_xy], axis=0) #, cyl_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, lwall_uv, rwall_uv, bwall_uv], axis=0) #, cyl_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # refine points around cylider
    # refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    # refine_lb = np.array([xc - 2 * r, yc - 2 * r])

    # xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    # xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder
    # dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    # xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

    # concatenate all xy for collocation
    xy_col = np.concatenate((xy_col, xy_bnd), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    return xy_col, xy_bnd, uv_bnd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP PINN Training Study')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--Re', type=float, default='test')
    args = parser.parse_args()

    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0
    # r = 0.05
    # xc = 0.5
    # yc = 0.5

    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])

    ### Data Prepareation ###
    N_b = 200  # inlet & outlet
    N_w = 400  # wall
    N_s = 200  # surface
    N_c = 40000  # collocation
    N_r = 10000

    xy_col, xy_bnd, uv_bnd = getData_cavity(N_b,N_w,N_s,N_c,N_r)
    pinn = PINN_cavity(ub=ub,lb=lb,Re=args.Re)

    pinn.assign_dataset(xy_bnd, xy_col, uv_bnd)
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    torch.save(pinn.net.state_dict(), "model_weights_adam.pt")
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "model_weights.pt")
    plotLoss(pinn.losses)

    np.save('training_losses.npy',pinn.losses, allow_pickle=True)
    
    shutil.copyfile('training_losses.npy'   , '/content/drive/MyDrive/Results/'+ args.name + '.npy')
    shutil.copyfile("model_weights.pt"      , '/content/drive/MyDrive/Results/'+ args.name + '.pt')