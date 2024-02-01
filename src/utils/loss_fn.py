import torch
import torch.nn as nn


def l_distil(b_g, b_lip):
    diff = 0
    loss = torch.nn.MSELoss()
    for t in range(len(b_g)):
        l2_norm = torch.norm(b_g[t] - b_lip[t])
        diff += torch.square(l2_norm)
        # diff += torch.square(b_g[t] - b_lip[t])
    return (1 / (len(b_g))) * diff
    # return loss(b_g, b_lip)


def l_eye(p, z_blink):
    loss = 0
    for t in range(len(p)):
        e_w = (torch.linalg.norm(p[t][39]-p[t][36]) + torch.linalg.norm(p[t][45]-p[t][42]))/2
        e_h = (torch.linalg.norm(p[t][37]+p[t][38]-p[t][40]-p[t][41]) + torch.linalg.norm(p[t][43]+p[t][44]-p[t][46]-p[t][47]))/2
        r_t = e_h/e_w
        loss += torch.norm(r_t - z_blink)

    # l_sum = 0
    # for t in range(len(r)):
    #     l_sum += abs(r[t] - z_blink[t])
    return loss


def l_lks(p, p_dash, z_blink, lambda_eye=200):
    loss = 0
    # eye area = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    M = [i for i in range(len(p[0])) if i<36 or i>47]
    for t in range(len(p)):
        for m in M:
            loss += torch.square(torch.linalg.norm(p[t][m] - p_dash[t][m]))
    
    return lambda_eye*l_eye(p, z_blink) + loss


def l_read(c_gt, c_p):
    criterion = nn.CrossEntropyLoss()
    M = [i for i in range(48, len(c_p[0]))]
    loss = 0
    for m in range(len(c_p)-5):
        loss += criterion(c_gt[m:m+5], c_p[m:m+5])
    return loss


# noinspection PyShadowingNames
def l_exp(lambda_distil, lambda_read, lambda_lks, l_distil, l_eye, l_lks):
    sum_exp = lambda_distil * l_distil + lambda_read * l_eye + lambda_lks * l_lks
    return sum_exp
