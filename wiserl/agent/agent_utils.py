import torch
from wiserl.net.nn_net import ActorDiscretePPO, ActorPPO, ActorDDPG, CriticDDPG, ActorContinuousPPO, ActorTD3, ActorTRPO
from wiserl.net.nn_net import CriticPPO, CriticPPO2, CriticTD3, CriticTRPO
from wiserl.net.sac_net import SACActor, SACCritic, SACQnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer == "mbgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    if optimizer == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise Exception("no such optimizer")


def make_actor_net(net_name, config):
    if net_name == "dis_nn":
        return ActorDiscretePPO(config.net_dims, config.state_space, config.action_space)
    elif net_name == "ppo2_nn":
        return ActorContinuousPPO(config["state_space"], config["action_space"], config["net_width"]).to(device)
    elif net_name == "nn":
        return ActorPPO(config.net_dims, config.state_space, config.action_space)
    elif net_name == "ddpg_nn":
        return ActorDDPG(config.state_space, config.hidden_dim, config.action_space, config.action_bound).to(device)
    elif net_name == "sac_nn":
        return SACActor(config.state_space).to(device)
    elif net_name == "td3_nn":
        return ActorTD3(config.state_space, config.action_space, config.action_bound).to(device)
    elif net_name == "trpo_nn":
        return ActorTRPO(config.state_space, config.action_space).to(device)
    raise Exception("no such actor network")


def make_q_net(net_name, config):
    if net_name == "sac_qnet":
        return SACQnet(config.state_space, config.action_space).to(device)
    raise Exception("no such actor network")


def make_critic_net(net_name, config):
    if net_name == "nn":
        return CriticPPO(config.net_dims, config.state_space, config.action_space)
    elif net_name == "ppo2_nn":
        return CriticPPO2(config["state_space"], config["net_width"]).to(device)
    elif net_name == "ddpg_nn":
        return CriticDDPG(config.state_space, config.hidden_dim, config.action_space).to(device)
    elif net_name == "sac_nn":
        return SACCritic(config.state_space).to(device)
    elif net_name == "td3_nn":
        return CriticTD3(config.state_space, config.action_space).to(device)
    elif net_name == "trpo_nn":
        return CriticTRPO(config.state_space).to(device)
    raise Exception("no such critic network")

