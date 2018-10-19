from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from torch.autograd import Variable
from test import Tester
import gym


def train(rank, args, input_model, max_episodes=100000):
    xse4e5 utility values initialization
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = create_env(args)
    env.seed(args.seed + rank)

    # player initialization
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    if args.model == 'MLP':
        player.model = A3C_MLP(
            player.env.observation_space.shape[0], player.env.action_space, args.stack_frames)
    if args.model == 'CONV':
        player.model = A3C_CONV(args.stack_frames, player.env.action_space)

    # load the input model to the player
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model.load_state_dict(input_model.state_dict())
    else:
        player.model.load_state_dict(input_model.state_dict())

    # initialize the player optimizer
    optimizer = None
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(player.model.parameters(), lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(player.model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(player.model.parameters(), lr=args.lr)

    # reset the environment and initialize the player state
    player.state = player.env.reset(args)
    player.state = torch.from_numpy(player.state).float()

    # If on GPU, do as GPU
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()

    player.model.train()
    
    last_iter = 0
    episode = 0
    # Start looping over episodes
    while episode < max_episodes:
        last_iter += 1

        # reset cx and hx if the enlvironmnent is over.
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 128).cuda())
                    player.hx = Variable(torch.zeros(1, 128).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 128))
                player.hx = Variable(torch.zeros(1, 128))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
            
        # Roll out actions and collect reward for one episode
        for step in range(args.num_steps):
            player.action_train()

            if player.done:
                break

        if player.done:
            episode += 1
            player.eps_len = 0
            # reset state
            state = player.env.reset(args)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = torch.zeros(1, 1).cuda()
        else:
            R = torch.zeros(1, 1)
        
        if not player.done:
            state = player.state
            if args.model == 'CONV':
                state = state.unsqueeze(0)
            value, _, _, _ = player.model(
                (Variable(state), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = torch.zeros(1, 1).cuda()
        else:
            gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (0.01 * player.entropies[i].sum())

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()
        player.clear_actions()

    tester = Tester(args, player.model)
    fitness = tester.test(last_iter)

    ## This part should be off for the Baldwin evolution
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            input_model.load_state_dict(player.model.state_dict())
    else:
        input_model.load_state_dict(player.model.state_dict())
    
    return fitness
    
