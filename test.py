from __future__ import division
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_CONV, A3C_MLP
from player_util import Agent
import time
import logging

class Tester:

    def __init__(self, args, shared_model, log_level=0):
        """
        args - command line argument object
        shared_model - model that you want to test
        log_level - 0 = log nothing, just run
                    1 = log only the cummulative reward to the log file
                    2 = log the cummulative reward to the log file and stdout
                    3 = log everything
        """
        self.args = args
        self.shared_model = shared_model

        self.gpu_id = self.args.gpu_ids[-1]
        self.reward_sum = 0
        self.start_time = time.time()
        self.reward_sum = 0
        self.num_tests = 0
        self.reward_total_sum = 0
        self.max_score = -float("inf")
        self.log = {}
        #setup_logger('{}_log'.format(self.args.env),
        #             r'{0}_{1}_{2}_{3}_log'.format(self.args.log_dir, self.args.env, self.args.scale_legs, time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
        #self.log['{}_log'.format(self.args.env)] = logging.getLogger(
        #    '{}_log'.format(self.args.env))
        d_args = vars(self.args)
        #for k in d_args.keys():
        #    self.log['{}_log'.format(self.args.env)].info('{0}: {1}'.format(k, d_args[k]))

        torch.manual_seed(self.args.seed)
        if self.gpu_id >= 0:
            torch.cuda.manual_seed(self.args.seed)

    def test(self, iteration, show='none', save_max=False):
        env = create_env(self.args)
 
        player = Agent(None, env, self.args, None)
        player.gpu_id = self.gpu_id
        if self.args.model == 'MLP':
            player.model = A3C_MLP(
                player.env.observation_space.shape[0], player.env.action_space, self.args.stack_frames)
        if self.args.model == 'CONV':
            player.model = A3C_CONV(self.args.stack_frames, player.env.action_space)

        # load the input model
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                player.model.load_state_dict(self.shared_model.state_dict())
        else:
            player.model.load_state_dict(self.shared_model.state_dict())
        
        player.state = player.env.reset(self.args)
        player.state = torch.from_numpy(player.state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                player.model = player.model.cuda()
                player.state = player.state.cuda()
        player.model.eval()

        while True:
            player.action_test()
            if self.args.show != 'none' or show != 'none':
                player.env.render()

            self.reward_sum += player.reward

            if player.done:
                self.num_tests += 1
                self.reward_total_sum += self.reward_sum
                reward_mean = self.reward_total_sum / self.num_tests
                self.reward_sum = 0
                player.eps_len = 0
                state = player.env.reset(self.args)
                player.state = torch.from_numpy(state).float()
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        player.state = player.state.cuda()
                if self.args.show != 'none' or show != 'none':
                    player.env.close()
                break
        return self.reward_total_sum