from wiserl.agent.ppo_agent.ppo_agent import PPOAgent
from wiserl.core.wise_rl import WiseRLFactory
import multiprocessing
import argparse
from wiserl.utils.store_utils import make_buffer
from wiserl.envs.env_tool import *
from wiserl.core.runner import Runner
from wiserl.utils.normalization import Normalization

class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        seed = 0
        self.agent = args.agent
        self.replay_buffer = make_buffer(args, discrete=True, use_threads=args.use_threads, type="onpolicy")
        self.wsenv = args.env
        self.config = args
        self.local_rank = local_rank
        self.n_agents = 1
        self.state_norm = Normalization(shape=self.config.state_space)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training

        while total_steps < self.config.max_train_steps:
            state = self.wsenv.reset()[0]
            d = False
            episode_reward = 0
            while not d:
                action, action_logprob = self.agent.choose_action(state)  # Action and the corresponding log probability
                state_, observation_obs, reward, done, info, available_actions = self.wsenv.step(action)
                if self.config.use_state_norm:
                    state_ = self.state_norm(state_)
                if self.config.use_reward_scaling: # Trick 4: reward scaling
                    reward *= self.config.scale_factor
                reward_env = np.mean(reward)
                episode_reward += reward_env
                d = np.any(done)
                if d and total_steps != self.config.max_train_steps:
                    dw = True
                else:
                    dw = False
                done = np.any(done)
                self.replay_buffer.store(state, action, action_logprob, reward, state_, dw, done)
                state = state_
                total_steps += 1
                # When the number of transitions in buffer reaches batch_size,then update
                if self.replay_buffer.count == self.config.batch_size:
                    self.agent.update(self.replay_buffer, total_steps)
                    self.replay_buffer.count = 0
                if self.local_rank == 0 and total_steps % 500 == 0:
                    print(self.local_rank, ' Ep: ', total_steps, ' |', 'Ep_r: ', episode_reward)

if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    parser = argparse.ArgumentParser("Hyperparameter Setting for SAC")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=6, help="Mini Batch size")
    parser.add_argument("--net_dims", default=(256, 128),
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="epoch_num")
    parser.add_argument("--scale_factor", type=int, default=2, help="scale_factor")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    parser.add_argument("--use_ray", type=bool, default=True, help="use ray or not")
    parser.add_argument("--ray_num", type=int, default=1, help="num of ray")
    parser.add_argument("--use_threads", type=bool, default=True, help="env use threads or not")
    parser.add_argument("--n_rollout_threads", type=int, default=2, help="num of threads")
    parser.add_argument("--env_args", type=str, default="CartPole-v1", help="env name")
    args = parser.parse_args()

    wise_rl = WiseRLFactory.wiserl(use_ray=args.use_ray)
    wsenv = make_wise_env("gym", args.n_rollout_threads, args.env_args, args.use_threads)
    setattr(args, 'state_space', wsenv.state_space)
    setattr(args, 'action_space', wsenv.action_space)
    setattr(args, 'action_bound', 2.0)
    setattr(args, 'n_agents', 1)
    agent = wise_rl.make_agent(name="ppo", agent_class=PPOAgent, config=args, sync=True)
    setattr(args, 'agent', agent)
    setattr(args, 'env', wsenv)
    runners = wise_rl.make_runner(GymRunner, args, num=1)
    wise_rl.start_all_runner(runners)
