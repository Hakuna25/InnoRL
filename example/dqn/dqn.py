from wiserl.core.wise_rl import WiseRLFactory
from wiserl.agent.dqn_agent.dqn_agent import DqnAgent
from wiserl.envs.env import make_env
from wiserl.core.runner import Runner
import time
import argparse
import multiprocessing

env = make_env("CartPole-v1")
# wise_rl = WiseRL()
class GymRunner(Runner):
    def __init__(self, args, local_rank=0):
        self.local_rank = local_rank
        self.total_steps = 0
        self.agent = args.agent
        self.args = args

    def run(self):
        episode = 0
        while self.total_steps < self.args.max_train_steps:
            s = env.reset()[0]
            ep_r = 0
            episode_steps = 0
            episode += 1
            while True:
                a = self.agent.choose_action(s)
                s_, _, r, done, truncated, _ = env.step(a)
                x, x_dot, theta, theta_dot = s_
                r1 = (env.env.x_threshold - abs(x)) / env.env.x_threshold - 0.8
                r2 = (env.env.theta_threshold_radians - abs(theta)) / env.env.theta_threshold_radians - 0.5
                r = r1 + r2
                ep_r += r

                self.agent.update(s, a, r, s_, done)
                if done:
                    break
                s = s_
                self.total_steps += 1
                episode_steps += 1
            if episode % 20 == 0:
                print('E_id: ', episode, ' |', 'Ep_r: ', round(ep_r, 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--net_dims", default=(256,256), help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
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
    args = parser.parse_args()
    wise_rl = WiseRLFactory.wiserl(use_ray=True)
    print("env",env)
    setattr(args, 'state_dim', env.state_space)
    setattr(args, 'action_dim', env.action_space)
    agent = wise_rl.make_agent(name="dqn", agent_class=DqnAgent, config=args, sync=True)
    setattr(args, 'agent', agent)
    runners = wise_rl.make_runner(GymRunner, args, num=1)
    wise_rl.start_all_runner(runners)
