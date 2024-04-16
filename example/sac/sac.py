from wiserl.agent.sac_agent.sac_agent import SACAgent
from wiserl.core.wise_rl import WiseRLFactory
import multiprocessing
import argparse
from wiserl.envs.env_tool import *
from wiserl.core.runner import Runner


class GymRunner(Runner):
    def __init__(self, args=None, local_rank=0):
        seed = 0
        self.agent = args.agent
        self.wsenv = args.env
        self.config = args
        self.local_rank = local_rank
        self.n_agents = 1
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self):
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        ep_r = 0
        for i in range(500):
            state = self.wsenv.reset()[0]
            for t in range(300):
                action = self.agent.choose_action(state)
                next_state, observation_obs, reward, done, info, available_actions = self.wsenv.step(action)
                rewards = np.mean(reward)
                ep_r += rewards
                self.agent.update(state, action, reward, next_state, done)
                done = np.any(done)
                state = next_state
                if done or t == 299:
                    if self.local_rank == 0:
                        print("Ep_i: {},  ep_r: {}, time_step: {}".format(i, ep_r, t))
                    break
            ep_r = 0


if __name__ == '__main__':
    print("use GPU to train" if torch.cuda.is_available() else "use CPU to train")
    parser = argparse.ArgumentParser("Hyperparameter Setting for SAC")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=int(200), help=" Maximum number of episode steps")
    parser.add_argument("--net_dims", default=(256, 128),
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--sigma", type=float, default=0.01, help="Gaussian factor")
    parser.add_argument("--tau", type=float, default=0.005, help="factor for soft-update")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    parser.add_argument("--use_ray", type=bool, default=True, help="use ray or not")
    parser.add_argument("--ray_num", type=int, default=1, help="num of ray")
    parser.add_argument("--use_threads", type=bool, default=True, help="env use threads or not")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="num of threads")
    parser.add_argument("--env_args", type=str, default="Pendulum-v1", help="env name")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    args = parser.parse_args()

    wise_rl = WiseRLFactory.wiserl(use_ray=args.use_ray)
    wsenv = make_wise_env("gym", args.n_rollout_threads, args.env_args, args.use_threads)
    setattr(args, 'state_space', wsenv.state_space)
    setattr(args, 'action_space', wsenv.action_space)
    setattr(args, 'action_bound', 2.0)
    setattr(args, 'n_agents', 1)
    agent = wise_rl.make_agent(name="sac", agent_class=SACAgent, config=args, sync=True)
    setattr(args, 'agent', agent)
    setattr(args, 'env', wsenv)
    runners = wise_rl.make_runner(GymRunner, args, num=1)
    wise_rl.start_all_runner(runners)
