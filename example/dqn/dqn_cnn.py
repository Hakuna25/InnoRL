
from wiserl.agent.dqn_agent.dqn_agent import DqnAgent
from wiserl.core.multi.multi_wise_rl import MultiWiseRL
from wiserl.env import make_env 
import time
import argparse
import configparser
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2



wise_rl = MultiWiseRL()
class GymRunner(Runner):
    def __init__(self, args,local_rank=0):
        self.local_rank = local_rank
        self.agent_name ="dqn_agent"
        self.env = make_env(args.env_name)
        self.config = args
        setattr(self.config, 'state_dim',[4,84,84])
        setattr(self.config, 'action_dim', self.env.action_dim)
        self.stack_size = 4
        self.stacked_frames = deque([np.zeros((84,84), dtype=np.int64) for i in range(self.stack_size)], maxlen=4)
        self.total_steps = 0
        if local_rank == 0:
            wise_rl.make_agent(name=self.agent_name, agent_class=DqnAgent,config = self.config)
            self.agent = wise_rl.get_agent(self.agent_name)
        else:
            self.agent = wise_rl.get_agent(self.agent_name)

    def save_gym_state(self,env,i):
        next_state = env.render()
        #print("next_state=====",next_state.shape)
        plt.imsave('./image/state{}.png'.format(i),self.preprocess_frame(next_state)) 
        next_state = cv2.imread('./image/state{}.png'.format(i), cv2.IMREAD_GRAYSCALE)
        return next_state

    def preprocess_frame(self,frame):
        #print("frame=====",frame.shape)
        gray = rgb2gray(frame)
        #crop the frame
        #cropped_frame = gray[:,:]
        normalized_frame = gray/255.0
        preprocessed_frame = transform.resize(normalized_frame, [84,84])
        #print("preprocessed_frame",preprocessed_frame.shape)
        return preprocessed_frame

    def stack_frames(self, frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros((84,84), dtype=np.int64) for i in range(self.stack_size)], maxlen=4)
        
            # Because we're in a new episode, copy the same frame 4x
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            # frames =[]
            # for i in range(4):
            #     print("i",i)
            #     print("i",self.stacked_frames[i])
            #     frames.append(self.stacked_frames[i])
            # print(" self.stacked_frames", frames)
            # # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=0)
        
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)
           
            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=0)

        # print("self.stacked_frames",self.stacked_frames[2].shape)
        # print("self.stacked_frames",self.stacked_frames[3].shape)
        return stacked_state.flatten()

    def run(self):
        start = time.time()
        while self.total_steps < args.max_train_steps:
            self.env.reset()
            state_frame = self.save_gym_state(self.env,'init')
            state = self.stack_frames(state_frame, True)
            ep_r = 0
            i = 0 
            while True:
                a = self.agent.choose_action(state)
                s_, r, done, info, _ = self.env.step(a)
                next_state_frame = self.save_gym_state(self.env, i)
                next_state = self.stack_frames( next_state_frame, False)
                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.env.x_threshold - abs(x)) / self.env.env.x_threshold - 0.8
                r2 = (self.env.env.theta_threshold_radians - abs(theta)) / self.env.env.theta_threshold_radians - 0.5
                r = r1 + r2
                ep_r += r
                #print("r", r,done)
                self.agent.update(state, a, r, next_state, done)
                if done:
                    r = -10
                    end = time.time()
                    print(self.local_rank, 'time', round((end - start), 2), ' Ep: ', self.total_steps, ' |', 'Ep_r: ',
                          round(ep_r, 2))
                    break
                state = next_state
                i += 1
                self.total_steps += 1


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
#     parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
#     parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
#     parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
#     parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
#     parser.add_argument("--net_dims", default=(256,256), help="The number of neurons in hidden layers of the neural network")
#     parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
#     parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
#     parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
#     parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
#     parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
#     parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
#     parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
#     parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
#     parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
#     parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
#     parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
#     parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
#     parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
#     args = parser.parse_args()
def main(args):
    runners = wise_rl.make_runner("runner", GymRunner,args, num=5)
    wise_rl.start_all_runner(runners)

