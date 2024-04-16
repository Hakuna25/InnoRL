from wiserl.utils.replay_buffer import OnPolicyReplayBuffer, OnPolicyBaseReplayBuffer
from wiserl.utils.mem_store import OffPolicyReplayBuffer, OffPolicyBaseReplayBuffer

def make_buffer(config, discrete, use_threads, type):
    if type == "offpolicy":
        if use_threads:
            return OffPolicyReplayBuffer(config, discrete=discrete)
        else:
            return OffPolicyBaseReplayBuffer(config, discrete=discrete)
    elif type == "onpolicy":
        if use_threads:
            return OnPolicyReplayBuffer(config, discrete)
        else:
            return OnPolicyBaseReplayBuffer(config, discrete)
    else:
        raise Exception("no such type")
