import os
class Runner(object):
    def __init__(self, sync, use_ray=False):
        super().__init__()
        self.pipe=None
        self.runner_pipe_dict=None
        self.set_registre = None

    def run():
        pass
    def get_pid(self):
        return os.getpid()
   
    def set_pipe(self, pipe):
        self.pipe = pipe

    def get_pipe(self):
        return self.pipe
    
      
    def set_runner_pipe_dict(self, dict):
        self.runner_pipe_dict = dict

    # def set_agent_queue_dict(self,dict):
    #     self.agent_queue_dict = dict

    def get_runner_pipe(self,pid):
        #print("self.runner_pipe_dict",self.runner_pipe_dict, pid)
        return self.runner_pipe_dict[pid]
    
    def set_registre(self,registre):
        self.set_registre = registre