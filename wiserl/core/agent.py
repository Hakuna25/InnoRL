# -- coding: utf-8 --False
import os
class Agent(object):  
    def __init__(self, sync, use_ray=False):
        super().__init__()
        self.use_ray = use_ray
        self.copy_agent = None
        self.sync = sync 
        self.runner_pipe_dict=None
        self.set_registre = None
        # self.agent_queue_dict =None
        self.name =None
        self.queue = None

    def set_name(self,name):
        self.name = name

    def choose_action(self, *args,**kwargs):
        pass

    def update(self,*args,**kwargs):
        pass

    def get_copy_agent(self):
        return self.copy_agent
    
    def set_copy_agent(self,copy_agent):
        self.copy_agent = copy_agent

    def _sync_model(self):
        pass

    def _update_model(self,param):
        pass

    def _fire(self,*args,**kwargs):
       ref = self.copy_agent._update_model(*args, **kwargs)
       
    # def set_queue(self,queue):
    #     self._queue = queue 
    #     print("set queue88888888888888888888888888888888888888888888888888888888888",self._queue)    

    # def get_queue(self):
    #    return self._queue

    def get_pid(self):
        return os.getpid()
    
    def set_runner_pipe_dict(self, dict):
        self.runner_pipe_dict = dict

    # def set_agent_queue_dict(self,dict):
    #     self.agent_queue_dict = dict

    def get_runner_pipe(self,pid):
        return self.runner_pipe_dict[pid]
    
    # def get_agent_queue(self,pid):
    #     print("agent_queue_dict",self.agent_queue_dict, pid)
    #     return self.agent_queue_dict[pid]
    def set_registre(self,registre):
        self.set_registre = registre