import os
class MultiAgentProxy(object):
    def __init__(self,agent,copy_agent=None):
       
        self.agent = agent
        self.copy_agent = copy_agent

    def choose_action(self, *args,**kwargs):
        queue = None
        if self.copy_agent != None:
            queue = self.copy_agent.queue
        else:
            queue = self.agent.queue
        data = {
            "method": "choose_action",
            "args":  args,
            "kwargs": kwargs,
            "runner_id": os.getpid()
        }
      
        queue.put(data)
        pipe = self.agent.get_runner_pipe(os.getpid())[1]
        re = pipe.recv()
        return re



    def update(self,*args,**kwargs):
    
        queue = self.agent.queue
        
        data = {
            "method": "update",
            "args":  args,
            "kwargs": kwargs,
            "runner_id": os.getpid()
        }
        queue.put(data)
        pipe = self.agent.get_runner_pipe(os.getpid())[1]
        re = pipe.recv()
        return re
    
    
    def _update_model(self,*args, **kwargs):
        queue = self.copy_agent.queue
        data = {
            "method": "_update_model",
            "args":  args,
            "kwargs": kwargs,
            "runner_id":os.getpid()
        }
        queue.put(data)
        pipe = self.agent.get_runner_pipe(os.getpid())[1]
        re = pipe.recv()
        return re