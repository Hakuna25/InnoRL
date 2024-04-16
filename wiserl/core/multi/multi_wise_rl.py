# -- coding: utf-8 --
import multiprocessing
from multiprocessing import Queue, Pipe
from .multi_agent_proxy import MultiAgentProxy
from multiprocessing import Manager
import os

def agent_run(agent):
    queue = agent.queue
    while True:
        data = queue.get()
        queue.task_done()
        # 获取方法名和参数
        method_name = data['method']
        params = data['args']
        kwargs = data['kwargs']
        runner_id = data['runner_id']
        # 使用getattr调用方法
        method = getattr(agent, method_name)
        re = method(*params,**kwargs)
        pipe = agent.get_runner_pipe(runner_id)[0]
        pipe.send(re)

def runner_run(runner):
    pipe = Pipe()
    runner.runner_pipe_dict[os.getpid()] = pipe
    runner.run()

class MultiWiseRL(object):
    def __init__(self, use_ray=False):
        multiprocessing.freeze_support()
        os.environ['MULTIPROCESSING_CONTEXT'] = 'fork'
        multiprocessing.set_start_method('spawn')
        self.manager = Manager()
        self.runner_pipe_dict =  self.manager.dict()  
        self.lock= multiprocessing.Lock()
   
    def make_runner(self, runner_class, args=None, num=1):
        runners = []
        for i in range(num):
            runner = runner_class(args,local_rank=i)
            self.lock.acquire()
            runner.runner_pipe_dict= self.runner_pipe_dict
            self.lock.release()
            # # runner.agent_queue_dict = self.agent_queue_dict
            p = multiprocessing.Process(target=runner_run, args=(runner,))
            runners.append(p)
        return runners

    def make_agent(self,name,agent_class,config=None,num=1, sync=True, resource=None) :
        agent =agent_class(config,sync)
        queue = self.manager.Queue()
        agent.set_name(name)
        agent.queue=queue
        agent.runner_pipe_dict= self.runner_pipe_dict
        p = multiprocessing.Process(target=agent_run,args=(agent,))
        p.start()
        copy_agent = None
        if sync == False:
            copy_agent =agent_class(config,sync)
            copy_agent.runner_pipe_dict=self.runner_pipe_dict
            copy_queue = self.manager.Queue()
            copy_name="copy_"+name
            copy_agent.set_name(copy_name)
            copy_agent.queue=copy_queue
            #self._add_queue('copy'+str(agent.get_pid()), queue)
            agent.set_copy_agent(MultiAgentProxy(copy_agent,None))
            copy_p = multiprocessing.Process(target=agent_run)
            copy_p.start()
        return MultiAgentProxy(agent,copy_agent)


    def start_all_runner(self, runners):
        for runner in runners:
            runner.start()
        for runner in runners:
           runner.join()
