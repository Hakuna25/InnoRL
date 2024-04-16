# -- coding: utf-8 --
import time
import uuid
from  .registre_server import RegistreServer
import ray
from .ray_agent_proxy import AgentProxy

class RayWiseRL(object):
    def __init__(self):
        self.registre = RegistreServer.remote()

    def make_runner(self, Runner, args=None, num=1,resource=None):  
        name = str(uuid.uuid1())
        for i in range(num):
            runner =ray.remote(Runner)
            if resource != None:
                runner = runner.options(**resource)
            runner =runner.remote(args,local_rank=i)
            self.registre.add_runner.remote(name,runner)
            runner.set_registre.remote(self.registre)
        retref = self.registre.get_all_runner.remote(name)
        return ray.get(retref)

    def get_runner(self, name):
        return ray.get(self.registre.get_runner.remote(name))

    def make_agent(self,name,agent_class,config=None, sync=True, resource=None) :
        copy_name= None
        copy_agent = None
        if sync == False:
            copy_name="_wise_copy_" + name + str(uuid.uuid1())
            copy_agent =ray.remote(agent_class)
            if resource != None:
                copy_agent = copy_agent.options(**resource)
            copy_agent = copy_agent.remote(config,sync,user_ray=True)
            self.registre.add_agent.remote(copy_name,copy_agent,copy_agent)
            copy_agent.set_registre.remote(self.registre)
                  
        agent = ray.remote(agent_class)
        if resource != None:
            agent = agent.options(**resource)
        agent = agent.remote(config,sync)
        self.registre.add_agent.remote(name,agent,copy_agent)

        agent.set_registre.remote(self.registre)
        if sync == False:
            remoteAgent = AgentProxy(agent ,copy_agent) 
            agent.set_copy_agent.remote(remoteAgent)
        retref = self.registre.get_agent.remote(name)
        return ray.get(retref)

    def get_agent(self, name):
        for i in range(100):
            agent =ray.get(self.registre.get_agent.remote(name))
            if agent != None:
                return agent
            time.sleep(1)
        raise ValueError(name + " agent not found ,please check that the name is correct")

    def start_all_runner(self, runners):
        results =[]
        for runner in runners:
            ref = runner.run.remote()
            results.append(ref)
        ray.get(results)