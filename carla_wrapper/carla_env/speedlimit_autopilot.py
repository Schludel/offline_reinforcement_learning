from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner

class SpeedAutopilot(RoamingAgent):

    def __init__(self, vehicle):
        super(RoamingAgent, self).__init__(vehicle)

        target_speed = 15
        self._local_planner = LocalPlanner(self._vehicle, opt_dict={'target_speed': target_speed})