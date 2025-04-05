from abc import ABC, abstractmethod
from typing import List, Type
from gr_libs.environment.environment import EnvProperty, SUPPORTED_DOMAINS
from gr_libs.environment.utils.utils import domain_to_env_property
from gr_libs.ml.base.rl_agent import RLAgent

class Recognizer(ABC):
	def __init__(self, domain_name: str, env_name:str, collect_statistics=False, rl_agent_type: Type[RLAgent]=None):
		assert domain_name in SUPPORTED_DOMAINS
		self.rl_agent_type = rl_agent_type
		self.domain_name = domain_name
		self.env_prop_type = domain_to_env_property(self.domain_name)
		self.env_prop = self.env_prop_type(env_name)
		self.collect_statistics = collect_statistics

	@abstractmethod
	def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
		pass

class LearningRecognizer(Recognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def domain_learning_phase(self, base_goals: List[str], train_configs: List):
		self.original_train_configs = train_configs

# a recognizer that needs to train agents for every new goal as part of the goal adaptation phase (that's why it needs dynamic train configs)
class GaAgentTrainerRecognizer(Recognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@abstractmethod
	def goals_adaptation_phase(self, dynamic_goals: List[str], dynamic_train_configs):
		pass

	def domain_learning_phase(self, base_goals: List[str], train_configs: List):
		super().domain_learning_phase(base_goals, train_configs)

class GaAdaptingRecognizer(Recognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@abstractmethod
	def goals_adaptation_phase(self, dynamic_goals: List[str]):
		pass
