from abc import ABC, abstractmethod

from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import SUPPORTED_DOMAINS
from gr_libs.ml.base.rl_agent import RLAgent


class Recognizer(ABC):
    def __init__(
        self,
        domain_name: str,
        env_name: str,
        collect_statistics=False,
        rl_agent_type: type[RLAgent] = None,
        **kwargs,
    ):
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
    """
    A class that represents a learning recognizer.

    Inherits from the Recognizer class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def domain_learning_phase(self, train_configs: list, base_goals: list[str] = None):
        """
        Perform the domain learning phase.

        Args:
            base_goals (List[str]): The base goals for the learning phase.
            train_configs (List): The training configurations.

        """
        self.original_train_configs = train_configs


# a recognizer that needs to train agents for every new goal as part of the goal adaptation phase (that's why it needs dynamic train configs)
class GaAgentTrainerRecognizer(Recognizer):
    """
    A class representing a recognizer for GaAgentTrainer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def goals_adaptation_phase(self, dynamic_goals: list[str], dynamic_train_configs):
        """
        Perform the goals adaptation phase.

        Args:
            dynamic_goals (List[str]): The list of dynamic goals.
            dynamic_train_configs: The dynamic training configurations.

        Returns:
            None
        """

    def domain_learning_phase(self, train_configs: list, base_goals: list[str] = None):
        """
        Perform the domain learning phase.

        Args:
            train_configs (List): List of training configurations.
            base_goals (List[str]): List of base goals for the learning phase.

        Returns:
            None
        """
        super().domain_learning_phase(train_configs, base_goals)


class GaAdaptingRecognizer(Recognizer):
    """
    A recognizer that doesn't require more training given a set of new goals, hence it doesn't receive train configs in the goal adaptation phase.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def goals_adaptation_phase(self, dynamic_goals: list[str]):
        """
        Perform the goals adaptation phase.

        Args:
            dynamic_goals (List[str]): A list of dynamic goals to be adapted.

        Returns:
            None
        """
