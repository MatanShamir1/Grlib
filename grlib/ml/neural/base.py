import os.path
import pickle

import numpy
import torch
from abc import abstractmethod
import time
import tensorboardX
from grlib import ml
import ml.utils as utils
import ml.consts as ml_consts
from ml.utils import device

from ml.neural.acmodel import ACModel
from ml.base import RLAgent
from ml.neural.utils import DictList, ParallelEnv
import env_maker

def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)

class BaseAlgo(RLAgent):
    """The base class for RL algorithms."""

    def __init__(self, num_frames_per_proc, gamma, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, episodes,
                 decaying_epsilon, epsilon, problem_name, env_name, algo: str, seed: int, procs: int, use_text: bool,
                 argmax: bool, goal_hypothesis: str):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        super().__init__(
            episodes=episodes,
            decaying_eps=decaying_epsilon,
            epsilon=epsilon,
            learning_rate=lr,
            gamma=gamma,
            problem_name=problem_name,
            env_name=env_name,
            goal_hypothesis=goal_hypothesis
        )
        # Store parameters

        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.procs = procs
        self.optimizer = None
        self.argmax = argmax
        self.algo = algo

        # ACModel
        use_memory = recurrence > 1

        model_name = problem_name
        self.model_dir = utils.get_model_dir(
            env_name=env_name,
            model_name=problem_name,
            class_name=self.class_name()
        )
        self.txt_logger = utils.get_txt_logger(self.model_dir)

        self._states_seen_file = os.path.join(self.model_dir, "states_seen.pkl")
        self._load_states_seen_from_file()

        utils.seed(seed)

        # device
        self.txt_logger.info(f"Device: {device}\n")

        # Load environments

        self.envs = []
        for i in range(self.procs):
            # self.envs.append(utils.make_env(env_key=model_name, seed=seed + 10000 * i))
            self.envs.append(env_maker.make(env_name=model_name))
        self.env = ParallelEnv(self.envs)

        self.txt_logger.info("Environments loaded\n")

        # Load training status

        try:
            status = utils.get_status(self.model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        self.txt_logger.info("Training status loaded\n")

        self.status = status

        # Load observations preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(self.envs[0].observation_space)
        # obs_space, preprocess_obss = utils.get_panda_preprocessor(self.envs[0])

        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        self.obs_space = obs_space
        self.preprocess_obss = preprocess_obss
        # print(f"!!!!!!!!!!!!!!!!!!!!!! function to preproces env:{self.preprocess_obss}")

        self.txt_logger.info("Observations preprocessor loaded")

        # Load model
        self.acmodel = ACModel(obs_space=obs_space, action_space=self.envs[0].action_space, use_memory=use_memory,
                               use_text=use_text)
        if "model_state" in status:
            self.acmodel.load_state_dict(self.status["model_state"])

        self.acmodel.to(device)
        # self.txt_logger.info("Model loaded\n")
        # self.txt_logger.info("{}\n".format(self.acmodel))

        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        print(f"num_frames_per_proc,recurrence:{self.num_frames_per_proc},{self.recurrence}")
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        # print(f"num frames:{self.num_frames_per_proc}")
        # print(f"procs:{self.procs}")

        self.num_frames = self.num_frames_per_proc * self.procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.procs)

        self.obs = self.env.reset()
        # self.obs = self.env.envs[0].render()
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(
                shape[1], self.acmodel.memory_size, device=self.device
            )
            self.memories = torch.zeros(
                *shape, self.acmodel.memory_size, device=self.device
            )
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.procs
        self.log_reshaped_return = [0] * self.procs
        self.log_num_frames = [0] * self.procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            # preprocessed_obs = self.preprocess_obss(self.envs, device=self.device)

            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            # TODO: fixes for minigrid update
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # if any(done):
            #     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!done:{done}")
            # Update experiences values

            for observation in self.obs:
                self.update_states_counter(observation_str=str(observation))

            # Minigrid <--> Panda Switch
            self.obss[i] = self.obs
            # self.obss[i] = {"image": self.env.envs[0].render()}
            # self.obs

            # print(f"obss[{i}]:{self.obss[i].shape}")
            # self.obss[i] = self.envs

            self.obs = obs

            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        # Minigrid <--> Panda Switch
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        # preprocessed_obs = self.preprocess_obss(self.envs, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.gamma * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.gamma * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()

        # exps.obs = [self.obss[i]
        #             for i in range(self.num_frames_per_proc)]
        # for i in range(self.num_frames_per_proc):
        #     for j in range(self.procs):
        #         print(f"obss:{self.obss[i][j]['image'].shape}")
        #     if i == 3:
        #         asdasdadasd
        # for i in range(self.num_frames_per_proc):
        #     print(f"obss:{self.obss[i]['image'].shape}")

        # exps.obs = [self.obss[i]
        #    #         for j in range(self.procs)
                    # for i in range(self.num_frames_per_proc)]

        exps.obs = [self.obss[i][j]
                    for j in range(self.procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        # Minigrid <--> Panda Switch
        # print(f"expr obs shape:{type(exps.obs)}")
        # print(f"expr ovs:{type(exps.obs[0])}")
        # print(f"expr obs shape:{exps.obs[0].shape}")

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # exps.obs = utils.preprocess_panda(exps.obs, device=device)
        # print(f"len of expr.obs:{len(exps.obs)}")

        # Log some values
        keep = max(self.log_done_counter, self.procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.procs:]
        self.log_num_frames = self.log_num_frames[-self.procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass

    def learn(self, log_interval=ml_consts.LOG_INTERVAL, save_interval=ml_consts.SAVE_INTERVAL):
        csv_file, csv_logger = utils.get_csv_logger(self.model_dir)
        tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        num_frames = self.status["num_frames"]
        update = self.status["update"]
        start_time = time.time()

        while num_frames < self.episodes:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = self.collect_experiences()
            logs2 = self.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                # self.txt_logger.info(
                #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                #     .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if self.status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status
            if save_interval > 0 and update % save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                          "model_state": self.acmodel.state_dict(), "optimizer_state": self.optimizer.state_dict()}
                if hasattr(self.preprocess_obss, "vocab"):
                    status["vocab"] = self.preprocess_obss.vocab.vocab
                utils.save_status(status, self.model_dir)
                self.txt_logger.info("Status saved")

        self._save_states_seen_to_file()
        # print(f"number of unique states:{self.get_number_of_unique_states()}")

    def get_probabilities(self, state: numpy.array):

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(state, self.memory * self.mask.unsqueeze(1))
            else:
                dist, _ = self.acmodel(state)

        return dist

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        dist = self.get_probabilities(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def _save_states_seen_to_file(self):
        conf = {
            'states_counter': self.states_counter
        }
        with open(self._states_seen_file, "wb") as f:
            pickle.dump(obj=conf, file=f)

    def _load_states_seen_from_file(self):
        if os.path.exists(self._states_seen_file):
            print(self._states_seen_file)
            with open(self._states_seen_file, "rb") as f:
                conf = pickle.load(f)
            self.states_counter = conf['states_counter']
