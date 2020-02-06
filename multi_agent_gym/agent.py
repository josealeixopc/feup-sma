import logging
import numpy as np
import grpc
import gym
import json

import typing

from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = logging.getLogger(__name__)


class AgentEnv(gym.Env):

    def __init__(self, agent_id: str, server: str):
        self.agent_id = agent_id
        self.server = server
        self.channel = grpc.insecure_channel(self.server)
        self.stub = proto_env_message_pb2_grpc.TurnBasedServerStub(self.channel)

        self.observation_space: gym.Space
        self.action_space: gym.Space

        self._init_observation_space()
        self._init_action_space()

    def _init_observation_space(self) -> None:
        raise NotImplementedError

    def _init_action_space(self) -> None:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        sub_env_info_proto = proto_env_message_pb2.SubEnvInfo(sub_env_id=self.agent_id)

        initial_observation_proto: proto_env_message_pb2.InitialObservation = \
            self.stub.GetInitialObservation(sub_env_info_proto)

        initial_observation = utils.numproto.proto_to_ndarray(initial_observation_proto.observation)

        logger.info("Initial observation: {}".format(initial_observation))

        assert self.observation_space.contains(initial_observation)

        return initial_observation

    def step(self, action: np.ndarray) -> typing.Tuple[np.ndarray, float, bool, dict]:
        sub_env_info_proto = proto_env_message_pb2.SubEnvInfo(sub_env_id=self.agent_id)
        action_proto = utils.numproto.ndarray_to_proto(action)

        action_info_proto = proto_env_message_pb2.ActionInfo(sub_env_info=sub_env_info_proto, action=action_proto)

        observation_proto: proto_env_message_pb2.Observation = \
            self.stub.GetObservation(action_info_proto)

        obs = utils.numproto.proto_to_ndarray(observation_proto.observation)
        reward = observation_proto.reward
        done = observation_proto.done
        info = json.loads(observation_proto.info)

        logger.info("Observation: {}. Reward: {}. Done: {}. Info: {}.".format(obs, reward, done, info))

        assert self.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        return obs, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        super().close()  # Call in the end


if __name__ == '__main__':
    env = AgentEnv("localhost:50051")
    env.reset()
