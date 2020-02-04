import logging
import numpy as np
import grpc
import gym
from abc import ABCMeta, abstractmethod

from multi_agent_gym.protos.proto_env_message_pb2 import SubEnvInfo
from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = utils.logger.create_standard_logger(__name__, logging.DEBUG)


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

    def reset(self):
        sub_env_info = proto_env_message_pb2.SubEnvInfo(sub_env_id=self.agent_id)

        initial_observation_proto: proto_env_message_pb2.InitialObservation = self.stub.GetInitialObservation(sub_env_info)
        initial_observation = utils.numproto.proto_to_ndarray(initial_observation_proto.observation)

        assert self.observation_space.contains(initial_observation)

        logger.info("Initial observation: {}".format(initial_observation))

    def step(self, action):
        pass

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        super().close() # Call in the end


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = proto_env_message_pb2_grpc.TurnBasedServerStub(channel)
        logger.debug("-------------- GetObservation --------------")

        sub_env_info: proto_env_message_pb2.SubEnvInfo = proto_env_message_pb2.SubEnvInfo(sub_env_id="Env1")
        observation: proto_env_message_pb2.NDArray = utils.numproto.ndarray_to_proto(np.array([[1, 2, 3], [3, 4, 5]]))

        request: proto_env_message_pb2.RequestInfo = proto_env_message_pb2.RequestInfo(sub_env_info=sub_env_info,
                                                                                       observation=observation)

        logger.debug(utils.numproto.proto_to_ndarray(stub.GetObservation(request)))


if __name__ == '__main__':
    env = AgentEnv("localhost:50051")
    env.reset()
