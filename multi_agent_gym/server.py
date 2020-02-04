import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import grpc

from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = utils.logger.create_standard_logger(__name__, logging.DEBUG)


class MultiAgentServicer(proto_env_message_pb2_grpc.TurnBasedServerServicer):

    def GetInitialObservation(self,
                              request: proto_env_message_pb2.SubEnvInfo,
                              context) -> proto_env_message_pb2.InitialObservation:
        logger.debug("Got the following request: {}".format(request))

        initial_observation_arr = np.array([[1, 1, 1], [2, 2, 2]])
        initial_observation_arr_proto = utils.numproto.ndarray_to_proto(initial_observation_arr)

        initial_observation = proto_env_message_pb2.InitialObservation(observation=initial_observation_arr_proto)
        return initial_observation

    def GetObservation(self,
                       request: proto_env_message_pb2.ActionInfo,
                       context) -> proto_env_message_pb2.Observation:
        return super().GetObservation(request, context)


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    proto_env_message_pb2_grpc.add_TurnBasedServerServicer_to_server(
        MultiAgentServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logger.info("hello")
    serve()
