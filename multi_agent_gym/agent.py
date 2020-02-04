import logging
import numpy as np
import grpc

from multi_agent_gym.protos.proto_env_message_pb2 import SubEnvInfo
from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = utils.logger.create_standard_logger(__name__, logging.DEBUG)


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
    run()
