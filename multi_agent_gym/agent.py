import logging
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
        logger.debug("-------------- GetFeature --------------")
        logger.debug(stub.GetObservation(proto_env_message_pb2.RequestInfo()))


if __name__ == '__main__':
    run()
