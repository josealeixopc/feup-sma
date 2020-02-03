import logging
from concurrent.futures import ThreadPoolExecutor

import grpc

from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = utils.logger.create_standard_logger(__name__, logging.DEBUG)


class MultiAgentServicer(proto_env_message_pb2_grpc.TurnBasedServerServicer):
    def GetObservation(self, request, context):
        logger.debug("Request: {}.".format(request))
        logger.debug("Context: {}.".format(context))


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    proto_env_message_pb2_grpc.add_TurnBasedServerServicer_to_server(
        MultiAgentServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
