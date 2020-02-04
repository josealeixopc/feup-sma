import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import grpc

from multi_agent_gym import utils

from multi_agent_gym.protos import proto_env_message_pb2
from multi_agent_gym.protos import proto_env_message_pb2_grpc

logger = utils.logger.create_standard_logger(__name__, logging.DEBUG)


class MultiAgentServicer(proto_env_message_pb2_grpc.TurnBasedServerServicer):
    def GetObservation(self, request, context):
        request: proto_env_message_pb2.RequestInfo = request
        response: proto_env_message_pb2.NDArray = request.observation

        logger.debug("Request: {}.".format(request))
        logger.debug("Context: {}.".format(context))
        logger.debug("Response: {}".format(response))

        return response


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
