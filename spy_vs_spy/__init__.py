import logging
import sys

logging.basicConfig(stream=sys.stdout,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.DEBUG)

# Don't log messages from the multi_agent_gym package
# logging.getLogger('multi_agent_gym').propagate = False

