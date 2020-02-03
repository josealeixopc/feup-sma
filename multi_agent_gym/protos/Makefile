#
# Makefile to call protocol buffer compiler for Java/Python
#

CC          :=python -m grpc_tools.protoc
SRC         :=proto_env_message.proto

PYTHON_NAME :=proto_env_message_pb2.py
PYTHON_STUB_NAME :=proto_env_message_pb2.pyi
PYTHON_GRPC_NAME :=proto_env_message_pb2_grpc.py

PYTHON_RELATIVE_DEST_DIR := .

PYTHON_OUT		:= $(PYTHON_RELATIVE_DEST_DIR)/$(PYTHON_NAME)
PYTHON_STUB_OUT := $(PYTHON_RELATIVE_DEST_DIR)/$(PYTHON_STUB_NAME)
PYTHON_GRPC_OUT := $(PYTHON_RELATIVE_DEST_DIR)/$(PYTHON_GRPC_NAME)

all: $(PYTHON_OUT) $(PYTHON_STUB_OUT) $(PYTHON_GRPC_OUT)

$(PYTHON_OUT) $(PYTHON_STUB_OUT) $(PYTHON_GRPC_OUT): $(SRC)
	$(CC) $(SRC) -I=. --python_out=$(PYTHON_RELATIVE_DEST_DIR) --grpc_python_out=$(PYTHON_RELATIVE_DEST_DIR) --plugin=protoc-gen-mypy=./protoc-gen-mypy.py --mypy_out=$(PYTHON_RELATIVE_DEST_DIR)
# For some reason, Protobuf does not generate the import statement correctly for Python 3
# So we work around by adding the following line
# From here: https://github.com/protocolbuffers/protobuf/issues/1491#issuecomment-552995275
	sed -i -r 's/import (.+_pb2.*)/from . import \1/g' *_pb2*.py

clean:
	$(RM) $(PYTHON_OUT)
	$(RM) $(PYTHON_STUB_OUT)
	$(RM) $(PYTHON_GRPC_OUT)
