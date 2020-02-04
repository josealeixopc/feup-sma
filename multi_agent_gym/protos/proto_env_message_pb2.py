# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto_env_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto_env_message.proto',
  package='protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x17proto_env_message.proto\x12\x06protos\"]\n\x0bRequestInfo\x12(\n\x0csub_env_info\x18\x01 \x01(\x0b\x32\x12.protos.SubEnvInfo\x12$\n\x0bobservation\x18\x02 \x01(\x0b\x32\x0f.protos.NDArray\"\x1a\n\x07NDArray\x12\x0f\n\x07ndarray\x18\x01 \x01(\x0c\" \n\nSubEnvInfo\x12\x12\n\nsub_env_id\x18\x01 \x01(\t2K\n\x0fTurnBasedServer\x12\x38\n\x0eGetObservation\x12\x13.protos.RequestInfo\x1a\x0f.protos.NDArray\"\x00\x62\x06proto3')
)




_REQUESTINFO = _descriptor.Descriptor(
  name='RequestInfo',
  full_name='protos.RequestInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sub_env_info', full_name='protos.RequestInfo.sub_env_info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='observation', full_name='protos.RequestInfo.observation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=35,
  serialized_end=128,
)


_NDARRAY = _descriptor.Descriptor(
  name='NDArray',
  full_name='protos.NDArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ndarray', full_name='protos.NDArray.ndarray', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=130,
  serialized_end=156,
)


_SUBENVINFO = _descriptor.Descriptor(
  name='SubEnvInfo',
  full_name='protos.SubEnvInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sub_env_id', full_name='protos.SubEnvInfo.sub_env_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=158,
  serialized_end=190,
)

_REQUESTINFO.fields_by_name['sub_env_info'].message_type = _SUBENVINFO
_REQUESTINFO.fields_by_name['observation'].message_type = _NDARRAY
DESCRIPTOR.message_types_by_name['RequestInfo'] = _REQUESTINFO
DESCRIPTOR.message_types_by_name['NDArray'] = _NDARRAY
DESCRIPTOR.message_types_by_name['SubEnvInfo'] = _SUBENVINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RequestInfo = _reflection.GeneratedProtocolMessageType('RequestInfo', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTINFO,
  '__module__' : 'proto_env_message_pb2'
  # @@protoc_insertion_point(class_scope:protos.RequestInfo)
  })
_sym_db.RegisterMessage(RequestInfo)

NDArray = _reflection.GeneratedProtocolMessageType('NDArray', (_message.Message,), {
  'DESCRIPTOR' : _NDARRAY,
  '__module__' : 'proto_env_message_pb2'
  # @@protoc_insertion_point(class_scope:protos.NDArray)
  })
_sym_db.RegisterMessage(NDArray)

SubEnvInfo = _reflection.GeneratedProtocolMessageType('SubEnvInfo', (_message.Message,), {
  'DESCRIPTOR' : _SUBENVINFO,
  '__module__' : 'proto_env_message_pb2'
  # @@protoc_insertion_point(class_scope:protos.SubEnvInfo)
  })
_sym_db.RegisterMessage(SubEnvInfo)



_TURNBASEDSERVER = _descriptor.ServiceDescriptor(
  name='TurnBasedServer',
  full_name='protos.TurnBasedServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=192,
  serialized_end=267,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetObservation',
    full_name='protos.TurnBasedServer.GetObservation',
    index=0,
    containing_service=None,
    input_type=_REQUESTINFO,
    output_type=_NDARRAY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_TURNBASEDSERVER)

DESCRIPTOR.services_by_name['TurnBasedServer'] = _TURNBASEDSERVER

# @@protoc_insertion_point(module_scope)
