# protobuf

This folder contains the `.proto` file(s) declarating how messages between the modules of this project are structured.

The Makefile creates the classes for Python from the Protobuf file.

The `protoc-gen-mypy.py` script comes from [mypy-protobuf](https://github.com/dropbox/mypy-protobuf) and creates a `.pyi` file from the Protobuf file, which is useful for IDE autocompletion. It is called as a plugin in the `protoc` invocation.

**Warning**: you'll need to make `protoc-gen-mypy.py` executable using `chmod +x protoc-gen-mypy.py`. Also, you'll need to `pip install mypy`.

## Running

To create/update the classes generated from the Protobuf file:

`make`

To delete the generated classes:

`make clean`