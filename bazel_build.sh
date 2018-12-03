#!/usr/bin/env bash
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" && \
bazel-bin/tensorflow/tools/pip_package/build_pip_package /users/jinlianw/tensorflow-1.12.git//
