# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test RPCLauncher on x86_64 machines"""
import pytest
import os
import numpy as np

import tvm
from tvm.contrib import Launcher
from tvm.contrib.launcher.rpc_launcher import LauncherNotFound
from tvm import rpc
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor

LOCALHOST_ID = "localhost"


def test_not_supported_launcher():
    """Test not supported launcher name."""
    with pytest.raises(LauncherNotFound):
        Launcher(remote_id=LOCALHOST_ID, launcher_type="x86_")


def test_x86_launcher_multiple_names():
    """Test multiple supported names for x86_64 launcher."""
    Launcher(remote_id=LOCALHOST_ID, launcher_type="x86")
    Launcher(remote_id=LOCALHOST_ID, launcher_type="x64")
    Launcher(remote_id=LOCALHOST_ID, launcher_type="x86_64")


def test_copy_binaries():
    """Test binary files in remote location"""
    launcher = Launcher(remote_id=LOCALHOST_ID, launcher_type="x86")
    launcher._copy_binaries()
    workspace_content = os.listdir(path=launcher._workspace)
    for item in launcher.RPC_FILES:
        assert item in workspace_content


def test_workload():
    """Test a relay workload"""
    data = tvm.relay.var("data", tvm.relay.TensorType((1, 3, 64, 64), "float32"))
    weight = tvm.relay.var("weight", tvm.relay.TensorType((8, 3, 5, 5), "float32"))
    y = tvm.relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = tvm.relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = tvm.relay.transform.InferType()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(
            mod,
            target="llvm",
            executor=Executor("graph"),
        )
    tmp_dir = tvm.contrib.utils.tempdir()
    lib_fname = tmp_dir / "model.so"
    lib.export_library(lib_fname)

    weight_data = np.ones((8, 3, 5, 5)).astype("float32")
    input_data = np.ones((1, 3, 64, 64)).astype("float32")
    inputs = {"data": input_data, "weight": weight_data}

    port = 9091
    rpc_info = {"rpc_server_port": port}
    with Launcher(remote_id=LOCALHOST_ID, launcher_type="x86", rpc_info=rpc_info) as launcher:
        remote = rpc.connect("localhost", port)
        remote.upload(lib_fname)
        rlib = remote.load_module("model.so")
        dev = remote.cpu(0)
        module = graph_executor.GraphModule(rlib["default"](dev))
        module.set_input(**inputs)
        module.run()
        out = module.get_output(0).numpy()


if __name__ == "__main__":
    tvm.testing.main()
