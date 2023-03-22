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
"""Launcher API."""

from pathlib import Path

from .aarch64_ubuntu import AArch64UbuntuLauncher
from .rpc_launcher import LauncherNotFound
from .x86_64 import X86_64Launcher


def Launcher(
    remote_id: str,
    launcher_type: str,
    rpc_info: dict = None,
    **kwargs,
):
    """Create a Launcher.

    Parameters
    ----------
    remote_id : str
        Remote devicde unique id.
    launcher_type: str
        Specifies the launcher type. You can find all launcher types
        by finding LauncherType class in `rpc_launcher.py`.
    rpc_info : Optional[dict]
        Description of the RPC setup. Recognized keys:
            "rpc_server_port"  : int
                port number for the RPC server to use (default 7070).
            "workspace_base"   : str
                name of base test directory (default is set differently in each
                Launcher type).
            "rpc_tracker_host" : str
                name of the host running the tracker. If not set,
                RPCLauncher ignores connecting to tracker.
            "rpc_tracker_port" : int
                port number of the tracker. If not set,
                RPCLauncher ignores connecting to tracker.
    """
    if launcher_type in ["aarch64", "aarch64_cuda", "aarch64_tensorrt"]:
        return AArch64UbuntuLauncher(
            remote_id=remote_id,
            launcher_type=launcher_type,
            rpc_info=rpc_info,
            **kwargs,
        )
    elif launcher_type in ["x86_64", "x86", "x64"]:
        return X86_64Launcher(
            remote_id=remote_id, launcher_type="x86_64", rpc_info=rpc_info, **kwargs
        )
    else:
        raise LauncherNotFound()
