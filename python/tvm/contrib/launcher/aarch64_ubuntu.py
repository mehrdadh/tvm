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
"""AArch64 Ubuntu Launcher class."""

from pathlib import Path
import os
import stat
import tempfile
import time

from tvm.contrib.launcher.rpc_launcher import RPCLauncher, get_launcher_lib_path, LauncherNotFound
from tvm.contrib.launcher.remote import RemoteSecureShell


class AArch64UbuntuLauncher(RPCLauncher):
    """Launcher for AArch64 CPUs running Ubuntu.

    Parameters
    ----------
    remote_id : str
        See description in RPCLauncher.
    launcher_type : str
        Specifies the launcher type. Current supported launchers are:
        [aarch64, aarch64_cuda, aarch64_tensorrt].
    rpc_info : Optional[dict]
        See description in RPCLauncher.
    remote_username: Optional[str]
        Username for remote device. It is used for file transfers.
    workpsace: Optional[str]
        See description in RPCLauncher.
    """

    def __init__(
        self,
        remote_id: str,
        launcher_type: str,
        rpc_info: dict = None,
        **kwargs,
    ):
        if launcher_type not in ["aarch64", "aarch64_cuda", "aarch64_tensorrt"]:
            raise LauncherNotFound()

        self._lib_dir: Path = get_launcher_lib_path(launcher_type)
        self._remote_username = kwargs.get("remote_username")
        self._remote_hostname = remote_id
        if not self._remote_username:
            self._remote_username = os.getlogin()

        if not rpc_info:
            rpc_info = dict()
        rpc_info["device_key"] = "jetson" + "." + self._remote_hostname
        if "workspace_base" not in rpc_info:
            rpc_info["workspace_base"] = "/tmp/tvm_aarch64_ubuntu_launcher"

        self._remote_shell = RemoteSecureShell(
            username=self._remote_username, host=self._remote_hostname
        )
        workspace = kwargs.get("workpsace")
        super(AArch64UbuntuLauncher, self).__init__(
            rpc_info=rpc_info, remote_id=self._remote_hostname, workspace=workspace
        )

    def _copy_to_remote(self, local_path: Path, remote_path: Path):
        """Abstract method implementation. See description in RPCLauncher."""
        self._remote_shell.upload(local_path, remote_path)

    def _start_server(self):
        """Abstract method implementation. See description in RPCLauncher."""
        self._remote_shell.exec_command([f"./{self.RPC_SCRIPT_FILE_NAME}"], self._workspace)
        # Make sure RPC server is up
        time.sleep(5)

    def _stop_server(self):
        """Abstract method implementation. See description in RPCLauncher."""
        self._remote_shell.exec_command(
            ["kill", "-9", f"$(cat {str(self._workspace)}/{self.RPC_PID_FILE})"]
        )
        self._remote_shell.exec_command(
            ["kill", "-s", "sigint", f"$(cat {str(self._workspace)}/${self.RPC_PID_FILE})"]
        )
        # TODO: This would kill child processes as well, but shows error to user since the RPC is
        # terminated before RPC session closes.
        # self._remote_shell.exec_command(
        #     ["kill", "--", f"-$(ps -o pgid= $(cat {str(self._workspace)}/{self.RPC_PID_FILE}) | grep -o '[0-9]*')"]
        # )

    def _create_remote_directory(self, remote_path: Path) -> Path:
        """Abstract method implementation. See description in RPCLauncher."""
        self._remote_shell.exec_command(["mkdir", "-p", str(remote_path)])

    def _copy_extras(self):
        """Abstract method implementation. See description in RPCLauncher."""
        pass

    def _post_server_start(self):
        """Abstract method implementation. See description in RPCLauncher."""
        pass

    def _post_server_stop(self):
        """Abstract method implementation. See description in RPCLauncher."""
        self._remote_shell.exec_command(["rm", "-rf", str(self._workspace)])

    def _pre_server_start(self):
        """Abstract method implementation. See description in RPCLauncher."""
        for item in self.RPC_FILES:
            self._make_file_executable(self._workspace / item)

    def _pre_server_stop(self):
        """Abstract method implementation. See description in RPCLauncher."""
        pass

    def _make_file_executable(self, file_path: Path):
        self._remote_shell.exec_command(["chmod", "+x", str(file_path)])
