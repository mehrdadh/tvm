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
"""Host-driven RPC launcher for remote devices."""

import abc
import enum
import datetime
import random
import string
import tempfile
from pathlib import Path

from ..._ffi import libinfo


def _get_test_directory_name() -> str:
    """Generate a time-stamped name for use as a test directory name."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    random_str = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    return f"{date_str}-{random_str}"


class RPCLauncher(metaclass=abc.ABCMeta):
    """An abstract class to interact with a remote device
    over RPC.

    There are multiple private methods that each launcher need to implement:
    - _copy_to_remote
    - _start_server
    - _stop_server
    - _create_remote_directory

    And there are few private methods that could be implemented depends on the
    specific remote target.
    - _prepare_rpc_script
    - _pre_server_start
    - _post_server_start
    - _pre_server_stop
    - _post_server_stop
    - _copy_extras

    Parameters
    ----------
    rpc_info : dict
        Description of the RPC setup. Recognized keys:
            "rpc_server_port"  : int    port number for the RPC server to use (default 7070).
            "workspace_base"   : str    name of base test directory.
            "rpc_tracker_host" : str    name of the host running the tracker. If
                not set, RPCLauncher ignores connecting to tracker.
            "rpc_tracker_port" : int    port number of the tracker. If
                not set, RPCLauncher ignores connecting to tracker.
    remote_id : str
        Remote devicde unique id.
    workspace : str or patlib.Path
        The server's remote working directory. If this directory does not
        exist, it will be created. If it does exist, the servermust have
        write permissions to it.
        If this parameter is None, a subdirectory in the `workspace_base`
        directory will be created, otherwise the `workspace_base` is not
        used.
    """

    RPC_SCRIPT_FILE_NAME = "rpc_server.sh"
    RPC_FILES = [
        "libtvm_runtime.so",
        "tvm_rpc",
    ]
    RPC_PID_FILE = "rpc_pid.txt"

    def __init__(self, rpc_info: dict, remote_id: str, workspace: Path = None):
        assert "workspace_base" in rpc_info
        self._rpc_info = {
            "rpc_server_port": 7070,
        }
        self._rpc_info.update(rpc_info)
        self._workspace = self._create_workspace(workspace)
        self._remote_id = remote_id

    def __enter__(self):
        self.start_server()

    def __exit__(self, *exc_info):
        self.stop_server()

    def start_server(self):
        self._copy_binaries()
        self._pre_server_start()
        self._start_server()
        self._post_server_start()

    def stop_server(self):
        self._pre_server_stop()
        self._stop_server()
        self._post_server_stop()

    def _copy_binaries(self):
        """Prepare remote files and copy to remote."""
        self._write_rpc_script_and_upload()
        for item in self.RPC_FILES:
            self._copy_to_remote(self._lib_dir / item, self._workspace / item)
        self._copy_extras()

    def _write_rpc_script_and_upload(self):
        """Write RPC script file, copy to remote and make it executable."""
        rpc_script_info = self._prepare_rpc_script()
        if not rpc_script_info:
            rpc_script_info = {}

        if "BEFORE_SERVER_START" not in rpc_script_info:
            rpc_script_info["BEFORE_SERVER_START"] = ""
        if "AFTER_SERVER_START" not in rpc_script_info:
            rpc_script_info["AFTER_SERVER_START"] = ""

        rpc_tracker_mode = False
        if "rpc_tracker_host" in self._rpc_info and "rpc_tracker_port" in self._rpc_info:
            rpc_tracker_mode = True
        with open(self._lib_dir / f"{self.RPC_SCRIPT_FILE_NAME}.template", "r") as src_f:
            with tempfile.TemporaryDirectory() as temp_dir:
                bash_script_path = Path(temp_dir) / self.RPC_SCRIPT_FILE_NAME
                with open(bash_script_path, "w") as dest_f:
                    for line in src_f.readlines():
                        if "<RPC_TRACKER_ARG>" in line:
                            if rpc_tracker_mode:
                                rpc_tracker_arg = f'--tracker={str(self._rpc_info["rpc_tracker_host"])}:{str(self._rpc_info["rpc_tracker_port"])}'
                            else:
                                rpc_tracker_arg = ""
                            line = line.replace("<RPC_TRACKER_ARG>", rpc_tracker_arg)

                        if "<DEVICE_KEY_ARG>" in line:
                            if rpc_tracker_mode:
                                device_key_arg = f'--key={self._rpc_info["device_key"]}'
                            else:
                                device_key_arg = ""
                            line = line.replace("<DEVICE_KEY_ARG>", device_key_arg)

                        if "<RPC_SERVER_PORT>" in line:
                            line = line.replace(
                                "<RPC_SERVER_PORT>", str(self._rpc_info["rpc_server_port"])
                            )
                        if "<BEFORE_SERVER_START>" in line:
                            line = rpc_script_info["BEFORE_SERVER_START"]
                        if "<AFTER_SERVER_START>" in line:
                            line = rpc_script_info["AFTER_SERVER_START"]
                        dest_f.write(line)

                self._copy_to_remote(bash_script_path, self._workspace / bash_script_path.name)
                self._make_file_executable(self._workspace / bash_script_path.name)

    @abc.abstractmethod
    def _prepare_rpc_script(self) -> dict:
        """Prepare RPC script template variables.

        Implement this function to add extra steps before and after
        executing tvm_rpc start command on the remote device. This is
        an optional abstract function.

        Returns
        -------
        Optional[dict]:
            A dictionary of command before and after the rpc starts.
            This is optional in case no extra steps is required in a
            remote target. Here are the expected keys in this dictionary:
            - BEFORE_SERVER_START
            - AFTER_SERVER_START
        """
        ...

    @abc.abstractmethod
    def _copy_extras(self):
        """Copy extra files to remote."""
        ...

    @abc.abstractmethod
    def _copy_to_remote(self, local_path: Path, remote_path: Path):
        """Copy a local file to a remote location.

        Parameters
        ----------
        local_path : Path
            Path to the local file.
        remote_path : Path
            Path to the remote file (to be written).
        """
        ...

    @abc.abstractmethod
    def _pre_server_start(self):
        """Placeholder for initialization before starting RPC server."""
        ...

    @abc.abstractmethod
    def _start_server(self):
        """Server start implementation."""

    @abc.abstractmethod
    def _post_server_start(self):
        """Placeholder for after starting RPC server."""
        ...

    @abc.abstractmethod
    def _pre_server_stop(self):
        """Placeholder for initialization before stopping RCP server."""
        ...

    @abc.abstractmethod
    def _stop_server(self):
        """Server stop implementation."""
        ...

    @abc.abstractmethod
    def _post_server_stop(self):
        """Placeholder for post server stop."""
        ...

    @abc.abstractmethod
    def _create_remote_directory(self, remote_path: Path) -> Path:
        """Create a directory in the remote location.

        Parameters
        ----------
        remote_path : Path
            Name of the directory to be created.

        Returns
        -------
        Path :
            Absolute path of the remote workspace.
        """
        ...

    def _create_workspace(self, workspace: Path) -> Path:
        """Create a working directory for the server.

        Parameters
        ----------
        workspace : Path or NoneType
            Name of the directory to create. If None, a new name is constructed
            using workspace_base.

        Returns
        -------
        Path :
            Created workspace.
        """
        if not workspace:
            base_dir = self._rpc_info["workspace_base"]
            workspace = Path(base_dir) / _get_test_directory_name()
        self._create_remote_directory(workspace)
        return workspace


class LauncherType(enum.Enum):
    AARCH64 = "aarch64"
    AARCH64_CUDA = "aarch64_cuda"
    AARCH64_TENSORRT = "aarch64_tensorrt"
    X86_64 = "x86_64"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class LauncherNotFound(Exception):
    """Raise when the launcher cannot be found in launcher libraries."""


def get_launcher_lib_path(launcher: str) -> Path:
    """Find libraries path for specific launcher.

    Parameters
    ----------
    launcher: str
        Launcher type.

    Returns
    -------
    Path :
        Path to launcher library.
    """
    if launcher not in LauncherType.list():
        raise ValueError(f"Launcher {launcher} is not supported.")
    for path in libinfo.find_lib_path():
        launcher_path = Path(path).parent / "launchers" / launcher
        if launcher_path.exists():
            return launcher_path
    raise LauncherNotFound()
