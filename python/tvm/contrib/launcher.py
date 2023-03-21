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
"""Host-driven launcher for remote devices."""

import abc
import enum
import datetime
import random
import string
from pathlib import Path

from .._ffi import libinfo


def _get_test_directory_name() -> str:
    """Generate a time-stamped name for use as a test directory name."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    random_str = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    return f"{date_str}-{random_str}"


class Launcher(metaclass=abc.ABCMeta):
    """An abstract class to interact with a remote device
    over RPC.

    There are multiple private methods that each launcher need to implement:
    - _prepare_rpc_script
    - _copy_to_remote
    - _start_server
    - _stop_server
    - _create_remote_directory

    And there are few private methods that could be implemented depends on the
    specific remote target.
    - _pre_server_start
    - _post_server_start
    - _pre_server_stop
    - _post_server_stop
    - _copy_extras

    Parameters
    ----------
    rpc_info : dict
        Description of the RPC setup. Recognized keys:
            "rpc_tracker_host" : str    name of the host running the tracker (default "0.0.0.0")
            "rpc_tracker_port" : int    port number of the tracker (default: 9190)
            "rpc_server_port"  : int    port number for the RPC server to use (default 7070)
            "workspace_base"   : str    name of base test directory (default ".")
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

    def __init__(self, rpc_info: dict, workspace: Path = None, remote_id: str = None):
        self._rpc_info = {
            "rpc_server_port": 7070,
            "workspace_base": ".",
        }
        self._rpc_info.update(rpc_info)
        self._workspace = self._create_workspace(workspace)
        self._remote_id = remote_id

    def __enter__():
        self.start_server()

    def __exit__(*exc_info):
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
        self._prepare_rpc_script()
        for item in self.RPC_FILES:
            self._copy_to_remote(self._lib_dir / item, self._workspace / item)
        self._copy_extras()

    @abc.abstractmethod
    def _prepare_rpc_script(self):
        """Prepare RPC script and copy to remote."""

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
    AARCH64_CUDA = "aarch64_cuda"

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
