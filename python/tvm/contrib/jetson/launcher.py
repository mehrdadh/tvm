from pathlib import Path
import os
import stat
import tempfile
import time

from tvm.contrib import Launcher, get_launcher_lib_path
from tvm.contrib.remote import RemoteSecureShell


class JetsonLauncher(Launcher):
    def __init__(
        self,
        remote_hostname: str,
        rpc_info: dict,
        workspace: Path = None,
        remote_username: str = None,
    ):
        self._lib_dir: Path = get_launcher_lib_path("aarch64_cuda")
        self._remote_username = remote_username
        self._remote_hostname = remote_hostname
        if not self._remote_username:
            self._remote_username = os.getlogin()

        rpc_info["device_key"] = "jetson" + "." + self._remote_hostname
        self._remote_shell = RemoteSecureShell(
            username=self._remote_username, host=self._remote_hostname
        )
        super(JetsonLauncher, self).__init__(rpc_info, workspace, self._remote_hostname)

    def _prepare_rpc_script(self):
        """Abstract method implementation. See description in Launcher."""
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
                        if "<BEFORE_SERVER_START>" in line or "AFTER_SERVER_START" in line:
                            line = ""
                        dest_f.write(line)

                self._copy_to_remote(bash_script_path, self._workspace / bash_script_path.name)
                self._make_file_executable(self._workspace / bash_script_path.name)

    def _copy_to_remote(self, local_path: Path, remote_path: Path):
        """Abstract method implementation. See description in Launcher."""
        self._remote_shell.upload(local_path, remote_path)

    def _start_server(self):
        """Abstract method implementation. See description in Launcher."""
        self._remote_shell.exec_command([f"./{self.RPC_SCRIPT_FILE_NAME}"], self._workspace)
        # Make sure RPC server is up
        time.sleep(5)

    def _stop_server(self):
        """Abstract method implementation. See description in Launcher."""
        self._remote_shell.exec_command(
            ["kill", "-9", f"$(cat {str(self._workspace)}/{self.RPC_PID_FILE})"]
        )
        self._remote_shell.exec_command(
            ["kill", "-s", "sigint", f"$(cat {str(self._workspace)}/${self.RPC_PID_FILE})"]
        )

    def _create_remote_directory(self, remote_path: Path) -> Path:
        """Abstract method implementation. See description in Launcher."""
        self._remote_shell.exec_command(["mkdir", str(remote_path)])

    def _copy_extras(self):
        """Abstract method implementation. See description in Launcher."""
        pass

    def _post_server_start(self):
        """Abstract method implementation. See description in Launcher."""
        pass

    def _post_server_stop(self):
        """Abstract method implementation. See description in Launcher."""
        self._remote_shell.exec_command(["rm", "-rf", str(self._workspace)])
        pass

    def _pre_server_start(self):
        """Abstract method implementation. See description in Launcher."""
        for item in self.RPC_FILES:
            self._make_file_executable(self._workspace / item)

    def _pre_server_stop(self):
        """Abstract method implementation. See description in Launcher."""
        pass

    def _make_file_executable(self, file_path: Path):
        self._remote_shell.exec_command(["chmod", "+x", str(file_path)])
