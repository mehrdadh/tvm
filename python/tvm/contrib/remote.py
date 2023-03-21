from paramiko import SSHClient
from pathlib import Path
import typing


class RemoteSecureShell:
    def __init__(self, username, host):
        self._ssh = SSHClient()
        self._ssh.load_system_host_keys()
        self._ssh.connect(hostname=host, username=username)
        self._ftp_client = self._ssh.open_sftp()

    def upload(self, src: Path, dst: Path):
        self._ftp_client.put(str(src), str(dst))

    def exec_command(self, cmd: typing.List[str], workspace: Path = None):
        if workspace:
            cmd_str = f"cd {str(workspace)} && "
        else:
            cmd_str = ""

        for item in cmd:
            cmd_str += f"{item} "
        self._ssh.exec_command(cmd_str)

    def __exit__(self):
        self._ftp_client.close()
