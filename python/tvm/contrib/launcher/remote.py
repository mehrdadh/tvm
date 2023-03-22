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
"""Remote shell and file transfer."""

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
