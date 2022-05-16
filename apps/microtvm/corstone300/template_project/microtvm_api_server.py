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

import atexit
import collections.abc
import enum
import fcntl
import logging
import os
import os.path
import pathlib
from pickle import BUILD
import queue
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import datetime
from typing import Union

from tvm.micro.project_api import server
from tvm.micro import get_standalone_crt_dir

_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR / "build"


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


# BOARDS = API_SERVER_DIR / "boards.json"

ETHOSU_PATH = os.getenv("ETHOSU_PATH", "/opt/arm/ethosu")

CMSIS_PATH = os.getenv("CMSIS_PATH", "/opt/arm/ethosu/cmsis")

FVP_PATH = os.getenv("FVP_PATH", "/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4")


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


def subprocess_check_log_output(
    cmd, cwd: Union[str, pathlib.Path], logfile: Union[str, pathlib.Path]
):
    """
    This method runs a process and logs the output to both a log file and stdout
    """
    cwd = str(cwd)
    _LOG.info("Execute (%s): %s", cwd, cmd)
    cmd_base = cmd[0] if isinstance(cmd, (list, tuple)) else cmd.split(" ", 1)[0]
    process_output = subprocess.check_output(
        cmd,
        cwd=cwd,
        encoding="utf-8",
    )

    with open(logfile, "a") as f:
        msg = (
            "\n"
            + "-" * 80
            + f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Execute ({cwd}): {cmd}\n"
            + "-" * 80
        )
        f.write(msg)
        for line in process_output:
            _LOG.debug("%s: %s", cmd_base, line.rstrip("\n"))
            f.write(line)


CACHE_ENTRY_RE = re.compile(r"(?P<name>[^:]+):(?P<type>[^=]+)=(?P<value>.*)")


CMAKE_BOOL_MAP = dict(
    [(k, True) for k in ("1", "ON", "YES", "TRUE", "Y")]
    + [(k, False) for k in ("0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "")]
)


PROJECT_TYPES = []
if IS_TEMPLATE:
    for d in (API_SERVER_DIR / "src").iterdir():
        if d.is_dir():
            PROJECT_TYPES.append(d.name)


PROJECT_OPTIONS = [
    server.ProjectOption(
        "extra_files_tar",
        optional=["generate_project"],
        type="str",
        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
    ),
    server.ProjectOption(
        "project_type",
        choices=tuple(PROJECT_TYPES),
        required=["generate_project"],
        type="str",
        help="Type of project to generate.",
    ),
    server.ProjectOption(
        "verbose",
        optional=["build"],
        type="bool",
        help="Run build with verbose output.",
    ),
    server.ProjectOption(
        "ETHOSU_PATH",
        required=["generate_project"],
        default=ETHOSU_PATH,
        type="str",
        help="Path to ETHOSU",
    ),
    server.ProjectOption(
        "CMSIS_PATH",
        required=["generate_project"],
        default=CMSIS_PATH,
        type="str",
        help="Path to CMSIS.",
    ),
    server.ProjectOption(
        "FVP_PATH",
        required=["generate_project"],
        default=FVP_PATH,
        type="str",
        help="Path to FVP.",
    ),
    server.ProjectOption(
        "compile_definitions",
        optional=["generate_project"],
        type="str",
        help="Extra definitions added project compile.",
    ),
    server.ProjectOption(
        "workspace_bytes",
        optional=["generate_project"],
        type="int",
        help="TVM workpace size in bytes.",
    ),
    server.ProjectOption(
        "custom_params",
        optional=["build"],
        type="str",
        help="Add custom build parameters.",
    ),
    server.ProjectOption(
        "log_output_file",
        required=["build"],
        type="str",
        help="Path to a output log file.",
    ),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="zephyr",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
            project_options=PROJECT_OPTIONS,
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    # CRT_COPY_ITEMS = ("include", "Makefile", "src")
    CRT_COPY_ITEMS = ("include", "src")

    API_SERVER_MAKEFILE_TOKENS = [
        "<ETHOSU_PATH>",
        "<FVP_PATH>",
        "<CMSIS_PATH>",
        "<CFLAGS>",
        "<TVM_ROOT>",
        "<BUILD_DIR>",
    ]

    CRT_LIBS_BY_PROJECT_TYPE = {
        "host_driven": "microtvm_rpc_server microtvm_rpc_common common",
        "aot_demo": "memory microtvm_rpc_common common",
    }

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        project_dir = pathlib.Path(project_dir)
        # Make project directory.
        if not project_dir.exists():
            project_dir.mkdir()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # TODO(add support for multiple MLF files)
        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_tar_path = project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        shutil.copy2(model_library_format_path, project_model_library_format_tar_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = os.path.splitext(project_model_library_format_tar_path)[0]
        with tarfile.TarFile(project_model_library_format_tar_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        # TODO(mehrdadh): make linker file customizable
        shutil.copy(API_SERVER_DIR / "corstone300.ld", project_dir / "corstone300.ld")

        # Populate CRT.
        crt_path = project_dir / "crt"
        crt_path.mkdir()
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Populate Makefile.
        with open(API_SERVER_DIR / "corstone300.mk.template", "r") as make_template_f:
            with open(project_dir / "corstone300.mk", "w") as make_f:
                for line in make_template_f:
                    if any(token in line for token in self.API_SERVER_MAKEFILE_TOKENS):
                        if "<ETHOSU_PATH>" in line:
                            ethosu_path = (
                                options.get("ETHOSU_PATH")
                                if options.get("ETHOSU_PATH")
                                else ETHOSU_PATH
                            )
                            line = line.replace("<ETHOSU_PATH>", f"ETHOSU_PATH={ethosu_path}")
                        elif "<CFLAGS>" in line:
                            cflags = ""
                            for item in options["compile_definitions"]:
                                cflags += f"{item} "
                            line = line.replace("<CFLAGS>", f'CFLAGS="{cflags}"')
                        elif "<TVM_ROOT>" in line:
                            line = line.replace("<TVM_ROOT>", f"TVM_ROOT=/home/mhessar/work/tvm")
                        elif "<CMSIS_PATH>" in line:
                            cmsis_path = (
                                options.get("CMSIS_PATH")
                                if options.get("CMSIS_PATH")
                                else CMSIS_PATH
                            )
                            line = line.replace("<CMSIS_PATH>", f"CMSIS_PATH={cmsis_path}")
                        elif "<BUILD_DIR>" in line:
                            line = line.replace(
                                "<BUILD_DIR>", f'BUILD_DIR := {str(project_dir / "build")}'
                            )
                        elif "<FVP_PATH>" in line:
                            fvp_path = (
                                options.get("FVP_PATH") if options.get("FVP_PATH") else FVP_PATH
                            )
                            line = line.replace("<FVP_PATH>", f"FVP_PATH={fvp_path}")
                    make_f.write(line)
                    # TODO(mehrdad): add custom_params

        # Populate crt-config.h
        # crt_config_dir = project_dir / "crt_config"
        # crt_config_dir.mkdir()
        # shutil.copy2(
        #     API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        # )

        # Populate content of src/
        dst_dir = project_dir / "src"
        if not dst_dir.exists():
            dst_dir.mkdir()
        src_dir = API_SERVER_DIR / "src"

        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        # Populate extra_files
        if options.get("extra_files_tar"):
            with tarfile.open(options["extra_files_tar"], mode="r:*") as tf:
                tf.extractall(project_dir)

        # Add host crt template files
        include_path = project_dir / "include"
        if not include_path.exists():
            include_path.mkdir()

        crt_root = pathlib.Path(get_standalone_crt_dir())
        shutil.copy2(
            crt_root / "template" / "crt_config-template.h",
            include_path / "crt_config.h",
        )

    def build(self, options):
        BUILD_DIR.mkdir()
        shutil.copy(BUILD_DIR / ".." / "corstone300.mk", BUILD_DIR / "Makefile")

        # Build
        make_args = ["make"]
        make_args.append("aot_test_runner")
        check_call(make_args, cwd=BUILD_DIR)

        # Run
        run_log_path = options.get("log_output_file")
        run_args = ["make"]
        run_args.append("run")

        subprocess_check_log_output(run_args, BUILD_DIR, run_log_path)

    def flash(self, options):
        return

    def open_transport(self, options):
        transport = ZephyrQemuTransport(options)

        to_return = transport.open()
        self._transport = transport
        atexit.register(lambda: self.close_transport())
        return to_return

    def close_transport(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def read_transport(self, n, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.read(n, timeout_sec)

    def write_transport(self, data, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.write(data, timeout_sec)


def _set_nonblock(fd):
    flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
    new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
    assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"


class ZephyrQemuMakeResult(enum.Enum):
    QEMU_STARTED = "qemu_started"
    MAKE_FAILED = "make_failed"
    EOF = "eof"


class ZephyrQemuTransport:
    """The user-facing Zephyr QEMU transport class."""

    def __init__(self, options):
        self.options = options
        self.proc = None
        self.pipe_dir = None
        self.read_fd = None
        self.write_fd = None
        self._queue = queue.Queue()

    def open(self):
        self.pipe_dir = pathlib.Path(tempfile.mkdtemp())
        self.pipe = self.pipe_dir / "fifo"
        self.write_pipe = self.pipe_dir / "fifo.in"
        self.read_pipe = self.pipe_dir / "fifo.out"
        os.mkfifo(self.write_pipe)
        os.mkfifo(self.read_pipe)

        env = None
        if self.options.get("gdbserver_port"):
            env = os.environ.copy()
            env["TVM_QEMU_GDBSERVER_PORT"] = self.options["gdbserver_port"]

        self.proc = subprocess.Popen(
            ["make", "run", f"QEMU_PIPE={self.pipe}"],
            cwd=BUILD_DIR,
            env=env,
            stdout=subprocess.PIPE,
        )
        self._wait_for_qemu()

        # NOTE: although each pipe is unidirectional, open both as RDWR to work around a select
        # limitation on linux. Without this, non-blocking I/O can't use timeouts because named
        # FIFO are always considered ready to read when no one has opened them for writing.
        self.read_fd = os.open(self.read_pipe, os.O_RDWR | os.O_NONBLOCK)
        self.write_fd = os.open(self.write_pipe, os.O_RDWR | os.O_NONBLOCK)
        _set_nonblock(self.read_fd)
        _set_nonblock(self.write_fd)

        return server.TransportTimeouts(
            session_start_retry_timeout_sec=2.0,
            session_start_timeout_sec=10.0,
            session_established_timeout_sec=10.0,
        )

    def close(self):
        did_write = False
        if self.write_fd is not None:
            try:
                server.write_with_timeout(
                    self.write_fd, b"\x01x", 1.0
                )  # Use a short timeout since we will kill the process
                did_write = True
            except server.IoTimeoutError:
                pass
            os.close(self.write_fd)
            self.write_fd = None

        if self.proc:
            if not did_write:
                self.proc.terminate()
            try:
                self.proc.wait(5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        if self.read_fd:
            os.close(self.read_fd)
            self.read_fd = None

        if self.pipe_dir is not None:
            shutil.rmtree(self.pipe_dir)
            self.pipe_dir = None

    def read(self, n, timeout_sec):
        return server.read_with_timeout(self.read_fd, n, timeout_sec)

    def write(self, data, timeout_sec):
        to_write = bytearray()
        escape_pos = []
        for i, b in enumerate(data):
            if b == 0x01:
                to_write.append(b)
                escape_pos.append(i)
            to_write.append(b)

        while to_write:
            num_written = server.write_with_timeout(self.write_fd, to_write, timeout_sec)
            to_write = to_write[num_written:]

    def _qemu_check_stdout(self):
        for line in self.proc.stdout:
            line = str(line)
            _LOG.info("%s", line)
            if "[QEMU] CPU" in line:
                self._queue.put(ZephyrQemuMakeResult.QEMU_STARTED)
            else:
                line = re.sub("[^a-zA-Z0-9 \n]", "", line)
                pattern = r"recipe for target (\w*) failed"
                if re.search(pattern, line, re.IGNORECASE):
                    self._queue.put(ZephyrQemuMakeResult.MAKE_FAILED)
        self._queue.put(ZephyrQemuMakeResult.EOF)

    def _wait_for_qemu(self):
        threading.Thread(target=self._qemu_check_stdout, daemon=True).start()
        while True:
            try:
                item = self._queue.get(timeout=120)
            except Exception:
                raise TimeoutError("QEMU setup timeout.")

            if item == ZephyrQemuMakeResult.QEMU_STARTED:
                break

            if item in [ZephyrQemuMakeResult.MAKE_FAILED, ZephyrQemuMakeResult.EOF]:
                raise RuntimeError("QEMU setup failed.")

            raise ValueError(f"{item} not expected.")


if __name__ == "__main__":
    server.main(Handler())
