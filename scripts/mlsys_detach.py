#!/usr/bin/env python3
"""Run a command in a NEW SESSION, detached from the caller's process group.

WHY THIS EXISTS, with the failure it was written for.

`nohup cmd & disown` is not enough to survive the thing that actually kills these
processes. On 2026-10-09T22:38Z the armed watcher was started that way from an
agent session, and at 2026-10-10T00:37:04Z it received a SIGTERM. The copilot
runtime process it had been launched under restarted at 00:37:15Z, and the
watcher -- plus the separately-launched launch monitor, which stopped heartbeating
at 00:36:31Z -- both died within 35 seconds of each other, mid-run, with no
in-loop reason logged.

What was lost with them: the watcher's TERM trap terminated a healthy 8xA100
instance and pulled nothing, so that run's gate_calibration.csv (676s, $4.19) and
its three completed engine_cap_smoke rows went with the pod. (The trap has since
been fixed to pull first; this file fixes the reason the trap fired at all.)

`nohup` ignores SIGHUP and `disown` removes the job from the shell's job table,
but NEITHER takes the process out of the parent's session, so a session- or
process-group-wide signal still reaches it. `setsid(1)` is not on macOS, so
`os.setsid()` in a double-forked child is the way to get a new session, and the
fork is doubled so the session leader exits immediately and the survivor can
never reacquire a controlling terminal.

Usage:
    python3 scripts/mlsys_detach.py --pidfile P --log L -- cmd arg...

The command's stdout and stderr are appended to L, its stdin is /dev/null (a
command that cannot read a dead terminal cannot be stopped by its absence), and
the surviving pid is written to P so the operator can find it again.

Returns 0 as soon as the child is forked; the child keeps running.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pidfile", default=None,
                    help="write the surviving pid here")
    ap.add_argument("--log", default=None,
                    help="append the command's stdout and stderr here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- then the command and its arguments")
    args = ap.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("FATAL: no command given (use: -- cmd arg...)", file=sys.stderr)
        return 2

    # FIRST fork: the caller's shell gets control back immediately.
    if os.fork() > 0:
        return 0

    # The child becomes a session leader, which is the part nohup cannot do.
    # It has no controlling terminal from here on, so nothing about the parent
    # session -- a runtime restart, a closed terminal, a group signal -- reaches
    # the command.
    os.setsid()

    # SECOND fork: the session leader exits, so the survivor is not a session
    # leader and can never acquire a controlling terminal by opening a tty.
    if os.fork() > 0:
        os._exit(0)

    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    if args.log:
        fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)

    if args.pidfile:
        with open(args.pidfile, "w") as fh:
            fh.write(f"{os.getpid()}\n")

    os.execvp(cmd[0], cmd)          # never returns
    return 0                        # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
