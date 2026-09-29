#!/usr/bin/env bash
#
# Start, stop or inspect the SMA DQM analyzer: `python -m mdqm.dqm.analyzer
# --plugin sma`, MIDAS client `sma_analyzer`, as a low-priority guest on the DAQ PC.
#
#     scripts/start-sma-analyzer.sh --experiment bt2026            # in a tmux session
#     scripts/start-sma-analyzer.sh --experiment bt2026 --status
#     scripts/start-sma-analyzer.sh --experiment bt2026 --stop
#     scripts/start-sma-analyzer.sh --experiment bt2026 --foreground   # here, for debugging
#     scripts/start-sma-analyzer.sh ... -- --no-cpu-budget            # after --: to the analyzer
#
# Self-contained: needs only this checkout (not pip-installed; it runs with
# PYTHONPATH=<checkout>/src), a python with numpy and the MIDAS bindings, and
# MIDAS's odbedit. It reads the same environment as wavedream-scalar-readout's
# scripts (WDS_EXPT_NAME, WDS_PYTHON, WDS_TMUX_PREFIX), so it can be called from
# them without flags; see "3. Starting and stopping the analyzer" in docs/SMA-DQM.md.
#
# The analyzer is monitoring, not readout: no equipment, no transition callbacks,
# non-blocking buffer requests. It is safe to start and stop at any time; stopping
# it only loses the accumulated plots.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG="$(basename "$0")"

usage() {
    cat <<EOF
Usage: $PROG [options] [-- analyzer args...]

Start the SMA DQM analyzer (mdqm plugin "sma") in a tmux session, or stop it,
or report on it. Everything after -- goes to mdqm.dqm.analyzer unchanged.

Actions (default: start):
  --stop                 SIGTERM the analyzer, wait up to --stop-timeout s for it to
                         leave the MIDAS client list (SIGKILL after that), close its
                         tmux session. Exit 0 also when nothing was running.
  --status               client, process (nice, memory, RLIMIT_AS), tmux session, log.
                         Exit 0 running, 1 not running, 2 ODB unreadable.
  --dry-run              run the checks and print the command, start nothing.

Options (flag, else environment, else default):
  --experiment NAME      MIDAS experiment [MIDAS_EXPT_NAME, else WDS_EXPT_NAME]
  --python PATH          python with numpy + midas [MDQM_PYTHON, else WDS_PYTHON, else python3]
  --client NAME          MIDAS client name [sma_analyzer]
  --session NAME         tmux session [\$WDS_TMUX_PREFIX-<client with - for _> if
                         WDS_TMUX_PREFIX is set, else <client with - for _>,
                         i.e. sma-analyzer]
  --log FILE|none        output copy [MDQM_LOG_DIR, else \${XDG_STATE_HOME:-~/.local/state}/mdqm,
                         file <experiment>-<client>.log]; rotated to .1 above 10 MiB
  --mem-limit BYTES      RLIMIT_AS via prlimit [1073741824 = 1 GiB]; 0 = no limit
  --cpu-pin CPUS         run under taskset -c CPUS (e.g. 3 or 2,3) [off]
  --stop-timeout S       seconds --stop waits for the client to detach [20]
  --foreground           exec the analyzer in this terminal (debugging; no tmux, no log)
  --no-tmux              run it detached without tmux (setsid); output only to the log
  -h, --help             this text

Always applied: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MALLOC_ARENA_MAX=2, nice -n 19, ionice -c3 (if available). MIDASSYS and MIDAS_EXPTAB
are passed on when set; MIDASSYS is taken from odbedit's location otherwise.

Refuses to start while a client of the same name is attached to the experiment,
an analyzer with that client name runs on this machine, or the tmux session has
a live process: two analyzers would split the sampled frames between them.
EOF
}

die()  { echo "$PROG: ERROR: $*" >&2; exit 1; }
warn() { echo "$PROG: WARNING: $*" >&2; }
info() { echo "$PROG: $*"; }

# -- options -----------------------------------------------------------------------

ACTION=start
MODE=tmux
EXPT="${MIDAS_EXPT_NAME:-${WDS_EXPT_NAME:-}}"
PY="${MDQM_PYTHON:-${WDS_PYTHON:-python3}}"
CLIENT=sma_analyzer
SESSION=""
LOG=""
LOG_SET=0
MEM=1073741824
PIN=""
STOP_TIMEOUT=20
START_TIMEOUT=30
EXTRA=()

while [ $# -gt 0 ]; do
    opt="$1"
    has_val=0
    val=""
    case "$opt" in
        --*=*) val="${opt#*=}"; opt="${opt%%=*}"; has_val=1 ;;
    esac
    case "$opt" in
        --experiment|--python|--client|--session|--log|--mem-limit|--cpu-pin|--stop-timeout)
            if [ "$has_val" = 0 ]; then
                [ $# -ge 2 ] || die "$opt needs a value (see --help)"
                val="$2"
                shift
            fi
            case "$opt" in
                --experiment)   EXPT="$val" ;;
                --python)       PY="$val" ;;
                --client)       CLIENT="$val" ;;
                --session)      SESSION="$val" ;;
                --log)          LOG="$val"; LOG_SET=1 ;;
                --mem-limit)    MEM="$val" ;;
                --cpu-pin)      PIN="$val" ;;
                --stop-timeout) STOP_TIMEOUT="$val" ;;
            esac
            ;;
        --stop|--status|--dry-run|--foreground|--no-tmux)
            [ "$has_val" = 0 ] || die "$opt takes no value"
            case "$opt" in
                --stop)       ACTION=stop ;;
                --status)     ACTION=status ;;
                --dry-run)    ACTION=dry-run ;;
                --foreground) MODE=foreground ;;
                --no-tmux)    MODE=detached ;;
            esac
            ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA=("$@"); break ;;
        *)            die "unknown option '$1' (analyzer options go after --; see --help)" ;;
    esac
    shift
done

[ -n "$EXPT" ] || die "no experiment: pass --experiment NAME or set MIDAS_EXPT_NAME"
[ -n "$CLIENT" ] || die "--client must not be empty"
case "$MEM" in ''|*[!0-9]*) die "--mem-limit takes a number of bytes (0 = no limit), not '$MEM'" ;; esac
case "$STOP_TIMEOUT" in ''|*[!0-9]*) die "--stop-timeout takes whole seconds, not '$STOP_TIMEOUT'" ;; esac
if [ -n "$PIN" ]; then
    case "$PIN" in *[!0-9,-]*) die "--cpu-pin takes a CPU list for taskset -c (e.g. 3 or 2,3), not '$PIN'" ;; esac
fi
if [ -z "$SESSION" ]; then
    SESSION="${WDS_TMUX_PREFIX:+$WDS_TMUX_PREFIX-}${CLIENT//_/-}"
fi
case "$SESSION" in *[.:]*|"") die "tmux session name '$SESSION' must be non-empty and without . or :" ;; esac
if [ "$LOG_SET" = 0 ]; then
    LOG="${MDQM_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/mdqm}/$EXPT-$CLIENT.log"
fi
[ "$LOG" = none ] && LOG=""

# -- MIDAS: odbedit, MIDASSYS ------------------------------------------------------

ODBEDIT="$(command -v odbedit 2>/dev/null || true)"
if [ -z "$ODBEDIT" ] && [ -n "${MIDASSYS:-}" ] && [ -x "$MIDASSYS/bin/odbedit" ]; then
    ODBEDIT="$MIDASSYS/bin/odbedit"
fi
if [ -z "${MIDASSYS:-}" ] && [ -n "$ODBEDIT" ]; then
    # odbedit lives in $MIDASSYS/bin; the python bindings need MIDASSYS to find the library.
    MIDASSYS="$(cd "$(dirname "$(readlink -f "$ODBEDIT")")/.." && pwd)"
fi
[ -n "$ODBEDIT" ] || die "odbedit not found on PATH or in \$MIDASSYS/bin; set MIDASSYS or PATH"
export MIDASSYS

# Print "<pid><TAB><host><TAB><name>" for every client in /System/Clients, live or not.
# Returns 1 if the ODB could not be read. Parses the whole listing in one go:
# odbedit's ls does not expand wildcards, and names may contain spaces.
odb_clients() {
    local listing
    listing=$("$ODBEDIT" -e "$EXPT" -q -c "ls -lr /System/Clients" 2>/dev/null) || return 1
    grep -q '^Clients[[:space:]]*DIR' <<< "$listing" || return 1
    awk 'function flush() { if (pid != "" && name != "") print pid "\t" host "\t" name }
         $2=="DIR" && $1 ~ /^[0-9]+$/ { flush(); pid=$1; name=""; host=""; next }
         ($1=="Name" || $1=="Host") && $2=="STRING" {
             v=$8; for (i=9; i<=NF; i++) v = v " " $i
             if ($1=="Name") name=v; else host=v }
         END { flush() }' <<< "$listing"
}

# True if this PID is a running process. A zombie keeps /proc/<pid> (tmux does not reap
# a remain-on-exit pane's process), so read the state rather than test the directory.
pid_alive() {
    local stat rest state
    # cat, not $(< file): bash 5.2 under set -e exits the whole script when $(< file)
    # hits a missing file inside a function, whatever the || -- and a PID that has just
    # exited is exactly the case this function is for.
    stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
    rest=${stat##*) }
    state=${rest%% *}
    case "$state" in Z|X|x|"") return 1 ;; *) return 0 ;; esac
}

is_local_host() {
    case "$1" in
        ""|localhost|127.0.0.1|"$(hostname 2>/dev/null)"|"$(hostname -s 2>/dev/null)"|"$(hostname -f 2>/dev/null)") return 0 ;;
    esac
    return 1
}

# Where is a client called $CLIENT? Sets CLIENT_PID (local) or CLIENT_HOST (remote).
# 0 = attached, 1 = not (an entry whose local PID is gone is stale: a client killed
# without deregistering stays listed until another client's watchdog reaps it),
# 2 = ODB unreadable.
CLIENT_PID=""
CLIENT_HOST=""
client_state() {
    local clients pid host name
    CLIENT_PID=""
    CLIENT_HOST=""
    clients=$(odb_clients) || return 2
    while IFS=$'\t' read -r pid host name; do
        [ "$name" = "$CLIENT" ] || continue
        if is_local_host "$host"; then
            if pid_alive "$pid"; then CLIENT_PID="$pid"; return 0; fi
        else
            CLIENT_HOST="$host"; CLIENT_PID="$pid"; return 0
        fi
    done <<< "$clients"
    return 1
}

# PIDs on this machine running mdqm.dqm.analyzer (or the mdqm-analyzer entry point) with
# this client name for this experiment -- including one that is not attached, e.g.
# sitting in its reconnect loop after MIDAS went away.
analyzer_pids() {
    local d pid args a i is_mdqm client expt envexpt
    for d in /proc/[0-9]*; do
        pid="${d#/proc/}"
        [ "$pid" = "$$" ] && continue
        args=()
        { mapfile -d '' -t args < "$d/cmdline"; } 2>/dev/null || continue
        [ "${#args[@]}" -gt 1 ] || continue
        is_mdqm=0; client=""; expt=""
        for ((i = 0; i < ${#args[@]}; i++)); do
            a="${args[$i]}"
            case "$a" in
                mdqm.dqm.analyzer|*/mdqm-analyzer|mdqm-analyzer) is_mdqm=1 ;;
                --client)       client="${args[$((i + 1))]:-}" ;;
                --client=*)     client="${a#--client=}" ;;
                --experiment)   expt="${args[$((i + 1))]:-}" ;;
                --experiment=*) expt="${a#--experiment=}" ;;
            esac
        done
        [ "$is_mdqm" = 1 ] && [ "$client" = "$CLIENT" ] || continue
        if [ -z "$expt" ]; then
            envexpt=$({ tr '\0' '\n' < "$d/environ"; } 2>/dev/null | sed -n 's/^MIDAS_EXPT_NAME=//p')
            expt="$envexpt"
        fi
        [ "$expt" = "$EXPT" ] || continue
        pid_alive "$pid" && echo "$pid"
    done
    return 0
}

have_tmux() { command -v tmux > /dev/null 2>&1; }
tmux_exists() { have_tmux && tmux has-session -t "=$SESSION" 2>/dev/null; }
# The session exists AND its pane process is alive (remain-on-exit keeps dead panes).
tmux_live() {
    local dead
    have_tmux || return 1
    dead=$(tmux list-panes -t "=$SESSION" -F '#{pane_dead}' 2>/dev/null) || return 1
    grep -qx 0 <<< "$dead"
}

how_to_stop() {
    local self
    self="$REPO/scripts/$PROG --experiment $EXPT"
    [ "$CLIENT" = sma_analyzer ] || self+=" --client $CLIENT"
    echo "           $self --stop"
}

# -- status ------------------------------------------------------------------------

proc_summary() {
    local pid="$1" ni vms rss as
    ni=$(ps -o ni= -p "$pid" 2>/dev/null | tr -d ' ')
    vms=$(awk '/^VmSize/ {print $2 " " $3}' "/proc/$pid/status" 2>/dev/null)
    rss=$(awk '/^VmRSS/ {print $2 " " $3}' "/proc/$pid/status" 2>/dev/null)
    as=$(awk '/^Max address space/ {print $4}' "/proc/$pid/limits" 2>/dev/null)
    echo "  process   pid $pid: nice ${ni:-?}, VmSize ${vms:-?}, VmRSS ${rss:-?}, RLIMIT_AS ${as:-?}"
    echo "            $({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null)"
}

do_status() {
    local st=0 pids pid rc=1
    echo "$CLIENT on experiment $EXPT"
    client_state || st=$?
    case "$st" in
        0) if [ -n "$CLIENT_HOST" ]; then
               echo "  MIDAS     attached from host $CLIENT_HOST (pid $CLIENT_PID there)"
           else
               echo "  MIDAS     attached, pid $CLIENT_PID"
           fi
           rc=0 ;;
        1) echo "  MIDAS     not attached" ;;
        2) echo "  MIDAS     ODB of '$EXPT' unreadable (is the experiment running? MIDAS_EXPTAB=${MIDAS_EXPTAB:-unset})"
           rc=2 ;;
    esac
    pids=$(analyzer_pids)
    if [ -n "$pids" ]; then
        for pid in $pids; do proc_summary "$pid"; done
        [ "$rc" = 1 ] && { echo "  (running but not attached: reconnecting, see the log)"; rc=0; }
    else
        echo "  process   none on this machine"
    fi
    if ! have_tmux; then
        echo "  tmux      not installed"
    elif tmux_live; then
        echo "  tmux      session $SESSION running   (tmux attach -t $SESSION)"
    elif tmux_exists; then
        echo "  tmux      session $SESSION has a dead pane (the analyzer exited; its output is there)"
    else
        echo "  tmux      no session $SESSION"
    fi
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        echo "  log       $LOG, last lines:"
        tail -n 5 "$LOG" | cut -c 1-150 | sed 's/^/            /'
    else
        echo "  log       ${LOG:-none} (not written yet)"
    fi
    echo "  skip_method, sampling, frame counts: brpc dqm::status to $CLIENT, shown on SMAPlots"
    return "$rc"
}

# -- stop --------------------------------------------------------------------------

# Wait for a PID to exit. The analyzer deregisters from MIDAS on SIGTERM before it
# exits, and client_state ignores entries whose PID is gone, so "exited" is "detached".
wait_gone() {
    local pid="$1" timeout="$2" waited=0
    while [ "$waited" -lt $((timeout * 5)) ]; do
        pid_alive "$pid" || return 0
        sleep 0.2
        waited=$((waited + 1))
    done
    return 1
}

stop_pid() {
    local pid="$1"
    pid_alive "$pid" || return 0
    if ! { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null | grep -qE 'mdqm(\.dqm\.analyzer|-analyzer)'; then
        die "client '$CLIENT' (pid $pid) is not an mdqm analyzer; not touching it"
    fi
    info "stopping $CLIENT (pid $pid): SIGTERM, waiting up to ${STOP_TIMEOUT}s for it to detach"
    kill -TERM "$pid" 2>/dev/null || true
    if wait_gone "$pid" "$STOP_TIMEOUT"; then
        info "$CLIENT detached"
    else
        warn "$CLIENT (pid $pid) still there after ${STOP_TIMEOUT}s; SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
}

do_stop() {
    local st=0 pid stopped=0 pane
    client_state || st=$?
    case "$st" in
        0) if [ -n "$CLIENT_HOST" ]; then
               die "$CLIENT is attached from host $CLIENT_HOST (pid $CLIENT_PID there); stop it on that host"
           fi
           stop_pid "$CLIENT_PID"; stopped=1 ;;
        2) warn "ODB of '$EXPT' unreadable; stopping by process table and tmux only" ;;
    esac
    for pid in $(analyzer_pids); do stop_pid "$pid"; stopped=1; done
    if tmux_live; then
        pane=$(tmux list-panes -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null | head -n 1)
        if [ -n "$pane" ] && pid_alive "$pane"; then
            info "tmux session $SESSION still has pid $pane; SIGTERM"
            kill -TERM "$pane" 2>/dev/null || true
            wait_gone "$pane" "$STOP_TIMEOUT" || kill -KILL "$pane" 2>/dev/null || true
            stopped=1
        fi
    fi
    if tmux_exists; then
        tmux kill-session -t "=$SESSION" 2>/dev/null && info "closed tmux session $SESSION"
    fi
    [ "$stopped" = 1 ] || info "$CLIENT was not running on $EXPT"
    return 0
}

# -- start -------------------------------------------------------------------------

SKIP_METHOD=""
preflight() {
    local out
    command -v "$PY" > /dev/null 2>&1 || die "python '$PY' not found; pass --python or set MDQM_PYTHON"
    PY="$(command -v "$PY")"
    [ -d "$REPO/src/mdqm" ] || die "no $REPO/src/mdqm: this script must stay in the checkout's scripts/"
    [ -e "$MIDASSYS/lib/libmidas-c-compat.so" ] || [ -e "$MIDASSYS/lib/libmidas-c-compat.dylib" ] ||
        die "no libmidas-c-compat in MIDASSYS=$MIDASSYS/lib (the python bindings need it)"

    # Under the same wrapper as the real start: numpy under an address-space limit is
    # exactly what fails if the thread settings are missing.
    # shellcheck disable=SC2016  # $MIDASSYS is expanded by python, not the shell
    if ! out=$("${WRAP[@]}" "$PY" -c '
import ctypes, glob, os
import numpy, midas, midas.client, mdqm.dqm.analyzer
lib = ctypes.CDLL(sorted(glob.glob(os.path.join(os.environ["MIDASSYS"], "lib", "libmidas-c-compat.*")))[0])
names = ("c_bm_skip_event", "_Z13bm_skip_eventi")
print("bm_skip_event" if any(hasattr(lib, n) for n in names) else "drain")
' 2>&1); then
        echo "$out" | tail -n 5 >&2
        die "'$PY' cannot import numpy, midas and mdqm (PYTHONPATH=$PYTHONPATH, RLIMIT_AS $MEM)"
    fi
    SKIP_METHOD="$(tail -n 1 <<< "$out")"
    if [ "$SKIP_METHOD" != bm_skip_event ]; then
        warn "the MIDAS library has no bm_skip_event: the analyzer will drop frames by reading"
        warn "them (skip_method=drain, a copy each, still within its CPU budget)"
    fi

    local st=0
    client_state || st=$?
    case "$st" in
        0)  if [ -n "$CLIENT_HOST" ]; then
                die "'$CLIENT' is already attached to $EXPT from host $CLIENT_HOST (pid $CLIENT_PID there).
       Two analyzers would split the sampled frames between them. Stop that one first."
            fi
            die "'$CLIENT' is already attached to $EXPT (pid $CLIENT_PID).
       Two analyzers would split the sampled frames between them and both sets
       of plots would be half-filled. Stop it first:
$(how_to_stop)" ;;
        2)  die "cannot read /System/Clients of experiment '$EXPT' with $ODBEDIT
       (MIDAS_EXPTAB=${MIDAS_EXPTAB:-unset}). Refusing to start blind: is the experiment up?" ;;
    esac
    local pids
    pids=$(analyzer_pids)
    if [ -n "$pids" ]; then
        die "an analyzer with client '$CLIENT' for $EXPT is already running here (pid $(echo "$pids" | tr '\n' ' '))
       but not attached (reconnecting?). Stop it first:
$(how_to_stop)"
    fi
    if tmux_live; then
        die "tmux session $SESSION still has a running process. Look (tmux attach -t $SESSION)
       or stop it:
$(how_to_stop)"
    fi

    if [ "$MODE" = tmux ] && ! have_tmux; then
        if [ "$ACTION" = dry-run ]; then warn "tmux not found; the start would fail (use --no-tmux or --foreground)"
        else die "tmux not found; install it, or use --no-tmux (log only) or --foreground"; fi
    fi
    return 0
}

build_wrap() {
    WRAP=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
          PYTHONUNBUFFERED=1 "PYTHONPATH=$PYTHONPATH" "MIDASSYS=$MIDASSYS" "PATH=$PATH")
    [ -n "${MIDAS_EXPTAB:-}" ] && WRAP+=("MIDAS_EXPTAB=$MIDAS_EXPTAB")
    WRAP+=(nice -n 19)
    if command -v ionice > /dev/null 2>&1; then
        WRAP+=(ionice -c3)
    else
        warn "ionice not found; running without the idle I/O class"
    fi
    if [ -n "$PIN" ]; then
        if command -v taskset > /dev/null 2>&1; then WRAP+=(taskset -c "$PIN")
        else warn "taskset not found; --cpu-pin $PIN ignored"; fi
    fi
    if [ "$MEM" != 0 ]; then
        if command -v prlimit > /dev/null 2>&1; then WRAP+=(prlimit "--as=$MEM")
        else warn "prlimit not found; running without the address-space limit"; MEM=0; fi
    fi
}

prepare_log() {
    [ -n "$LOG" ] || return 0
    mkdir -p "$(dirname "$LOG")" || die "cannot create the log directory for $LOG"
    if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt 10485760 ]; then
        mv -f "$LOG" "$LOG.1"
    fi
    : >> "$LOG" || die "cannot write the log $LOG (pass --log FILE or --log none)"
}

log_tail_hint() {
    if [ "$MODE" = tmux ]; then
        echo "           tmux capture-pane -p -S - -t $SESSION | tail" >&2
    fi
    if [ -n "$LOG" ]; then
        echo "           tail $LOG" >&2
        tail -n 8 "$LOG" 2>/dev/null | cut -c 1-200 | sed 's/^/    | /' >&2 || true
    fi
}

# The analyzer died before attaching (a dead tmux pane, or the detached PID gone).
started_process_gone() {
    case "$MODE" in
        tmux)     ! tmux_live ;;
        detached) ! pid_alive "$DETACHED_PID" ;;
    esac
}

DETACHED_PID=""
do_start() {
    local cmd quoted c inner waited=0 st
    cmd=("${WRAP[@]}" "$PY" -m mdqm.dqm.analyzer --experiment "$EXPT" --plugin sma --client "$CLIENT")
    if [ "${#EXTRA[@]}" -gt 0 ]; then cmd+=("${EXTRA[@]}"); fi

    quoted=""
    for c in "${cmd[@]}"; do quoted+=" $(printf '%q' "$c")"; done

    if [ "$ACTION" = dry-run ]; then
        echo "mode: $MODE   session: $SESSION   log: ${LOG:-none}   skip_method: $SKIP_METHOD"
        echo "command:$quoted"
        return 0
    fi

    if [ "$MODE" = foreground ]; then
        info "starting $CLIENT in the foreground (experiment $EXPT, skip_method $SKIP_METHOD)"
        echo "==$quoted"
        exec "${cmd[@]}"
    fi

    prepare_log
    if [ "$MODE" = detached ]; then
        {
            echo "== $(date '+%F %T') $PROG: detached start"
            echo "==$quoted"
        } >> "${LOG:-/dev/null}"
        setsid "${cmd[@]}" >> "${LOG:-/dev/null}" 2>&1 < /dev/null &
        DETACHED_PID=$!
        disown "$DETACHED_PID" 2>/dev/null || true
    else
        # A tmux session inherits the tmux server's environment, not ours, so every
        # variable the analyzer needs is on its command line (env ... above). exec keeps
        # the analyzer as the pane's process, so #{pane_dead} and #{pane_pid} are its.
        inner="echo \"== \$(date '+%F %T') $PROG: tmux start\"; echo $(printf '%q' "==$quoted"); exec$quoted"
        if [ -n "$LOG" ]; then
            inner="exec > >(tee -a $(printf '%q' "$LOG")) 2>&1; $inner"
        fi
        # A dead session (remain-on-exit after an exit) holds nothing; replace it.
        tmux kill-session -t "=$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" -c / "bash -c $(printf '%q' "$inner")"
        if ! tmux set-option -w -t "=$SESSION:" remain-on-exit on > /dev/null 2>&1; then
            warn "could not set remain-on-exit on $SESSION: if the analyzer dies, its output"
            warn "goes with the session (still in ${LOG:-no log})"
        fi
    fi

    # Poll for the client rather than sleeping: an analyzer that aborted in 50 ms must
    # not be reported as started.
    while [ "$waited" -lt $((START_TIMEOUT * 5)) ]; do
        st=0
        client_state || st=$?
        if [ "$st" = 0 ]; then
            info "started $CLIENT on $EXPT (pid $CLIENT_PID, skip_method $SKIP_METHOD)"
            [ "$MODE" = tmux ] && info "  tmux session: $SESSION   (tmux attach -t $SESSION)"
            [ -n "$LOG" ] && info "  log: $LOG"
            info "  status: $PROG --experiment $EXPT --status; stop: --stop"
            return 0
        fi
        if started_process_gone; then
            echo "$PROG: ERROR: $CLIENT exited before attaching to $EXPT. Its output:" >&2
            log_tail_hint
            exit 1
        fi
        sleep 0.2
        waited=$((waited + 1))
    done
    echo "$PROG: ERROR: $CLIENT did not attach to $EXPT within ${START_TIMEOUT}s (it may still be retrying):" >&2
    log_tail_hint
    exit 1
}

case "$ACTION" in
    status) do_status; exit $? ;;
    stop)   do_stop ;;
    start|dry-run)
        # This checkout first; MIDAS's own bindings last, as a fallback for a python
        # that does not have them already.
        PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
        case ":$PYTHONPATH:" in *":$MIDASSYS/python:"*) ;; *) PYTHONPATH+=":$MIDASSYS/python" ;; esac
        export PYTHONPATH
        [ "$MODE" = foreground ] && LOG=""
        build_wrap
        preflight
        do_start ;;
esac
