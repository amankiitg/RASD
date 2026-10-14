#!/usr/bin/env bash
# THE PARAMETER CONTRACT between a Mac-side driver and a pod-side script.
#
# WHY (2026-10-12, $5.47). A vLLM 8x run was launched with REHEARSE_TP=8,
# REHEARSE_CTXS='131072 524288 1048576' and REHEARSE_PROMPT_DIR=samp_prompts. The
# driver's remote command forwarded only PROFILE_OUT/PROFILE_TAG/PROFILE_NPROC, so
# the pod started with the WRAPPER'S OWN DEFAULTS -- one card, 8192 tokens, a
# synthetic prompt -- on an 8-GPU box. Nothing complained: the wrapper printed
# "### running the baseline: ... @ [8192] tokens, TP=1" into a log nobody was reading
# yet, and the CSV held one row at the wrong context.
#
# What was missing was not only the forwarding. A pod-side script had no way to know
# what it was SUPPOSED to have received. This file supplies that: the driver sends a
# manifest of its intended values in its OWN variable, and the remote script compares
# what it actually got against it BEFORE any heavy work -- before the venv install,
# before the model load, before the first stage.
#
# Two properties matter:
#   * the manifest travels in a DIFFERENT VARIABLE from the settings, so a setting
#     dropped in transit is still detectable: its intended value arrives anyway;
#   * a MISSING manifest is itself a failure. An unverifiable parameter set is exactly
#     the state that cost $5.47, so this refuses to guess rather than proceed.
#
# Sourced, not executed:
#   . scripts/mlsys_remote_expect.sh
#   mlsys_expect "vllm-remote" "$REMOTE_EXPECT" \
#     "CTXS=$CTXS;TP=$TP;PROMPT_DIR=$PROMPT_DIR;OUT=$OUT" || exit 2
#
# Both arguments are "K=V;K=V" strings. Every split appends a newline first: `read`
# returns non-zero on an unterminated final line, so splitting "A;B" without one skips
# the LAST pair -- which is how a dropped TP=8 first went unnoticed here.
# Splitting is on the FIRST '=' of each pair, so a value may contain '=' and may
# contain spaces (a context list). Deliberately bash 3.2 (macOS) and dependency-free:
# no associative arrays, no temp files.

# One implementation of the comparison, used by every caller. Emits one line per
# problem: "key|intended|received-or-<absent>". No temp files and no associative
# arrays (bash 3.2); keys are [A-Za-z_][A-Za-z0-9_]* so they are safe in a regex, and
# the value comparison is an exact -F match on "K=V" so a value full of regex
# metacharacters (a context list, a path) cannot be misread.
_mlsys_kv_problems() {
  local manifest="$1" received="$2"
  local recv_lines kv k want got
  # A caller passes `<absent>` for a setting that never arrived, so that a dropped
  # forward reads as "never received that setting" instead of as the default the
  # pod silently fell back to. Drop those pairs here: the value is not an arrival.
  recv_lines=$(printf '%s\n' "$received" | tr ';' '\n' | grep -v '^$' | grep -v '=<absent>$')
  printf '%s\n' "$manifest" | tr ';' '\n' | while IFS= read -r kv; do
    [ -n "$kv" ] || continue
    k=${kv%%=*}
    [ "$k" = "$kv" ] && continue                    # not a pair
    want=${kv#*=}
    printf '%s\n' "$recv_lines" | grep -Fxq "$k=$want" && continue
    if printf '%s\n' "$recv_lines" | grep -q "^$k="; then
      got=$(printf '%s\n' "$recv_lines" | sed -n "s/^$k=//p" | head -1)
      printf '%s|%s|%s\n' "$k" "$want" "$got"
    else
      printf '%s|%s|<absent>\n' "$k" "$want"
    fi
  done
}

# mlsys_expect <label> <manifest> <received>  -> 0 when every intended setting arrived
# intact, 2 otherwise. Prints the received block, the intended block, and the reason.
mlsys_expect() {
  local label="$1" manifest="$2" received="$3" kv bad n

  printf '%s: parameters as received\n' "$label"
  printf '%s\n' "$received" | tr ';' '\n' | while IFS= read -r kv; do
    [ -n "$kv" ] || continue
    printf '  %-14s = %s\n' "${kv%%=*}" "${kv#*=}"
  done
  printf '%s: intended by the driver\n' "$label"
  if [ -z "$manifest" ]; then
    printf '  (nothing: REMOTE_EXPECT is empty)\n'
  else
    printf '%s\n' "$manifest" | tr ';' '\n' | while IFS= read -r kv; do
      [ -n "$kv" ] || continue
      printf '  %-14s = %s\n' "${kv%%=*}" "${kv#*=}"
    done
  fi

  if [ -z "$manifest" ]; then
    printf '%s: FATAL no expectation manifest arrived (REMOTE_EXPECT is empty).\n' "$label" >&2
    printf '%s:   The pod cannot verify that it received the parameters the driver\n' "$label" >&2
    printf '%s:   intended, and running unverified is the failure this check exists\n' "$label" >&2
    printf '%s:   for (2026-10-12: one card, 8192 tokens, a synthetic prompt, on an\n' "$label" >&2
    printf '%s:   8-GPU box, ~$5.5). Refusing before any heavy work.\n' "$label" >&2
    return 2
  fi

  bad=$(_mlsys_kv_problems "$manifest" "$received")
  if [ -n "$bad" ]; then
    printf '%s\n' "$bad" | while IFS='|' read -r k want got; do
      if [ "$got" = "<absent>" ]; then
        printf '%s: FATAL %s was intended as "%s" but the pod never received that setting, so it is silently using a default.\n' \
          "$label" "$k" "$want" >&2
      else
        printf '%s: FATAL %s arrived as "%s" but the driver intended "%s".\n' \
          "$label" "$k" "$got" "$want" >&2
      fi
    done
    n=$(printf '%s' "$bad" | grep -c . )
    printf '%s: refusing to start: %s parameter(s) differ from what the driver intended.\n' \
      "$label" "$n" >&2
    printf '%s:   This is the check that would have caught the 2026-10-12 8x run.\n' "$label" >&2
    return 2
  fi
  n=$(printf '%s\n' "$manifest" | tr ';' '\n' | grep -c . )
  printf '%s: all %s expected setting(s) verified\n' "$label" "$n"
  return 0
}
