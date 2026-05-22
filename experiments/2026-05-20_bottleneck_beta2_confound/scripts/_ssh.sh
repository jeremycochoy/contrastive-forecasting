#!/bin/bash
# Sourced helper. ssh_coords <inst_id> -> echoes "HOST PORT" (re-resolved
# live; the vast.ai SSH proxy IP/port can rotate over a run's lifetime).
ssh_coords(){
  local id="$1"
  vastrun-forward --force show instance "$id" --raw 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
h,p=d.get("ssh_host"),d.get("ssh_port")
if not (h and p):
    ip=d.get("public_ipaddr"); m=(d.get("ports") or {}).get("22/tcp") or []
    if ip and m: h,p=ip,m[0].get("HostPort")
if h and p: print(h,p)
else: sys.exit(1)
'
}
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
