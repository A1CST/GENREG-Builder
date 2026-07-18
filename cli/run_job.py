"""Dispatch a training script to a running I2 node (primary or secondary),
signed with the same admin Ed25519 key push_to_primary.py uses. The node
must already have the script on disk (pushed via push_to_primary.py, whose
whitelist covers corpora/wikipedia/build/ and jobs/) and it must match
i2_node.py's JOB_WHITELIST.

  python run_job.py --node http://10.0.0.15:8800 corpora/wikipedia/build/foo.py
  python run_job.py --node http://<poweredge-ip>:8800 jobs/smoke_test.py --watch
  python run_job.py --node http://127.0.0.1:8800 --status <job_id>
  python run_job.py --node http://127.0.0.1:8800 --cancel <job_id>
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(APP_DIR, "i2_admin_key.json")


def load_priv():
    try:
        with open(KEY_PATH, "r", encoding="utf-8") as fh:
            raw = base64.b64decode(json.load(fh)["private"])
    except (OSError, ValueError, KeyError):
        sys.exit(f"no admin private key at {KEY_PATH} — run the keygen step first")
    from cryptography.hazmat.primitives.asymmetric import ed25519
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw)


def sign(prefix, body):
    msg = prefix + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(load_priv().sign(msg)).decode("ascii")


def _http_json(url, data=None, timeout=15):
    req = urllib.request.Request(url)
    if data is not None:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            body = json.load(exc)
        except ValueError:
            body = {"error": str(exc)}
        sys.exit(f"request failed ({exc.code}): {body.get('error', body)}")
    except urllib.error.URLError as exc:
        sys.exit(f"node unreachable: {exc}")


def submit(node, script, args):
    doc = {"ts": int(time.time()), "script": script, "args": args}
    doc["sig"] = sign(b"i2job\x00", {k: doc[k] for k in doc if k != "sig"})
    res = _http_json(node + "/api/i2/admin/job/submit", data=doc)
    print(f"submitted {script} -> job {res['job_id']}")
    return res["job_id"]


def poll(node, job_id, interval=3):
    # Remote logs can contain characters the local console codepage (cp1252)
    # can't encode; without this the watcher dies mid-stream on a
    # UnicodeEncodeError while the remote job keeps running.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    last_len = 0
    while True:
        st = _http_json(node + f"/api/i2/admin/job/{job_id}/status")
        log = _http_json(node + f"/api/i2/admin/job/{job_id}/log")["log"]
        if len(log) > last_len:
            sys.stdout.write(log[last_len:])
            sys.stdout.flush()
            last_len = len(log)
        if st["status"] in ("done", "failed", "cancelled"):
            print(f"\n-- job {job_id} {st['status']} (rc={st['returncode']}) --")
            return st
        time.sleep(interval)


def cancel(node, job_id):
    doc = {"ts": int(time.time())}
    doc["sig"] = sign(b"i2job\x00", {k: doc[k] for k in doc if k != "sig"})
    print(_http_json(node + f"/api/i2/admin/job/{job_id}/cancel", data=doc))


def retry(node, job_id):
    doc = {"ts": int(time.time())}
    doc["sig"] = sign(b"i2job\x00", {k: doc[k] for k in doc if k != "sig"})
    print(_http_json(node + f"/api/i2/admin/job/{job_id}/retry", data=doc))


def pause(node):
    doc = {"ts": int(time.time())}
    doc["sig"] = sign(b"i2job\x00", {k: doc[k] for k in doc if k != "sig"})
    print(_http_json(node + "/api/i2/admin/queue/pause", data=doc))


def resume(node):
    doc = {"ts": int(time.time())}
    doc["sig"] = sign(b"i2job\x00", {k: doc[k] for k in doc if k != "sig"})
    print(_http_json(node + "/api/i2/admin/queue/resume", data=doc))


def fetch_artifact(node, remote_path, local_path=None):
    """Pull a training script's own log/.npz output back from a remote node
    (distinct from the captured stdout at /job/<id>/log)."""
    res = _http_json(node + "/api/i2/admin/artifact?path=" + urllib.parse.quote(remote_path))
    data = base64.b64decode(res["content_b64"])
    dst = local_path or os.path.join(APP_DIR, remote_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(data)
    print(f"fetched {remote_path} ({res['size']:,} bytes) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description="Submit a signed job to an I2 node")
    ap.add_argument("script", nargs="?", help="whitelisted script path (relative to repo root)")
    ap.add_argument("job_args", nargs="*", help="args passed through to the script")
    ap.add_argument("--node", required=True, help="node URL, e.g. http://10.0.0.15:8800")
    ap.add_argument("--watch", action="store_true", help="poll status + stream log until done")
    ap.add_argument("--status", metavar="JOB_ID", help="print status + log for an existing job")
    ap.add_argument("--cancel", metavar="JOB_ID", help="cancel a queued/running job")
    ap.add_argument("--retry", metavar="JOB_ID", help="requeue a failed/cancelled/interrupted/done job")
    ap.add_argument("--pause", action="store_true", help="pause this node's job queue (running job finishes)")
    ap.add_argument("--resume", action="store_true", help="resume this node's job queue")
    ap.add_argument("--queue", action="store_true", help="print queue status (paused, pending, running)")
    ap.add_argument("--fetch", metavar="REMOTE_PATH",
                    help="pull a script's own log/.npz output back to the same relative path locally")
    args = ap.parse_args()
    node = args.node.rstrip("/")

    if args.fetch:
        fetch_artifact(node, args.fetch)
        return
    if args.pause:
        pause(node); return
    if args.resume:
        resume(node); return
    if args.queue:
        print(json.dumps(_http_json(node + "/api/i2/admin/queue/status"), indent=2)); return
    if args.retry:
        retry(node, args.retry)
        return
    if args.cancel:
        cancel(node, args.cancel)
        return
    if args.status:
        print(json.dumps(_http_json(node + f"/api/i2/admin/job/{args.status}/status"), indent=2))
        print(_http_json(node + f"/api/i2/admin/job/{args.status}/log")["log"])
        return
    if not args.script:
        sys.exit("need a script to run (or --status/--cancel)")

    job_id = submit(node, args.script, args.job_args)
    if args.watch:
        poll(node, job_id)


if __name__ == "__main__":
    main()
