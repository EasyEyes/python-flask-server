# python-flask-server

This Flask service performs server-side signal processing for speaker
calibration. `speaker-calibration/src/server/PythonServerAPI.js` references the
public endpoint `https://easyeyes-python-flask-server.herokuapp.com`, which was
reachable during the 2026-08-07 security review. The individual maintainer and
current deployment pipeline are not yet documented.

## Runtime

Use Python 3.11, as recorded in `.python-version`. The scientific dependency
pins do not build under Python 3.13 in the current development environment.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip_audit --local
```

## Security configuration

- `EASYEYES_ALLOWED_ORIGINS`: comma-separated exact browser origins allowed to
  call `/task/*` and `/model/*`. There is no wildcard or default browser origin.
- `EASYEYES_MAX_CONTENT_LENGTH`: maximum request body in bytes. The default is
  33,554,432 bytes (32 MiB) to accommodate calibration arrays while bounding
  memory use.
- `EASYEYES_DIAGNOSTICS_TOKEN`: bearer token for `/memory` and `/snapshot`.
  Those routes return `404` when no token is configured and `401` for an
  incorrect token.

Task routes remain an unauthenticated computation API because the active
client contract does not provide an identity mechanism. Before deploying this
change, operators must configure the exact active origins and verify realistic
calibration payload sizes. A public deployment still needs an upstream rate
limit, request-duration/CPU controls, monitoring, a named owner, and a decision
on whether task authentication is required.

The older `python-server` repository is a near-duplicate but no active client
reference was found. Confirm and retire it instead of maintaining two public
copies of this service.
