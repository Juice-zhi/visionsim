"""Smoke test of the fog viewer: start its server on a free port, wait for its warm-up, request configurations through
its API, and check its numbers against those of ``analysis/eval_closed_form.py`` for the same configuration.

Run it from the root of the main checkout, with the library to test first on ``PYTHONPATH`` if it isn't the
installed one::

    .conda/python.exe examples/medium/experiment/acceleration/viewer/smoke_test.py [--device cpu] [--threads 4] \\
        [--expected runs/fog-comparison/acceleration/analysis/eval/<name>.json]

Without ``--expected``, the day and night cases of ``eval_closed_form.py`` are evaluated in this process first, with the
same library, on the CPU (2-3 min with 4 threads, most of it the night's halo). Then the server is started, and:

- the page and its script and style are served;
- the default configuration of each scene (every switch and input on, factor 1, clean frames without fog) gives the
  errors of ``eval_closed_form.py`` for whole frames, the fog and surfaces, as is, blurred and as ratios, within float
  tolerance (rounding differs on the GPU);
- turning an effective switch off changes them, while turning off a switch whose prerequisites aren't met (a lamp's in
  the day scene) gives the very same frames, which the library computes as off;
- the prerequisites come from the library: without shadow maps, the shadows of suns are greyed out for lack of
  ``shadows``, and lamps' switches are greyed out by day;
- images come as float16 of the frame's size, factor 4 and Cycles' frames without fog are computed, and bad
  configurations, malformed requests and missing renders are refused with a message;
- a request of the page drops its earlier ones still queued, also when it is answered from memory;
- the table of what each switch does lists every switch of the library, measured or ineffective, and tells which
  ablations are of the same model as the library's once the default configuration is known;
- for a library whose ``apply_medium`` takes the albedo of each pixel, the page offers it as an input, off by default,
  which changes the surfaces' error when on, and whose image is served;
- a second run of the server names the images of the same configuration differently, so that a browser can't show
  those of the first run, and gives the same numbers.

The expected errors must come from the same library as the server's, as its fingerprint (a hash of its source) tells,
whichever checkout it sits in.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server import STUDY, fingerprint, library_path, study_root

KINDS = ("whole", "fog", "surfaces")
METRICS = ("rel_l1", "rel_l1_blurred", "ratio")


class Client:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def call(self, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, bytes, dict]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST")
        request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)

    def json(self, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
        status, raw, _ = self.call(path, body, headers)
        return status, json.loads(raw)

    def compute(self, config: dict, timeout: float = 900) -> dict:
        """Result of a configuration, polling until it is computed."""
        start = time.time()
        while time.time() - start < timeout:
            status, response = self.json("/api/compute", config)
            if status != 200 and response.get("state") != "loading":
                raise AssertionError(f"{config} was refused: {response}")
            if response["state"] == "done":
                return response
            if response["state"] == "error":
                raise AssertionError(f"{config} failed: {response['error']}")
            time.sleep(0.5)
        raise TimeoutError(f"{config} wasn't computed in {timeout} s")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def expected_errors(args) -> dict:
    """Errors of eval_closed_form.py for the day and night cases, from its JSON or evaluated here."""
    if args.expected:
        results = json.loads(Path(args.expected).read_text())
        print(f"Expected errors from {args.expected} (library {results.get('library')})", flush=True)
        return results
    # The same library as the server's
    if library := library_path(args.library):
        sys.path.insert(0, str(library))
    sys.path[1:1] = [str(Path(STUDY)), str(Path(STUDY) / "analysis")]
    import eval_closed_form
    import torch

    torch.set_num_threads(args.threads)  # common.py sets its own when imported
    print("Evaluating eval_closed_form.py's day and night cases in this process, on the CPU", flush=True)
    start = time.time()
    results = {"day": eval_closed_form.evaluate_day(1), "night": eval_closed_form.evaluate_night(1)}
    print(f"... in {time.time() - start:.0f} s", flush=True)
    return results


def main(args) -> int:
    root = study_root(args.root)
    os.chdir(root)
    os.environ.setdefault("PYTHONUTF8", "1")
    expected = expected_errors(args)
    port = free_port()
    command = [sys.executable, str(HERE / "server.py"), "--port", str(port), "--device", args.device]
    command += ["--threads", str(args.threads), "--root", str(root)]
    if args.library:
        command += ["--library", args.library]
    if args.gpu_lock:
        command += ["--gpu-lock", args.gpu_lock]
    log = open(args.log, "w", encoding="utf-8") if args.log else subprocess.DEVNULL
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    client = Client(port)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"{'ok  ' if condition else 'FAIL'} {message}", flush=True)
        if not condition:
            failures.append(message)

    def same_errors(case: str, result: dict) -> None:
        for kind in KINDS:
            for metric in METRICS:
                ours, theirs = result["metrics"]["eval"][kind][metric], expected[case][kind][metric]
                check(
                    math.isclose(ours, theirs, rel_tol=args.tolerance, abs_tol=args.tolerance),
                    f"{case} {kind} {metric}: viewer {ours:.6%} vs eval_closed_form {theirs:.6%}",
                )

    try:
        start = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"The server stopped with code {server.returncode}, see --log")
            try:
                status, body = client.json("/api/status")
                if status == 200 and body["warmup"].get("day", {}).get("state") == "done":
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            if time.time() - start > 900:
                raise TimeoutError("The day scene didn't warm up in 15 min")
            time.sleep(1)
        print(f"Day scene warm after {time.time() - start:.0f} s", flush=True)

        for path, text in (
            ("/", b"<title>Fog Control Panel</title>"),
            ("/app.js", b"/api/compute"),
            ("/style.css", b":root"),
        ):
            status, raw, _ = client.call(path)
            check(status == 200 and text in raw, f"GET {path} serves the page")
        status, schema = client.json("/api/schema")
        check(status == 200 and {"day", "night"} <= set(schema["scenes"]), "the schema lists both scenes")
        names = [s["name"] for s in schema["switches"]]
        print(f"Library {schema['library']}: {len(names)} switches, groups {list(schema['groups'])}", flush=True)
        check(bool(schema.get("instance")) and bool(schema.get("fingerprint")), "the schema names the run and library")
        status, body = client.json("/api/status")
        check(body.get("instance") == schema.get("instance"), f"the status names the same run: {body.get('instance')}")
        if expected.get("library"):
            # The same source, wherever it sits: an identical library in another checkout is the same library
            if (Path(expected["library"]) / "visionsim").is_dir():
                theirs = fingerprint(expected["library"])
                check(
                    theirs == schema["fingerprint"],
                    f"the expected errors come from the server's library: fingerprint {theirs} of "
                    f"{expected['library']} vs {schema['fingerprint']} of {schema['library']}",
                )
            else:
                print(f"note {expected['library']}, which made the expected errors, isn't there to compare", flush=True)

        def times(label: str, result: dict) -> None:
            t = result["timing"]
            first = f", first call {t['first']['seconds']:.3f} s (frame {t['first']['frame']})" if t["first"] else ""
            print(f"     {label}: warm {t['warm']:.3f} s{first}, {t['device_name']}", flush=True)

        day = client.compute({"scene": "day"})
        same_errors("day", day["result"])
        times("day, default", day["result"])
        height, width = schema["scenes"]["day"]["size"]
        status, raw, _ = client.call(f"/api/image?id={day['result']['id']}&name=radiance")
        check(
            status == 200 and len(raw) == width * height * 3 * 2,
            f"the closed form's image is float16 {height}x{width}x3",
        )
        status, raw, headers = client.call(f"/api/static?scene=day&frame=258&name=cycles&spp=4&v={schema['instance']}")
        check(status == 200 and len(raw) == width * height * 3 * 2, "Cycles' image at 4 spp is served")
        check("max-age" in headers.get("Cache-Control", ""), "a render named for this run may be kept by the browser")
        status, raw, headers = client.call("/api/static?scene=day&frame=258&name=cycles&spp=4&v=another-run")
        check(headers.get("Cache-Control") == "no-cache", "a render named for another run isn't kept as is")

        if names:
            missing = day["prerequisites"]["missing"]
            lamps = [n for n in names if missing[n] and "lamps" in missing[n]]
            check(bool(lamps), f"by day, switches that need lamps are greyed out: {lamps}")
            effective = [n for n in names if not missing[n]]
            chosen = "reflected_daylight" if "reflected_daylight" in effective else effective[0]
            off = client.compute({"scene": "day", "effects": {chosen: False}})["result"]["metrics"]["eval"]["whole"]
            on = day["result"]["metrics"]["eval"]["whole"]
            check(
                abs(off["rel_l1"] - on["rel_l1"]) > 1e-4, f"turning {chosen} off changes the error: {off['rel_l1']:.4%}"
            )
            if lamps:
                ineffective = client.compute({"scene": "day", "effects": {lamps[0]: False}})
                check(ineffective["result"]["id"] == day["result"]["id"], f"{lamps[0]} off by day gives the same frames")
            status, body = client.json("/api/prerequisites", {"scene": "day", "inputs": {"shadow_maps": False}})
            shadowed = [n for n in names if "shadows" in body["missing"][n]]
            check(
                status == 200 and "sun_shadows" in shadowed,
                f"without shadow maps, greyed for lack of shadows: {shadowed}",
            )

        quarter = client.compute({"scene": "day", "factor": 4})["result"]
        check(
            quarter["metrics"]["eval"]["whole"]["rel_l1"] != day["result"]["metrics"]["eval"]["whole"]["rel_l1"],
            f"factor 4 computes: whole {quarter['metrics']['eval']['whole']['rel_l1']:.3%}",
        )
        times("day, factor 4", quarter)
        flicker = client.compute({"scene": "day", "flicker": True})["result"].get("flicker") or {}
        temporal = (flicker.get("closed_form") or {}).get("temporal")
        check(temporal is not None and temporal > 0, f"flicker over frames {flicker.get('frames')}: {temporal}")
        clear_spp = schema["scenes"]["day"]["clear_spp"]
        if clear_spp:
            cheap = client.compute({"scene": "day", "clear": clear_spp[0]})["result"]
            check(
                cheap["cycles"]["pipeline"]["clear"] == clear_spp[0], f"frames without fog at {clear_spp[0]} spp compute"
            )

        status, body = client.json("/api/compute", {"scene": "day", "factor": 3})
        check(status == 400 and "factor" in body["error"], f"a bad factor is refused: {body.get('error')}")
        status, body = client.json("/api/compute", {"scene": "dusk"})
        check(status == 400, f"an unknown scene is refused: {body.get('error')}")
        malformed: list[tuple[str, dict, dict | None]] = [
            ("effects as a list", {"scene": "day", "effects": []}, None),
            ("an input as a string", {"scene": "day", "inputs": {"normals": "false"}}, None),
            ("a scene as a list", {"scene": ["day"]}, None),
            ("a factor as a boolean", {"scene": "day", "factor": True}, None),
            ("flicker as a string", {"scene": "day", "flicker": "no"}, None),
            ("a Content-Length that isn't a number", {"scene": "day"}, {"Content-Length": "abc"}),
            ("a negative Content-Length", {"scene": "day"}, {"Content-Length": "-1"}),
        ]
        if names:
            malformed.append(("a switch as a string", {"scene": "day", "effects": {names[0]: "false"}}, None))
        for label, config, extra in malformed:
            try:
                status, body = client.json("/api/compute", config, extra)
            except (urllib.error.URLError, ConnectionError, ValueError) as error:
                status, body = 0, {"error": repr(error)}
            check(status == 400, f"{label} is refused: {status} {body.get('error')}")
        status, body = client.json("/api/static?scene=night&frame=255&name=cycles&spp=8")
        check(status == 404, f"a missing render is reported: {body.get('error')}")
        status, body = client.json("/api/ablation?scene=day&factor=1&device=cpu")
        check(status == 200, f"ablations: {len(body['rows'])} rows from {body['files']}")
        listed = {r["switch"] for r in body["rows"]}
        check(set(names) <= listed, f"every switch is in the table, measured or ineffective: {set(names) - listed}")
        models: dict[str, list[str]] = {}
        for row in body["rows"]:
            models.setdefault(str(row["source"].get("model")), []).append(row["switch"])
        check(body.get("checked") is True, f"ablations are checked against the library's model once warm: {models}")

        if "albedo" in schema["inputs"]:
            check(not schema["inputs"]["albedo"]["default"], "the albedo of each pixel is an input, off by default")
            given = client.compute({"scene": "day", "inputs": {"albedo": True}})
            with_albedo = given["result"]["metrics"]["eval"]
            on = day["result"]["metrics"]["eval"]
            check(
                abs(with_albedo["surfaces"]["rel_l1"] - on["surfaces"]["rel_l1"]) > 1e-5,
                f"with the albedo of each pixel, surfaces' error changes: {on['surfaces']['rel_l1']:.4%} -> "
                f"{with_albedo['surfaces']['rel_l1']:.4%} (whole {with_albedo['whole']['rel_l1']:.4%})",
            )
            missing = given["prerequisites"]["missing"]
            check(
                all("albedo" not in (missing.get(n) or []) for n in names),
                "no switch misses the albedo when it is given",
            )
            status, raw, _ = client.call(f"/api/static?scene=day&frame=255&name=albedo&v={schema['instance']}")
            check(status == 200 and len(raw) == width * height * 3 * 2, "the albedo of each pixel is served")
        else:
            print(f"note this library's apply_medium doesn't take the albedo of each pixel: {list(schema['inputs'])}")

        # A request answered from memory still drops what the page asked for before: here a configuration queued
        # behind the night's warm-up, or running, which then stops before its next frame
        uncached = {"scene": "day", "factor": 2, "inputs": {"normals": False}, "supersede": True}
        client.json("/api/compute", uncached)
        client.json("/api/compute", {"scene": "day", "supersede": True})
        status, body = client.json("/api/status")
        queued = [j for j in body["queue"] if j["origin"] == "page"]
        check(not queued, f"a request answered from memory drops the page's earlier ones still queued: {queued}")

        night = client.compute({"scene": "night"}, timeout=1800)
        same_errors("night", night["result"])
        times("night, default", night["result"])
        # Regions are measured as eval_closed_form.py measures the fog in and out of the cube's lamp shadow
        regions = night["result"]["metrics"]["eval"]["regions"]
        for region, name in (("lamp_shadow", "fog_shadow_pixels"), ("lamp_lit", "fog_lit_pixels")):
            for metric in METRICS:
                ours, theirs = regions[region]["fog"][metric], expected["night"][name][metric]
                check(
                    math.isclose(ours, theirs, rel_tol=args.tolerance, abs_tol=args.tolerance),
                    f"night fog in region {region} {metric}: viewer {ours:.6%} vs eval_closed_form {name} {theirs:.6%}",
                )
        # A second run of the server, as when it is restarted with another library: the images of the same
        # configuration get other names, so that the browser can't show those of the first run, with the same numbers
        if not args.skip_restart:
            port2 = free_port()
            second = subprocess.Popen(
                [*command[:3], str(port2), *command[4:], "--no-warmup"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            try:
                client2 = Client(port2)
                start = time.time()
                while True:
                    if second.poll() is not None:
                        raise RuntimeError(f"The second server stopped with code {second.returncode}, see --log")
                    try:
                        status, body = client2.json("/api/status")
                        if status == 200 and not body["loading"]:
                            break
                    except (urllib.error.URLError, ConnectionError):
                        pass
                    if time.time() - start > 300:
                        raise TimeoutError("The second server didn't load its scenes in 5 min")
                    time.sleep(1)
                again = client2.compute({"scene": "day"})["result"]
                check(again["id"] != day["result"]["id"], f"a second run names images differently: {again['id']}")
                ours, theirs = again["metrics"]["eval"]["whole"], day["result"]["metrics"]["eval"]["whole"]
                check(
                    math.isclose(ours["rel_l1"], theirs["rel_l1"], rel_tol=args.tolerance, abs_tol=args.tolerance),
                    f"and gives the same numbers: {ours['rel_l1']:.6%}",
                )
                status, raw, _ = client2.call(f"/api/image?id={day['result']['id']}&name=radiance")
                check(status == 404, "and doesn't know the names of the first run's images")
            finally:
                second.terminate()
                try:
                    second.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    second.kill()
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        print(f"FAIL {type(error).__name__}: {error}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        if not isinstance(log, int):
            log.close()
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"], help="device of the server")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads, here and in the server")
    parser.add_argument("--expected", help="JSON of eval_closed_form.py with the day and night cases, else evaluated")
    parser.add_argument("--tolerance", type=float, default=2e-5, help="relative and absolute tolerance of errors")
    parser.add_argument("--root", help="checkout with runs/fog-comparison and the study's scripts, found by default")
    parser.add_argument("--library", help="visionsim library to test, as the server's --library")
    parser.add_argument("--gpu-lock", help="passed to the server: folder created while it computes on the GPU")
    parser.add_argument("--log", help="file to write the server's output to")
    parser.add_argument("--skip-restart", action="store_true", help="don't check a second run of the server")
    sys.exit(main(parser.parse_args()))
