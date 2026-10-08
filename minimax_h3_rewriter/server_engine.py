"""One multimodal model, held open across a run's references.

``mtmd_engine`` starts ``llama-mtmd-cli`` once per description, which is right
for one picture and wasteful for six: the loop in ``multi_caption`` and
``universal`` calls it per asset, and each call reloads the model and the
projector from scratch. Measured on an 8B captioner with a warm file cache,
that is about three seconds an asset before a pixel is read; on a cold disk, or
one of the larger captioners, it is considerably worse.

``llama-server`` is the same library behind an HTTP endpoint, and it is already
on disk -- the release archive puts it beside ``llama-mtmd-cli``, so nothing is
downloaded for this. It is started before the loop, asked once per asset, and
killed after. Same model, same projector, same sampling, same answers; the
loading happens once.

Three things shape the code:

- **It is an optimisation, never a requirement.** Every failure here -- no
  binary in this build, a port that will not bind, a server that never reports
  healthy -- returns rather than raises, and the caller goes back to starting a
  process per asset. A caption run must not fail because a speed-up did.
- **The wire format is not llama.cpp's own.** Attachments go through the
  OpenAI-compatible chat endpoint as content parts, base64 in a data URI for a
  picture and an ``input_audio`` part for a sound, because that is the only
  route the server offers to the projector. The files on disk are the same ones
  the command line would have named.
- **The child is tied to this process by the kernel.** A one-shot binary that
  outlives a crash is gone in seconds anyway; a server holding a model on the
  card is not. ``runner.spawn`` puts it in a job object that dies with
  ComfyUI -- see ``runner._adopt``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request

from . import devices, llamacpp, runner

log = logging.getLogger(__name__)

HOST = "127.0.0.1"

# A ``.ninfer`` artifact (NInfer v3) carries its own vision tower, MTP head and
# DFlash2 drafter, and only runs behind ninfer-serve: there is no one-shot
# binary to fall back to. ``ninfer_bin.txt`` beside ``llama_bin.txt`` names the
# folder holding ninfer-serve.exe.
NINFER_SUFFIX = ".ninfer"
NINFER_BIN_FILE = "ninfer_bin.txt"
NINFER_SERVER = "ninfer-serve.exe" if os.name == "nt" else "ninfer-serve"
NINFER_CONTEXT = 32768

STARTUP_SECONDS = 900.0
HEALTH_INTERVAL = 0.4

REQUEST_SECONDS = 900.0

STDERR_KEEP = 80

MEDIA_TYPE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}

CHOICE_ENV = "MINIMAX_H3_MTMD_SERVER"
AUTO, NEVER, ALWAYS = "auto", "never", "always"


class ServerUnavailable(RuntimeError):
    """The server could not be started. Never fatal: the binary still works."""


def wanted(assets: int) -> bool:
    """Whether a run of this many assets should hold a server open.

    One asset is the case the server loses: it pays a process, a port and a
    handshake to save a load it was going to do exactly once anyway. Two is
    already ahead. The variable is here because a machine where this misbehaves
    needs a way back that does not involve editing the pack.
    """
    choice = (os.environ.get(CHOICE_ENV) or AUTO).strip().lower()
    if choice == NEVER:
        return False
    if choice == ALWAYS:
        return True
    if choice != AUTO:
        log.warning(
            "[minimax_h3_rewriter.server_engine.wanted] %s is '%s', which is none of "
            "%s/%s/%s -- treating it as %s",
            CHOICE_ENV, choice, AUTO, NEVER, ALWAYS, AUTO,
        )
    return assets > 1


def free_port() -> int:
    """A port nothing is listening on, as of a moment ago.

    There is no way to reserve one for a child, so this is a race by
    construction: the port is released here and claimed by the server a moment
    later. ``start`` treats a bind failure as one more reason to fall back
    rather than as an error, which is the honest handling of a race that can be
    narrowed and not closed.
    """
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _data_uri(path: str) -> str:
    kind = MEDIA_TYPE.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as handle:
        return f"data:{kind};base64," + base64.b64encode(handle.read()).decode("ascii")


def content_parts(instruction: str, attachments: list[tuple[str, str]]) -> list[dict]:
    """The user turn, media first, exactly as the command line orders it.

    Order carries meaning here: the frames of a clip are chronological, and a
    first-and-last pair is told apart by which came first. ``--image a --image b
    --prompt ...`` puts the media ahead of the instruction, so this does too.
    """
    parts: list[dict] = []
    for kind, path in attachments:
        if kind == "image":
            parts.append({"type": "image_url", "image_url": {"url": _data_uri(path)}})
        elif kind == "audio":
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            parts.append({
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            })
        else:
            raise ValueError(f"unknown attachment kind '{kind}'")
    parts.append({"type": "text", "text": instruction})
    return parts


def build_command(
    binary: str,
    model_path: str,
    mmproj_path: str,
    port: int,
    gpu_layers: int,
    n_ctx: int,
    device: str = devices.AUTO,
    adapter_path: str | None = None,
    slots: int = 1,
    reasoning: dict | None = None,
) -> list[str]:
    """The same flags ``mtmd_engine.build_command`` uses, minus the one-shot ones.

    Sampling, the seed and the token ceiling are absent on purpose: on the
    command line they are properties of the single run, and here they are
    properties of each request, so they travel in the body instead. Everything
    that describes the *model* is still decided once, here.
    """
    layers = devices.layers_for(device, gpu_layers)
    layers = 999 if layers < 0 else layers
    command = [binary, "--model", model_path]
    if mmproj_path:
        command += ["--mmproj", mmproj_path]
    command += [
        *devices.llama_arguments(device),
        "--n-gpu-layers", str(layers),
        "--ctx-size", str(int(n_ctx)),
        "--host", HOST,
        "--port", str(int(port)),
        "--no-webui",
    ]
    if adapter_path:
        command += ["--lora", adapter_path]
    if int(slots) > 1:
        # Several references described at once: one slot each, drawing on a
        # single shared KV pool so a request may take what the others leave.
        command += ["--parallel", str(int(slots)), "--kv-unified"]
    if reasoning is not None:
        # The writer runs on this server too. llama.cpp applies the chat
        # template itself here (--jinja), which is what lets it open or close
        # the model's thinking block per request and enforce a token budget on
        # it. Thoughts are returned on a separate channel (reasoning_content),
        # so the answer text stays clean.
        command += [
            "--jinja",
            "--reasoning", "on" if reasoning.get("enabled") else "off",
            "--reasoning-format", "deepseek",
            "--reasoning-budget", str(int(reasoning.get("budget", -1))),
        ]
        message = str(reasoning.get("message") or "").strip()
        if message:
            command += ["--reasoning-budget-message", message]
    if int(slots) > 1 or reasoning is not None:
        # Long guide + several slots: an 8-bit KV cache halves what the pool costs.
        command += ["--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    return command


def is_ninfer(model_path: str) -> bool:
    return bool(model_path) and model_path.lower().endswith(NINFER_SUFFIX)


def ninfer_binary() -> str:
    """ninfer-serve from the folder ``ninfer_bin.txt`` names, or "" without one."""
    path = os.path.join(os.path.dirname(llamacpp.bin_file()), NINFER_BIN_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            folder = next((line.strip() for line in handle if line.strip() and not line.startswith("#")), "")
    except OSError:
        return ""
    binary = os.path.join(folder, NINFER_SERVER)
    return binary if os.path.isfile(binary) else ""


def build_ninfer_command(
    binary: str,
    model_path: str,
    port: int,
    n_ctx: int,
    slots: int = 1,
    reasoning: dict | None = None,
) -> list[str]:
    """ninfer-serve with the decode settings of the ninfer-custom fork's README.

    ``n_ctx`` is the per-request ceiling; the shared KV pool is sized by the
    engine from what the card has left (``--kv-capacity auto``). Thinking stays
    on by default and each request says whether to use it, as captions do.
    """
    command = [
        binary, model_path,
        "--host", HOST,
        "--port", str(int(port)),
        "--max-context", str(int(n_ctx)),
        "--max-concurrency", str(max(1, int(slots))),
        "--kv-capacity", "auto",
        "--kv-dtype", "int8",
        "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
        "--ngram-draft-tokens", "15", "--ngram-min-match", "12",
        # The server lives for one run: a pinned host tier (8 GiB by default)
        # would only take RAM from ComfyUI's offloaded models.
        "--host-cache-mib", "0",
        "--vision",
        "--log-stats-panel", "off",
    ]
    if reasoning is not None and reasoning.get("enabled"):
        budget = int(reasoning.get("budget", -1))
        message = str(reasoning.get("message") or "").strip()
        if budget > 0:
            command += ["--default-thinking-budget", str(budget)]
            if message:
                command += ["--thinking-budget-message", message]
    return command


def request_body(
    instruction: str,
    attachments: list[tuple[str, str]],
    seed: int = 42,
    greedy: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
    messages: list[dict] | None = None,
    repeat_penalty: float | None = None,
) -> dict:
    """The chat request, as the command line's flags would have spelled it.

    ``messages`` replaces the instruction-and-attachments turn with a chat
    already written out (the writer's guide and task), and ``enable_thinking``
    is forwarded to the chat template so a thinking model can be told per
    request whether to deliberate first.

    Apart on purpose: this is the half of the two paths that has to agree with
    the other, and the only way to check that it does without a model resident
    is to be able to look at it. See ``mtmd_engine.DEFAULT_SYSTEM`` for the
    part of the agreement that had to be found out the hard way.
    """
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_parts(instruction, attachments)})

    body = {
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "seed": int(seed),
        "stream": True,
    }
    if greedy:
        body["temperature"] = 0.0
    else:
        body.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
    if repeat_penalty is not None:
        body["repeat_penalty"] = float(repeat_penalty)
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    return body


class Server:
    """A resident ``llama-server``, asked once per reference."""

    def __init__(self, binary: str, command: list[str], port: int, n_ctx: int = 0):
        self.binary = binary
        self.command = command
        self.port = port
        self.n_ctx = int(n_ctx)
        self.slots = 1
        # True when the command line carried --jinja and the reasoning flags: the
        # server then honours chat_template_kwargs, and a caption has to say
        # enable_thinking=false or the model spends its short budget deliberating.
        self.thinking_aware = False
        # ninfer-serve has no repetition-penalty sampler (it refuses anything but
        # 1.0) and refuses a request that does not name its one model.
        self.repeat_penalty = True
        self.model_id = ""
        self.process = None
        self._stderr: list[str] = []
        self._lock = threading.Lock()

    @property
    def base(self) -> str:
        return f"http://{HOST}:{self.port}"

    def _watch(self, stream) -> None:
        """Keep the last of the child's stderr, and keep its pipe from filling.

        Both halves matter. A pipe nobody reads fills at 64 KB and blocks the
        writer, which would wedge the server mid-load; and when a start fails,
        llama.cpp's own last words are the only useful thing to put in the log.
        """
        try:
            for line in iter(stream.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                with self._lock:
                    self._stderr.append(text)
                    del self._stderr[:-STDERR_KEEP]
        except Exception:
            log.debug("[minimax_h3_rewriter.server_engine._watch] reader stopped",
                      exc_info=True)

    def tail(self, lines: int = 12) -> str:
        with self._lock:
            return "\n".join(self._stderr[-lines:])

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=2) as answer:
                return answer.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self, seconds: float = STARTUP_SECONDS, on_wait=None) -> None:
        log.info("[minimax_h3_rewriter.server_engine] %s", " ".join(self.command))
        try:
            self.process = runner.spawn(self.command, self.binary)
        except OSError as error:
            raise ServerUnavailable(f"could not start '{self.binary}': {error}") from error

        for stream in (self.process.stdout, self.process.stderr):
            threading.Thread(target=self._watch, args=(stream,), daemon=True).start()

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ServerUnavailable(
                    f"{os.path.basename(self.binary)} exited with code "
                    f"{self.process.returncode} while loading.\n{self.tail()}"
                )
            if self._healthy():
                return
            if on_wait is not None:
                on_wait(time.monotonic() - (deadline - seconds))
            time.sleep(HEALTH_INTERVAL)

        raise ServerUnavailable(
            f"{os.path.basename(self.binary)} did not answer /health within "
            f"{seconds:.0f} s.\n{self.tail()}"
        )

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
        except Exception:
            log.debug("[minimax_h3_rewriter.server_engine.close] not reaped", exc_info=True)
        finally:
            runner._LIVE.discard(process)

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    def ask(
        self,
        instruction: str,
        attachments: list[tuple[str, str]],
        seed: int = 42,
        greedy: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        system_prompt: str = "",
        on_text=None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Describe one set of attachments. Raises ``runner.ChildFailed`` on failure.

        Failures here are *not* ``ServerUnavailable``: by this point the model
        is loaded and answering, so a request that goes wrong is the run's
        problem and not a reason to start over with the binary. Falling back
        mid-loop would also mean two captioners in one strip.

        ``on_text`` follows ``runner.run``: called with the whole answer so far,
        and returning something truthy ends it and keeps what was written.
        """
        body = request_body(
            instruction, attachments, seed, greedy, max_new_tokens,
            temperature, top_p, top_k, system_prompt, enable_thinking=enable_thinking,
        )
        return self._complete(body, on_text)

    def chat(
        self,
        messages: list[dict],
        seed: int = 42,
        greedy: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float | None = None,
        enable_thinking: bool | None = None,
        on_text=None,
        on_reasoning=None,
    ) -> str:
        """One text-only chat completion on the resident model: the writer's turn.

        Same server, same wire format as :meth:`ask`, without attachments.
        ``on_reasoning`` is called with the thoughts so far while the model is
        still deliberating, so a long think shows on the node instead of a stall.
        """
        body = request_body(
            "", [], seed, greedy, max_new_tokens, temperature, top_p, top_k, "",
            enable_thinking=enable_thinking, messages=messages, repeat_penalty=repeat_penalty,
        )
        return self._complete(body, on_text, on_reasoning)

    def _complete(self, body: dict, on_text=None, on_reasoning=None) -> str:
        if not self.repeat_penalty:
            body.pop("repeat_penalty", None)
        if self.model_id:
            body["model"] = self.model_id
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        pieces: list[str] = []
        thoughts: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_SECONDS) as answer:
                for raw in answer:
                    if runner.interrupted():
                        import comfy.model_management as mm

                        raise mm.InterruptProcessingException()
                    piece, thought = _delta(raw)
                    if thought:
                        thoughts.append(thought)
                        if on_reasoning is not None:
                            on_reasoning("".join(thoughts))
                    if not piece:
                        continue
                    pieces.append(piece)
                    if on_text is not None and on_text("".join(pieces)):
                        log.info(
                            "[minimax_h3_rewriter.server] answer stopped early by the caller"
                        )
                        break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:600]
            raise runner.ChildFailed(
                f"{os.path.basename(self.binary)} refused the request "
                f"({error.code}): {detail}\n{self.tail()}"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise runner.ChildFailed(
                f"{os.path.basename(self.binary)} stopped answering: {error}\n{self.tail()}"
            ) from error

        return "".join(pieces).strip()


def _delta(raw: bytes) -> tuple[str, str]:
    """One token's worth of (answer text, reasoning text) out of an SSE line."""
    line = raw.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return "", ""
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return "", ""
    try:
        parsed = json.loads(payload)
    except ValueError:
        log.debug("[minimax_h3_rewriter.server_engine._delta] unparsed: %.120s", payload)
        return "", ""
    choices = parsed.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    return delta.get("content") or "", delta.get("reasoning_content") or ""


def open_server(
    binary: str,
    model_path: str,
    mmproj_path: str,
    gpu_layers: int,
    n_ctx: int,
    device: str = devices.AUTO,
    adapter_path: str | None = None,
    on_wait=None,
    slots: int = 1,
    reasoning: dict | None = None,
    n_ctx_total: int = 0,
) -> Server | None:
    """Start a server for this model, or return None having said why in the log.

    The one place the fallback is decided, so every caller gets the same
    behaviour: a reason written down once at INFO, and a caption run that
    carries on with ``llama-mtmd-cli``.
    """
    if not binary:
        return None
    try:
        port = free_port()
    except OSError as error:
        log.info("[minimax_h3_rewriter.server_engine] no free port (%s)", error)
        return None

    slots = max(1, int(slots))
    # ``n_ctx`` stays the per-request context (it sizes the media budget);
    # the pool the server allocates has to hold every slot at once, and the
    # writer's guide when that runs here too.
    pool = max(int(n_ctx_total or 0), int(n_ctx) * slots)
    ninfer = is_ninfer(model_path)
    if ninfer:
        command = build_ninfer_command(binary, model_path, port, -(-pool // slots), slots, reasoning)
    else:
        command = build_command(binary, model_path, mmproj_path, port, gpu_layers, pool,
                                device, adapter_path, slots=slots, reasoning=reasoning)
    server = Server(binary, command, port, n_ctx)
    server.slots = slots
    # ninfer-serve applies the chat template itself and honours enable_thinking
    # on every request, so captions must say false there too.
    server.thinking_aware = ninfer or reasoning is not None
    if ninfer:
        server.repeat_penalty = False
    try:
        server.start(on_wait=on_wait)
    except ServerUnavailable as error:
        log.info(
            "[minimax_h3_rewriter.server_engine] %s -- describing one process at a "
            "time instead", error,
        )
        server.close()
        return None
    if ninfer:
        with urllib.request.urlopen(f"{server.base}/v1/models", timeout=10) as answer:
            server.model_id = json.load(answer)["data"][0]["id"]
    log.info(
        "[minimax_h3_rewriter.server_engine] %s ready on port %d",
        os.path.basename(model_path), server.port,
    )
    return server
