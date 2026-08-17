"""Interactive Textual wizard for installing and serving an LLM with vLLM."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

from sparkctl import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_PORT, DEFAULT_SERVED_NAME, MODELS_DIR
from sparkctl.hardware import HardwareReport, probe
from sparkctl.hf_models import (
    download_model,
    load_hf_token,
    model_dir_for,
    save_hf_token,
    search_models,
    validate_local_model,
)
from sparkctl.recipes import CATALOG, default_recipe, find_recipe
from sparkctl.serve import (
    ServeConfig,
    apply_recipe_defaults,
    curl_example,
    install_systemd,
    ngc_login,
    public_endpoints,
    pull_image,
    save_config,
    smoke_test,
    start_container,
    wait_healthy,
)


@dataclass
class WizardState:
    hf_token: str = ""
    ngc_key: str = ""
    source_mode: str = "catalog"
    hf_repo: str = ""
    local_dir: str = ""
    download_first: bool = True
    served_name: str = DEFAULT_SERVED_NAME
    port: int = DEFAULT_PORT
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: str = ""
    start_on_boot: bool = True
    extra_args: list[str] = field(default_factory=list)
    trust_remote_code: bool = True
    recipe_name: str = ""

    def to_serve_config(self) -> ServeConfig:
        max_len = None
        if self.max_model_len.strip():
            max_len = int(self.max_model_len.strip())
        source = "local" if self.source_mode == "local" else "huggingface"
        model_dir = None
        if source == "local":
            model_dir = str(Path(self.local_dir).expanduser().resolve())
        elif self.download_first and self.hf_repo:
            model_dir = str(model_dir_for(self.hf_repo))
        cfg = ServeConfig(
            source=source,
            hf_repo=self.hf_repo or None,
            model_dir=model_dir,
            served_name=self.served_name.strip() or DEFAULT_SERVED_NAME,
            port=self.port,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=max_len,
            start_on_boot=self.start_on_boot,
            trust_remote_code=self.trust_remote_code,
            extra_args=list(self.extra_args),
            hf_token_set=bool(self.hf_token),
        )
        return apply_recipe_defaults(cfg)


def _mark(ok: bool, warn: bool = False) -> str:
    if ok and not warn:
        return "PASS"
    if ok:
        return "WARN"
    return "FAIL"


class WelcomeScreen(Screen):
    BINDINGS = [Binding("enter", "start", "Start", show=True)]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="hero"):
            yield Static("DGX Spark LLM Installer", id="title")
            yield Static(
                "Install vLLM (NVIDIA NGC) and serve a local OpenAI-compatible API.\n"
                "Pick a Hugging Face model or a checkpoint already on this machine.",
                id="subtitle",
            )
            yield Static(
                "This wizard will:\n"
                "  1. Check GB10 / Docker / CUDA\n"
                "  2. Pull a Blackwell-ready vLLM image\n"
                "  3. Download or attach your LLM\n"
                "  4. Start http://0.0.0.0:8000  (/v1/chat/completions)",
                id="blurb",
            )
            yield Button("Start", id="start", variant="success")
        yield Footer()

    def action_start(self) -> None:
        self.app.push_screen(HardwareScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()


class HardwareScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="body"):
            yield Static("Step 1 / 5  —  Hardware check", classes="step")
            table = DataTable(id="hw-table", cursor_type="row")
            table.add_columns("Status", "Check", "Detail")
            yield table
            yield Static("Scanning…", id="hw-summary")
            with Horizontal(classes="actions"):
                yield Button("Re-scan", id="rescan")
                yield Button("Continue", id="continue", variant="success", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_probe()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rescan":
            self.run_probe()
        elif event.button.id == "continue":
            self.app.push_screen(AuthScreen())

    @work(thread=True, exclusive=True)
    def run_probe(self) -> None:
        self.app.call_from_thread(self._set_busy, True)
        report = probe()
        self.app.call_from_thread(self._show_report, report)

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#continue", Button).disabled = True
        self.query_one("#hw-summary", Static).update("Scanning…" if busy else "")

    def _show_report(self, report: HardwareReport) -> None:
        table = self.query_one("#hw-table", DataTable)
        table.clear()
        for check in report.checks:
            table.add_row(_mark(check.ok, check.warn), check.name, check.detail)
        if report.ready:
            extra = "GB10 Spark detected." if report.is_gb10 else "Not a GB10, but required checks passed."
            self.query_one("#hw-summary", Static).update(f"Ready. {extra}")
            self.query_one("#continue", Button).disabled = False
        else:
            failed = ", ".join(c.name for c in report.required_failed)
            self.query_one("#hw-summary", Static).update(
                f"Required checks failed: {failed}. Fix those, then Re-scan."
            )
            self.query_one("#continue", Button).disabled = True


class AuthScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        existing = load_hf_token()
        hint = "A token is already saved on this machine." if existing else "Optional unless the model is gated."
        yield Header(show_clock=True)
        with VerticalScroll(id="body"):
            yield Static("Step 2 / 5  —  Tokens", classes="step")
            yield Static(
                "Hugging Face token: required for gated models (Llama, some NVIDIA checkpoints).\n"
                "Create one at https://huggingface.co/settings/tokens\n"
                f"{hint}",
                classes="help",
            )
            yield Label("Hugging Face token")
            yield Input(
                value=existing or "",
                password=True,
                placeholder="hf_…  (leave blank to skip)",
                id="hf-token",
            )
            yield Static(
                "NGC API key: only if docker pull nvcr.io/nvidia/vllm fails with 401.\n"
                "https://org.ngc.nvidia.com/setup/api-key  — username is $oauthtoken",
                classes="help",
            )
            yield Label("NVIDIA NGC API key")
            yield Input(password=True, placeholder="optional", id="ngc-key")
            with Horizontal(classes="actions"):
                yield Button("Skip", id="skip")
                yield Button("Continue", id="continue", variant="success")
        yield Footer()

    def _save_into_state(self) -> None:
        state: WizardState = self.app.state
        state.hf_token = self.query_one("#hf-token", Input).value.strip()
        state.ngc_key = self.query_one("#ngc-key", Input).value.strip()
        if state.hf_token:
            try:
                save_hf_token(state.hf_token)
                os.environ["HF_TOKEN"] = state.hf_token
            except Exception as exc:  # noqa: BLE001 — show in UI
                self.notify(f"Could not save HF token: {exc}", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._save_into_state()
        if event.button.id in {"skip", "continue"}:
            self.app.push_screen(ModelScreen())


class ModelScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="body"):
            yield Static("Step 3 / 5  —  Choose a model", classes="step")
            yield Static("Where should the weights come from?", classes="help")
            with RadioSet(id="source"):
                yield RadioButton("Spark-validated catalog", id="src-catalog", value=True)
                yield RadioButton("Search Hugging Face", id="src-search")
                yield RadioButton("Paste Hugging Face repo ID", id="src-paste")
                yield RadioButton("Local folder on this machine", id="src-local")

            with Container(id="panel-catalog", classes="panel"):
                catalog = DataTable(id="catalog", cursor_type="row")
                catalog.add_columns("Model", "Quant", "Hugging Face ID")
                yield catalog
                yield Static("", id="catalog-notes")

            with Container(id="panel-search", classes="panel hidden"):
                with Horizontal():
                    yield Input(placeholder="e.g. Qwen3 8B instruct", id="search-q")
                    yield Button("Search", id="search-btn", variant="primary")
                results = DataTable(id="search-results", cursor_type="row")
                results.add_columns("Repo", "Pipeline", "Downloads")
                yield results

            with Container(id="panel-paste", classes="panel hidden"):
                yield Label("Hugging Face repo (org/name)")
                yield Input(placeholder="nvidia/Qwen3-8B-NVFP4", id="paste-id")

            with Container(id="panel-local", classes="panel hidden"):
                yield Static(
                    "Point at a folder that already has config.json and weight files "
                    "(copy it onto the Spark with scp/USB first).",
                    classes="help",
                )
                yield Label("Model directory")
                yield Input(placeholder=str(Path.home() / "models" / "my-model"), id="local-path")
                yield DirectoryTree(str(Path.home()), id="tree")
                yield Static("", id="local-status")

            yield Static("No model selected yet.", id="selection")
            with Horizontal(classes="actions"):
                yield Button("Continue", id="continue", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#catalog", DataTable)
        for recipe in CATALOG:
            label = recipe.name + ("  *" if recipe.recommended else "")
            table.add_row(label, recipe.quantization, recipe.repo_id, key=recipe.repo_id)
        rec = default_recipe()
        self._select_hf(rec.repo_id, recipe_name=rec.name)

    def _show_panel(self, name: str) -> None:
        for panel in ("catalog", "search", "paste", "local"):
            widget = self.query_one(f"#panel-{panel}")
            widget.set_class(panel != name, "hidden")

    @on(RadioSet.Changed, "#source")
    def _source_changed(self, event: RadioSet.Changed) -> None:
        mapping = {
            "src-catalog": "catalog",
            "src-search": "search",
            "src-paste": "paste",
            "src-local": "local",
        }
        mode = mapping.get(event.pressed.id, "catalog")
        self.app.state.source_mode = mode
        self._show_panel(mode)

    @staticmethod
    def _row_key(event: DataTable.RowSelected) -> str:
        key = event.row_key
        if key is None:
            return ""
        return str(getattr(key, "value", key))

    def _select_hf(self, repo_id: str, recipe_name: str = "") -> None:
        state: WizardState = self.app.state
        state.hf_repo = repo_id.strip()
        state.local_dir = ""
        state.source_mode = state.source_mode if state.source_mode != "local" else "catalog"
        recipe = find_recipe(state.hf_repo)
        state.recipe_name = recipe_name or (recipe.name if recipe else "")
        state.trust_remote_code = recipe.trust_remote_code if recipe else True
        if recipe and recipe.extra_args:
            rest: list[str] = []
            args = list(recipe.extra_args)
            i = 0
            while i < len(args):
                if args[i] == "--gpu-memory-utilization" and i + 1 < len(args):
                    try:
                        state.gpu_memory_utilization = float(args[i + 1])
                    except ValueError:
                        pass
                    i += 2
                    continue
                rest.append(args[i])
                i += 1
            state.extra_args = rest
        else:
            state.extra_args = []
        note = recipe.notes if recipe else ""
        self.query_one("#selection", Static).update(f"Selected: {state.hf_repo}")
        notes = self.query_one("#catalog-notes", Static)
        notes.update(note)

    def _select_local(self, path: str) -> None:
        result = validate_local_model(path)
        status = self.query_one("#local-status", Static)
        status.update(result.detail)
        if result.ok:
            state: WizardState = self.app.state
            state.source_mode = "local"
            state.local_dir = str(result.path)
            state.hf_repo = ""
            state.recipe_name = result.path.name
            self.query_one("#selection", Static).update(f"Selected local: {result.path}")
            self.query_one("#local-path", Input).value = str(result.path)

    @on(DataTable.RowSelected, "#catalog")
    def _catalog_row(self, event: DataTable.RowSelected) -> None:
        repo_id = self._row_key(event)
        if repo_id:
            self._select_hf(repo_id)

    @on(DataTable.RowSelected, "#search-results")
    def _search_row(self, event: DataTable.RowSelected) -> None:
        repo_id = self._row_key(event)
        if repo_id:
            self.app.state.source_mode = "search"
            self._select_hf(repo_id)

    @on(Input.Submitted, "#paste-id")
    def _paste_submit(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.app.state.source_mode = "paste"
            self._select_hf(event.value.strip())

    @on(Input.Submitted, "#local-path")
    def _local_submit(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self._select_local(event.value.strip())

    @on(Input.Changed, "#paste-id")
    def _paste_changed(self, event: Input.Changed) -> None:
        if event.value.strip() and "/" in event.value:
            self.app.state.source_mode = "paste"
            self._select_hf(event.value.strip())

    def on_directory_tree_directory_selected(self, event) -> None:
        path = getattr(event, "path", None)
        if path:
            self._select_local(str(path))

    def on_directory_tree_file_selected(self, event) -> None:
        path = getattr(event, "path", None)
        if path:
            self._select_local(str(path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search-btn":
            query = self.query_one("#search-q", Input).value.strip()
            if query:
                self._search(query)
            return
        if event.button.id == "continue":
            local_input = self.query_one("#local-path", Input).value.strip()
            paste = self.query_one("#paste-id", Input).value.strip()
            if self.app.state.source_mode == "local" or local_input:
                if local_input:
                    self._select_local(local_input)
            elif paste and not self.app.state.hf_repo:
                self._select_hf(paste)
            if self.app.state.source_mode == "local":
                if not self.app.state.local_dir:
                    self.notify("Pick a valid local model folder first.", severity="error")
                    return
            elif not self.app.state.hf_repo:
                self.notify("Select or paste a Hugging Face repo first.", severity="error")
                return
            self.app.push_screen(SettingsScreen())

    @work(thread=True, exclusive=True)
    def _search(self, query: str) -> None:
        token = self.app.state.hf_token or load_hf_token()
        try:
            hits = search_models(query, token=token)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._notify_error, f"Search failed: {exc}")
            return
        self.app.call_from_thread(self._fill_search, hits)

    def _notify_error(self, message: str) -> None:
        self.notify(message, severity="error")

    def _fill_search(self, hits) -> None:
        table = self.query_one("#search-results", DataTable)
        table.clear()
        if not hits:
            self.notify("No models found.", severity="warning")
            return
        for hit in hits:
            downloads = f"{hit.downloads:,}" if hit.downloads else "—"
            table.add_row(hit.repo_id, hit.pipeline or "—", downloads, key=hit.repo_id)


class SettingsScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        state: WizardState = self.app.state
        if state.hf_repo:
            suggested = state.hf_repo.split("/")[-1]
        elif state.local_dir:
            suggested = Path(state.local_dir).name
        else:
            suggested = DEFAULT_SERVED_NAME
        yield Header(show_clock=True)
        with VerticalScroll(id="body"):
            yield Static("Step 4 / 5  —  Serve settings", classes="step")
            yield Label("Served model name (this is what clients send as `model`)")
            yield Input(value=suggested.replace(" ", "-").lower()[:64] or DEFAULT_SERVED_NAME, id="served")
            yield Label("Port")
            yield Input(value=str(DEFAULT_PORT), id="port", type="integer")
            yield Label("GPU memory utilization (0.10 – 0.95)")
            yield Input(value=str(state.gpu_memory_utilization), id="gpu-util")
            yield Label("Max model length (blank = vLLM default)")
            yield Input(placeholder="e.g. 8192", id="max-len")
            yield Checkbox("Download Hugging Face weights before starting vLLM", value=True, id="download-first")
            yield Checkbox("Start on boot (Docker restart + systemd if available)", value=True, id="boot")
            with Horizontal(classes="actions"):
                yield Button("Install and serve", id="go", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        if self.app.state.source_mode == "local":
            box = self.query_one("#download-first", Checkbox)
            box.value = False
            box.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "go":
            return
        state: WizardState = self.app.state
        state.served_name = self.query_one("#served", Input).value.strip() or DEFAULT_SERVED_NAME
        try:
            state.port = int(self.query_one("#port", Input).value.strip() or DEFAULT_PORT)
        except ValueError:
            self.notify("Port must be an integer.", severity="error")
            return
        try:
            util = float(self.query_one("#gpu-util", Input).value.strip() or DEFAULT_GPU_MEMORY_UTILIZATION)
        except ValueError:
            self.notify("GPU memory utilization must be a number.", severity="error")
            return
        if not 0.1 <= util <= 0.95:
            self.notify("GPU memory utilization should be between 0.10 and 0.95.", severity="error")
            return
        state.gpu_memory_utilization = util
        state.max_model_len = self.query_one("#max-len", Input).value.strip()
        state.download_first = self.query_one("#download-first", Checkbox).value
        state.start_on_boot = self.query_one("#boot", Checkbox).value
        self.app.push_screen(InstallScreen())


class InstallScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            yield Static("Step 5 / 5  —  Download, pull vLLM, start endpoint", classes="step")
            yield ProgressBar(id="progress")
            yield RichLog(id="log", highlight=False, markup=True)
            with Horizontal(classes="actions"):
                yield Button("Retry", id="retry", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#progress", ProgressBar).add_class("running")
        self.run_install()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "retry":
            self.query_one("#retry", Button).disabled = True
            self.query_one("#log", RichLog).clear()
            self.run_install()

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def _fail(self, message: str) -> None:
        self._log(f"[red]{message}[/red]")
        self.query_one("#retry", Button).disabled = False
        self.notify("Install failed. See the log.", severity="error")

    def _succeed(self, cfg: ServeConfig, smoke: str) -> None:
        self.app.final_config = cfg
        self.app.smoke_text = smoke
        self.app.push_screen(DoneScreen())

    @work(thread=True, exclusive=True)
    def run_install(self) -> None:
        def log(msg: str) -> None:
            self.app.call_from_thread(self._log, msg)

        state: WizardState = self.app.state
        try:
            if state.hf_token:
                os.environ["HF_TOKEN"] = state.hf_token
            if state.ngc_key:
                ngc_login(state.ngc_key, log)

            cfg = state.to_serve_config()
            pull_image(cfg, log)

            if state.source_mode != "local" and state.download_first and state.hf_repo:
                dest = model_dir_for(state.hf_repo)
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                download_model(state.hf_repo, dest, token=state.hf_token or load_hf_token(), on_progress=log)
                checked = validate_local_model(dest)
                if not checked.ok:
                    raise RuntimeError(f"Downloaded files look incomplete: {checked.detail}")
                cfg.model_dir = str(checked.path)
                cfg.source = "huggingface"

            save_config(cfg)
            start_container(cfg, log)
            wait_healthy(cfg, log)
            install_systemd(cfg, log)
            ok, text = smoke_test(cfg)
            if ok:
                log(f"Smoke test OK: {text[:200]}")
            else:
                log(f"Server is up, but smoke test failed: {text}")
            self.app.call_from_thread(self._succeed, cfg, text if ok else "")
        except Exception as exc:  # noqa: BLE001 — surface in the wizard
            self.app.call_from_thread(self._fail, str(exc))


class DoneScreen(Screen):
    BINDINGS = [Binding("q", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        cfg: ServeConfig = self.app.final_config
        urls = public_endpoints(cfg)
        url_lines = "\n".join(f"  {u}" for u in urls)
        smoke = getattr(self.app, "smoke_text", "") or ""
        yield Header(show_clock=True)
        with VerticalScroll(id="body"):
            yield Static("Ready  —  vLLM is serving", id="title")
            yield Static(
                f"OpenAI-compatible endpoint:\n{url_lines}\n\n"
                f"Health:  {cfg.endpoint()}/health\n"
                f"Models:  {cfg.endpoint()}/v1/models\n"
                f"Chat:    {cfg.endpoint()}/v1/chat/completions\n"
                f"Served name: {cfg.served_name}",
                id="blurb",
            )
            if smoke:
                yield Static(f"Smoke test reply: {smoke}", classes="help")
            yield Static("Try it:", classes="step")
            yield Static(curl_example(cfg), id="curl")
            yield Static(
                "Later:\n"
                "  sparkctl status\n"
                "  sparkctl logs\n"
                "  sparkctl test\n"
                "  sparkctl stop\n"
                "  sparkctl wizard    (run this installer again)",
                classes="help",
            )
            yield Button("Quit", id="quit", variant="success")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()


class WizardApp(App):
    TITLE = "DGX Spark LLM Installer"
    CSS = """
    Screen {
        background: #111111;
        color: #f2f2f2;
    }
    Header {
        background: #76B900;
        color: #111111;
        text-style: bold;
    }
    Footer {
        background: #1b1b1b;
    }
    #hero, #body {
        padding: 1 2;
    }
    #title {
        text-style: bold;
        color: #76B900;
        content-align: left middle;
        height: auto;
        margin-bottom: 1;
    }
    #subtitle, #blurb, .help {
        color: #c8c8c8;
        height: auto;
        margin-bottom: 1;
    }
    .step {
        text-style: bold;
        color: #76B900;
        height: auto;
        margin-bottom: 1;
    }
    .actions {
        height: auto;
        margin-top: 1;
        align: left middle;
    }
    Button {
        margin-right: 1;
    }
    Input {
        margin-bottom: 1;
        width: 100%;
    }
    Label {
        height: auto;
        margin-top: 1;
    }
    DataTable {
        height: 14;
        margin-bottom: 1;
    }
    DirectoryTree {
        height: 12;
        margin: 1 0;
        border: solid #333333;
    }
    #log {
        height: 1fr;
        border: solid #333333;
        background: #0b0b0b;
    }
    #progress {
        margin: 1 0;
    }
    #curl {
        background: #0b0b0b;
        border: solid #333333;
        padding: 1;
        height: auto;
        color: #d4ff8f;
    }
    #hw-summary, #selection, #local-status, #catalog-notes {
        height: auto;
        color: #d0d0d0;
        margin: 1 0;
    }
    .hidden {
        display: none;
    }
    .panel {
        height: auto;
        margin-bottom: 1;
    }
    RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = WizardState()
        token = load_hf_token()
        if token:
            self.state.hf_token = token
        self.final_config: ServeConfig | None = None
        self.smoke_text = ""

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


def run_wizard() -> None:
    WizardApp().run()
