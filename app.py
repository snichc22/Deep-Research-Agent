from __future__ import annotations

import sys
import time
from datetime import datetime

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import agent
from agent import (
    MODEL,
    AgentEvent,
    DoneEvent,
    FetchEvent,
    PlanEvent,
    SearchEvent,
    StatusEvent,
)
import os

# ──────────────────────────────────────────────────────────────

LOGO = (
    "  ██████╗ ██████╗  █████╗ \n"
    "  ██╔══██╗██╔══██╗██╔══██╗\n"
    "  ██║  ██║██████╔╝███████║\n"
    "  ██║  ██║██╔══██╗██╔══██║\n"
    "  ██████╔╝██║  ██║██║  ██║\n"
    "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"
)

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


# ──────────────────────────────────────────────────────────────

class ResearchPanel:
    def __init__(self, topic: str, model: str) -> None:
        self.topic = topic
        self.model = model
        self.plan: list[str] = []
        self.approach = ""
        self.events: list[tuple[str, str, str]] = []
        self.status = "Starting up..."
        self.n_search = 0
        self.n_fetch = 0
        self.t0 = time.time()
        self._tick = 0

    def _elapsed(self) -> str:
        s = int(time.time() - self.t0)
        return f"{s // 60:02d}:{s % 60:02d}"

    def _spin(self) -> str:
        ch = SPINNER_FRAMES[self._tick % len(SPINNER_FRAMES)]
        self._tick += 1
        return ch

    def push(self, icon: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append((ts, icon, msg))
        if len(self.events) > 14:
            self.events = self.events[-14:]

    def on_event(self, ev: AgentEvent) -> None:
        if isinstance(ev, PlanEvent):
            self.plan = ev.sub_questions
            self.approach = ev.approach
            self.push("📋", f"Plan set {len(ev.sub_questions)} sub-questions")

        elif isinstance(ev, SearchEvent):
            self.n_search = ev.n
            self.push("🔍", f'Search #{ev.n} - "{ev.query}"')
            self.status = f"Searching: {ev.query[:80]}"

        elif isinstance(ev, FetchEvent):
            self.n_fetch = ev.n
            short = ev.url.replace("https://", "").replace("http://", "")[:68]
            self.push("📄", f"Reading #{ev.n} - {short}")
            self.status = f"Reading: {short}"

        elif isinstance(ev, StatusEvent):
            self.status = ev.message

        elif isinstance(ev, DoneEvent):
            self.status = (
                f"✓  Complete — "
                f"{ev.n_search} searches · {ev.n_fetch} pages read · "
                f"{int(ev.elapsed) // 60:02d}:{int(ev.elapsed) % 60:02d}"
            )

    def render(self) -> Panel:
        t = Text()

        t.append("\n")
        t.append("  TOPIC  ", style="bold black on cyan")
        t.append(f"  {self.topic}\n", style="bold white")

        if self.plan:
            t.append("\n")
            t.append("  RESEARCH PLAN\n", style="bold yellow")
            if self.approach:
                t.append("  Strategy  ", style="dim")
                t.append(f"{self.approach}\n", style="italic white")
            t.append("\n")
            for i, q in enumerate(self.plan):
                bullet = CIRCLED[i] if i < len(CIRCLED) else f"{i + 1}."
                t.append(f"  {bullet} ", style="bold cyan")
                t.append(f"{q}\n", style="white")

        if self.events:
            t.append("\n")
            t.append("  ACTIVITY\n", style="bold yellow")
            for ts, icon, msg in self.events:
                t.append(f"  {ts}  ", style="dim")
                t.append(f"{icon}  ")
                t.append(f"{msg}\n", style="white")

        t.append("\n")
        t.append(f"  {self._spin()}  ", style="bold cyan")
        t.append(f"{self.status}\n", style="italic cyan")

        t.append("\n")
        t.append(f"  Searches: {self.n_search}", style="bold green")
        t.append("   *   ", style="dim")
        t.append(f"Pages read: {self.n_fetch}", style="bold blue")
        t.append("   *   ", style="dim")
        t.append(f"Elapsed: {self._elapsed()}\n", style="bold magenta")

        return Panel(
            t,
            title="[bold cyan] Deep Research [/bold cyan]",
            subtitle=f"[dim]{self.model}[/dim]",
            border_style="cyan",
            padding=(0, 1),
            box=box.ROUNDED,
        )


# ──────────────────────────────────────────────────────────────

def print_header(con: Console) -> None:
    con.clear()

    logo = Text(LOGO, style="bold cyan")

    info = Table.grid(padding=(0, 1))
    info.add_column(style="dim", min_width=9)
    info.add_column(style="bold white", min_width=28)
    info.add_row("", "")
    info.add_row("", Text("Deep Research Agent", style="bold white"))
    info.add_row("", Text("─" * 27, style="dim cyan"))
    info.add_row("Model", Text(MODEL, style="bold green"))
    info.add_row("Engine", Text("Ollama  (local)", style="bold yellow"))
    info.add_row("Search", Text("DuckDuckGo", style="bold blue"))
    info.add_row("Mode", Text("Agentic * Tool-use * Think", style="bold magenta"))

    layout = Table.grid(padding=(0, 4))
    layout.add_column()
    layout.add_column()
    layout.add_row(Padding(logo, (0, 0, 0, 1)), info)

    con.print()
    con.print(layout)
    con.print()
    con.print(Rule(style="bright_black"))
    con.print()


# ──────────────────────────────────────────────────────────────

def prompt_topic(con: Console) -> str:
    con.print(Text("  What would you like to research?", style="bold white"))
    con.print()
    con.print(Text("  > ", style="bold cyan"), end="")
    try:
        topic = input().strip()
    except (EOFError, KeyboardInterrupt):
        con.print("\n[dim]Cancelled.[/dim]")
        sys.exit(0)
    if not topic:
        con.print("[red]No topic provided.[/red]")
        sys.exit(0)
    con.print()
    return topic


# ──────────────────────────────────────────────────────────────

def run_with_ui(topic: str, con: Console) -> str:
    panel = ResearchPanel(topic, MODEL)

    with Live(
            panel.render(),
            console=con,
            refresh_per_second=10,
            vertical_overflow="visible",
            transient=False,
    ) as live:
        def on_event(ev: AgentEvent) -> None:
            panel.on_event(ev)
            live.update(panel.render())

        report = agent.run(topic, on_event=on_event)

        time.sleep(1.2)

    return report


# ──────────────────────────────────────────────────────────────

def print_report(con: Console, topic: str, report: str) -> None:
    con.print()
    con.print(Rule("[bold cyan]  Research Report  [/bold cyan]", style="cyan"))
    con.print()
    con.print(Padding(Markdown(report), (0, 4, 1, 4)))
    con.print(Rule(style="bright_black"))

    # con.print()
    # con.print("  [dim]Raw Report Content:[/dim]")
    # con.print(report)

    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:50]
    fname = os.path.join("reports", f"report_{safe.strip().replace(' ', '_')}.md")

    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(f"# Research Report: {topic}\n\n{report}")

    con.print()
    con.print(f"  [dim]Saved ->[/dim] [bold green]{fname}[/bold green]")
    con.print()


# ──────────────────────────────────────────────────────────────

def main() -> None:
    con = Console()
    print_header(con)

    topic = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else prompt_topic(con)
    report = run_with_ui(topic, con)
    print_report(con, topic, report)


if __name__ == "__main__":
    main()
