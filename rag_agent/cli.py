import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


@click.group()
def main():
    """RAG Agent - Ask questions about your documents."""


@main.command()
@click.argument("path")
def ingest(path: str):
    """Ingest documents from a file or directory into the vector store."""
    from rag_agent.config import settings
    from rag_agent.ingest import ingest as run_ingest
    from rag_agent.ingest import load_documents, chunk_documents

    try:
        settings.validate_api_keys()
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"[bold]Loading documents from:[/bold] {path}")

    try:
        docs = load_documents(path)
        console.print(f"  Loaded [green]{len(docs)}[/green] document(s)")

        chunks = chunk_documents(docs)
        console.print(f"  Split into [green]{len(chunks)}[/green] chunks")

        console.print("  Embedding and storing...", end=" ")
        run_ingest(path)
        console.print("[green]done![/green]")

        console.print(Panel(
            f"Successfully ingested {len(docs)} document(s) "
            f"({len(chunks)} chunks) into the vector store.",
            title="Ingestion Complete",
            border_style="green",
        ))
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("question")
def ask(question: str):
    """Ask a one-shot question about your documents."""
    from rag_agent.agent import build_agent
    from rag_agent.config import settings

    try:
        settings.validate_api_keys()
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(1)

    try:
        agent = build_agent()
        result = agent.invoke({"input": question})
        console.print()
        console.print(Panel(
            Markdown(result["output"]),
            title="Answer",
            border_style="blue",
        ))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@main.command()
def chat():
    """Start an interactive chat session with the RAG agent."""
    from langchain_core.messages import AIMessage, HumanMessage

    from rag_agent.agent import build_agent
    from rag_agent.config import settings

    try:
        settings.validate_api_keys()
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(1)

    console.print(Panel(
        "Interactive RAG Chat\n"
        "Ask questions about your ingested documents.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to end the session.",
        title="RAG Agent",
        border_style="blue",
    ))

    try:
        agent = build_agent()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    chat_history: list[HumanMessage | AIMessage] = []

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        try:
            result = agent.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            answer = result["output"]

            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=answer))

            console.print()
            console.print(Panel(
                Markdown(answer),
                title="Agent",
                border_style="blue",
            ))
            console.print()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
    main()
