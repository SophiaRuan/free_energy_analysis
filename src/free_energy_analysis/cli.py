"""Console script for free_energy_analysis."""
import free_energy_analysis

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for free_energy_analysis."""
    console.print("Replace this message by putting your code into "
               "free_energy_analysis.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
