# src/spine_segmentation/commands/root.py
import click

@click.group()
@click.option('--verbose', is_flag=True, help="Enable verbose output")
@click.option('--debug', is_flag=True, help="Wait for debugger to attach on port 5678")
@click.pass_context
def cli(ctx, verbose, debug):
    """Spine Segmentation CLI."""
    # Store global options in context if needed
    ctx.obj = {"verbose": verbose}
    if debug:
        import debugpy
        debugpy.connect(("localhost", 5678))
        click.echo("🐛 Waiting for debugger attach on port 5678...")
        debugpy.wait_for_client()
