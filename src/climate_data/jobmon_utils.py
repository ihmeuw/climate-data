"""
Helpers for working with jobmon in the Climate Data CLI.

The main entrypoint is ``run_parallel_maybe_dry_run`` which mirrors
``rra_tools.jobmon.run_parallel`` but adds a ``dry_run`` flag.

When ``dry_run`` is ``True``, this helper:
* Expands ``flat_node_args`` or ``node_args`` into per-job CLI argument sets.
* Builds sbatch-like preview commands using the supplied task resources.
* Prints a short summary plus representative example commands.
* Returns a dummy success status without touching jobmon or the scheduler.

When ``dry_run`` is ``False``, it simply delegates to
``jobmon.run_parallel`` with identical semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rra_tools import jobmon

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_MAX_EXAMPLE_JOBS = 10


def _normalize_task_args(task_args: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(task_args or {})


def _iter_jobs_from_flat_node_args(
    flat_node_args: tuple[Sequence[str], Sequence[Sequence[Any]]],
) -> list[dict[str, Any]]:
    keys, rows = flat_node_args
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(dict(zip(keys, row, strict=False)))
    return jobs


def _iter_jobs_from_node_args(
    node_args: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    import itertools

    # jobmon treats ``node_args`` as a cartesian product over the sequences.
    items = list(node_args.items())
    if not items:
        return []

    keys = [k for k, _ in items]
    value_lists = [list(vs) for _, vs in items]

    jobs: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        jobs.append(dict(zip(keys, combo, strict=False)))
    return jobs


def _format_cli_args(args: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, list | tuple | set):
            for v in value:
                parts.append(f"{flag} {v}")
            continue
        parts.append(f"{flag} {value}")
    return " ".join(parts)


def _format_sbatch_like_line(
    *,
    runner: str,
    task_name: str,
    job_args: Mapping[str, Any],
    task_args: Mapping[str, Any],
    task_resources: Mapping[str, Any],
) -> str:
    base_cmd_parts = [runner, task_name]
    cli_for_job = _format_cli_args(job_args)
    cli_for_task = _format_cli_args(task_args)
    for segment in (cli_for_job, cli_for_task):
        if segment:
            base_cmd_parts.append(segment)
    command = " ".join(base_cmd_parts)

    # Escape any embedded double quotes for the wrap.
    command = command.replace('"', r"\"")

    queue = task_resources.get("queue", "")
    cores = task_resources.get("cores", "")
    memory = task_resources.get("memory", "")
    runtime = task_resources.get("runtime", "")
    project = task_resources.get("project", "")

    sbatch_parts = ["sbatch"]
    if queue:
        sbatch_parts.append(f"--partition={queue}")
    if cores:
        sbatch_parts.append(f"--cpus-per-task={cores}")
    if memory:
        sbatch_parts.append(f"--mem={memory}")
    if runtime:
        sbatch_parts.append(f"--time={runtime}")
    if project:
        sbatch_parts.append(f"--account={project}")
    sbatch_parts.append(f'--wrap="{command}"')

    return " ".join(sbatch_parts)


def run_parallel_maybe_dry_run(
    *,
    runner: str,
    task_name: str,
    task_resources: Mapping[str, Any],
    flat_node_args: tuple[Sequence[str], Sequence[Sequence[Any]]] | None = None,
    node_args: Mapping[str, Sequence[Any]] | None = None,
    task_args: Mapping[str, Any] | None = None,
    log_root: str | Path | None = None,
    max_attempts: int | None = None,
    concurrency_limit: int | None = None,
    dry_run: bool,
) -> Any:
    """Wrapper around ``jobmon.run_parallel`` with optional dry-run behavior.

    Parameters
    ----------
    runner
        The executable used for tasks, e.g. ``"cdtask"`` or ``"cdtask aggregate"``.
    task_name
        A short, human-readable name for the job group.
    task_resources
        Resource profile for the tasks (queue, cores, memory, runtime, project, ...).
    flat_node_args
        Tuple of (argument names, rows) describing per-node CLI arguments.
    node_args
        Mapping from argument name to a sequence of values; all combinations are used.
    task_args
        Shared CLI arguments that are the same for all nodes.
    log_root
        Optional log directory passed through to jobmon.
    max_attempts
        Maximum number of attempts; passed through to jobmon. Defaults to ``None``
        to match ``jobmon.run_parallel`` (jobmon then applies its own default).
    dry_run
        If ``True``, only print sbatch-like previews instead of submitting.
    """
    if not dry_run:
        kwargs: dict[str, Any] = {
            "runner": runner,
            "task_name": task_name,
            "flat_node_args": flat_node_args,
            "node_args": node_args,
            "task_args": task_args,
            "task_resources": task_resources,
            "log_root": log_root,
            "max_attempts": max_attempts,
        }
        if concurrency_limit is not None:
            kwargs["concurrency_limit"] = concurrency_limit
        return jobmon.run_parallel(**kwargs)

    normalized_task_args = _normalize_task_args(task_args)

    jobs: list[dict[str, Any]] = []
    if flat_node_args is not None:
        jobs = _iter_jobs_from_flat_node_args(flat_node_args)
    elif node_args is not None:
        jobs = _iter_jobs_from_node_args(node_args)
    else:
        jobs = [{}]

    n_jobs = len(jobs)
    queue = task_resources.get("queue", "")
    cores = task_resources.get("cores", "")
    memory = task_resources.get("memory", "")
    runtime = task_resources.get("runtime", "")
    project = task_resources.get("project", "")

    print(
        "[DRY-RUN] "
        f"Would submit {n_jobs} job{'s' if n_jobs != 1 else ''} "
        f"for task '{task_name}' via runner '{runner}'."
    )
    print("  Resources:")
    print(f"    queue   = {queue}")
    print(f"    cores   = {cores}")
    print(f"    memory  = {memory}")
    print(f"    runtime = {runtime}")
    print(f"    project = {project}")

    n_examples = min(n_jobs, _MAX_EXAMPLE_JOBS)
    if n_examples > 0:
        print(f"  Example sbatch-like commands (showing {n_examples} of {n_jobs}):")
        for job in jobs[:n_examples]:
            line = _format_sbatch_like_line(
                runner=runner,
                task_name=task_name,
                job_args=job,
                task_args=normalized_task_args,
                task_resources=task_resources,
            )
            print("   ", line)
        if n_jobs > n_examples:
            print(
                f"  ... ({n_jobs - n_examples} more job"
                f"{'s' if n_jobs - n_examples != 1 else ''} not shown)"
            )

    # Return a dummy "success" status so callers that expect a jobmon status
    # code continue without raising errors.
    return "D"
