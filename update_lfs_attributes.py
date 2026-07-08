#!/usr/bin/env python3
#
# 遍历仓库文件，并根据 FileTypes_Binary.md 和 FileTypes_Text.md 判定文件类型。
# 被忽略的文件会跳过，包括 Git 标准忽略规则，以及可选的 .ignore 和
# .git hook exclude 文件。
# 如果有文件无法分类，脚本会输出错误并且不会更新 .gitattributes。
# 大于指定阈值的二进制文件会写入 .gitattributes 的自动 LFS 区域。
#
# 使用说明:
#   1. 预览结果，不修改 .gitattributes:
#      python update_lfs_attributes.py --dry-run
#   2. 使用默认阈值 128 KiB 更新 .gitattributes:
#      python update_lfs_attributes.py
#   3. 传入文件尺寸阈值，例如只处理大于 256 KiB 的二进制文件:
#      python update_lfs_attributes.py --threshold-kb 256
#   4. 输出全部未分类错误:
#      python update_lfs_attributes.py --dry-run --max-errors 0
#   5. 调整进度刷新频率，或关闭进度:
#      python update_lfs_attributes.py --progress-interval 1000
#      python update_lfs_attributes.py --no-progress

from __future__ import annotations

import argparse
import fnmatch
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


BEGIN_MARKER = "# Auto added LFS files. [begin]"
END_MARKER = "# Auto added LFS files. [end]"
LFS_ATTRIBUTES = "filter=lfs diff=lfs merge=lfs -text"
DEFAULT_THRESHOLD_KB = 128
DEFAULT_PROGRESS_INTERVAL = 1000


@dataclass(frozen=True)
class TypePattern:
    pattern: str
    source: str
    line_no: int


@dataclass(frozen=True)
class MatchResult:
    pattern: TypePattern
    score: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class PatternSet:
    exact_by_name: dict[str, tuple[TypePattern, ...]]
    suffix_by_name: dict[str, tuple[TypePattern, ...]]
    other_patterns: tuple[TypePattern, ...]


@dataclass(frozen=True)
class IgnoreRule:
    pattern: str
    source: str
    line_no: int
    negated: bool
    directory_only: bool
    anchored: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "根据文件类型清单扫描仓库文件，并把超过阈值的二进制文件写入 "
            ".gitattributes 的自动 LFS 区域。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python update_lfs_attributes.py --dry-run
  python update_lfs_attributes.py
  python update_lfs_attributes.py --threshold-kb 256
  python update_lfs_attributes.py --dry-run --max-errors 0
  python update_lfs_attributes.py --progress-interval 1000
  python update_lfs_attributes.py --no-progress""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划变更，不写入 .gitattributes",
    )
    parser.add_argument(
        "--threshold-kb",
        type=int,
        default=DEFAULT_THRESHOLD_KB,
        help=f"加入 LFS 的二进制文件最小尺寸，单位 KiB，默认 {DEFAULT_THRESHOLD_KB}",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="最多输出多少条分类错误；传 0 表示不限制",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="执行过程中不显示扫描进度",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help=f"每处理多少个文件输出一次进度，默认 {DEFAULT_PROGRESS_INTERVAL}",
    )
    return parser.parse_args()


def run_git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result


def find_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"not inside a Git repository: {stderr}")
    return Path(result.stdout.decode("utf-8", errors="replace").strip()).resolve()


def find_git_dir(repo_root: Path) -> Path:
    result = run_git(repo_root, ["rev-parse", "--git-dir"])
    git_dir = Path(result.stdout.decode("utf-8", errors="replace").strip())
    if not git_dir.is_absolute():
        git_dir = repo_root / git_dir
    return git_dir.resolve()


def normalize_pattern(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        tokens = shlex.split(stripped, comments=False, posix=True)
    except ValueError:
        tokens = [stripped]
    if not tokens:
        return None
    if len(tokens) == 1:
        pattern = tokens[0]
    else:
        pattern = stripped
    pattern = pattern.replace("\\", "/").strip()
    return pattern or None


def load_type_patterns(repo_root: Path, file_name: str) -> list[TypePattern]:
    path = repo_root / file_name
    if not path.exists():
        raise RuntimeError(f"required type list is missing: {path}")

    patterns: list[TypePattern] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        pattern = normalize_pattern(line)
        if pattern is None:
            continue
        patterns.append(TypePattern(pattern=pattern, source=file_name, line_no=line_no))
    return patterns


def iter_repo_files(repo_root: Path) -> list[str]:
    result = run_git(
        repo_root,
        ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    )
    files: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        path = repo_root / rel
        if path.is_file() and ".git/" not in rel and rel != ".git":
            files.append(rel)
    return sorted(set(files))


def parse_ignore_rule(raw: str, source: str, line_no: int) -> IgnoreRule | None:
    line = raw.rstrip("\n\r")
    line = line.rstrip()
    if not line:
        return None
    if line.startswith(r"\#"):
        line = line[1:]
    elif line.startswith("#"):
        return None

    negated = False
    if line.startswith(r"\!"):
        line = line[1:]
    elif line.startswith("!"):
        negated = True
        line = line[1:]

    line = line.replace("\\", "/")
    directory_only = line.endswith("/")
    line = line.strip("/")
    if not line:
        return None
    anchored = raw.lstrip().startswith("/")
    return IgnoreRule(
        pattern=line,
        source=source,
        line_no=line_no,
        negated=negated,
        directory_only=directory_only,
        anchored=anchored,
    )


def load_ignore_rules(paths: Iterable[Path]) -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, line in enumerate(lines, start=1):
            rule = parse_ignore_rule(line, str(path), line_no)
            if rule is not None:
                rules.append(rule)
    return rules


def match_path_pattern(rel_path: str, pattern: str, *, directory_only: bool, anchored: bool) -> bool:
    rel_path = rel_path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    if directory_only:
        parent_dirs = rel_path.split("/")[:-1]
        if not parent_dirs:
            return False
        dir_paths = ["/".join(parent_dirs[: index + 1]) for index in range(len(parent_dirs))]
        if anchored:
            candidates = dir_paths
        elif "/" in pattern:
            suffixes: list[str] = []
            for dir_path in dir_paths:
                parts = dir_path.split("/")
                suffixes.extend("/".join(parts[index:]) for index in range(1, len(parts)))
            candidates = dir_paths + suffixes
        else:
            candidates = [part for part in parent_dirs] + dir_paths
        return any(fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates)

    if "/" not in pattern:
        return any(fnmatch.fnmatchcase(part, pattern) for part in rel_path.split("/"))

    if anchored:
        return fnmatch.fnmatchcase(rel_path, pattern)

    if fnmatch.fnmatchcase(rel_path, pattern):
        return True
    parts = rel_path.split("/")
    suffixes = ["/".join(parts[index:]) for index in range(1, len(parts))]
    return any(fnmatch.fnmatchcase(suffix, pattern) for suffix in suffixes)


def custom_ignored_paths(rel_paths: Sequence[str], rules: Sequence[IgnoreRule]) -> set[str]:
    ignored: set[str] = set()
    for rel_path in rel_paths:
        ignored_state = False
        for rule in rules:
            if match_path_pattern(
                rel_path,
                rule.pattern,
                directory_only=rule.directory_only,
                anchored=rule.anchored,
            ):
                ignored_state = not rule.negated
        if ignored_state:
            ignored.add(rel_path)
    return ignored


def is_glob_pattern(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def is_simple_suffix_pattern(pattern: str) -> bool:
    return (
        "/" not in pattern
        and pattern.startswith("*.")
        and pattern.count("*") == 1
        and "?" not in pattern
        and "[" not in pattern
    )


def build_pattern_set(patterns: Sequence[TypePattern]) -> PatternSet:
    exact_by_name: dict[str, list[TypePattern]] = {}
    suffix_by_name: dict[str, list[TypePattern]] = {}
    other_patterns: list[TypePattern] = []

    for pattern in patterns:
        if "/" not in pattern.pattern and not is_glob_pattern(pattern.pattern):
            exact_by_name.setdefault(pattern.pattern, []).append(pattern)
        elif is_simple_suffix_pattern(pattern.pattern):
            suffix_by_name.setdefault(pattern.pattern[1:], []).append(pattern)
        else:
            other_patterns.append(pattern)

    return PatternSet(
        exact_by_name={key: tuple(value) for key, value in exact_by_name.items()},
        suffix_by_name={key: tuple(value) for key, value in suffix_by_name.items()},
        other_patterns=tuple(other_patterns),
    )


def basename_suffixes(name: str) -> list[str]:
    return [name[index:] for index, char in enumerate(name) if char == "."]


def pattern_specificity(pattern: str, rel_path: str) -> tuple[int, int, int, int, int]:
    exact = 0
    if not is_glob_pattern(pattern):
        if "/" in pattern and pattern == rel_path:
            exact = 2
        elif "/" not in pattern and pattern == Path(rel_path).name:
            exact = 1
    literal_chars = re.sub(r"[*?\[\]]", "", pattern)
    wildcard_count = len(pattern) - len(literal_chars)
    return (
        exact,
        pattern.count("/"),
        len(literal_chars),
        -wildcard_count,
        len(pattern),
    )


def type_pattern_matches(rel_path: str, pattern: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    if "/" not in pattern:
        return fnmatch.fnmatchcase(Path(rel_path).name, pattern)
    return fnmatch.fnmatchcase(rel_path, pattern)


def best_type_match(rel_path: str, patterns: PatternSet) -> MatchResult | None:
    best: MatchResult | None = None
    name = Path(rel_path).name
    candidates: list[TypePattern] = []
    candidates.extend(patterns.exact_by_name.get(name, ()))
    for suffix in basename_suffixes(name):
        candidates.extend(patterns.suffix_by_name.get(suffix, ()))
    candidates.extend(
        pattern for pattern in patterns.other_patterns if type_pattern_matches(rel_path, pattern.pattern)
    )

    for pattern in candidates:
        score = pattern_specificity(pattern.pattern, rel_path)
        current = MatchResult(pattern=pattern, score=score)
        if best is None or current.score > best.score:
            best = current
    return best


def classify_file(
    rel_path: str,
    text_patterns: PatternSet,
    binary_patterns: PatternSet,
) -> tuple[str | None, str | None]:
    text_match = best_type_match(rel_path, text_patterns)
    binary_match = best_type_match(rel_path, binary_patterns)

    if text_match is None and binary_match is None:
        return None, f"unclassified: {rel_path}"
    if text_match is not None and binary_match is None:
        return "text", None
    if binary_match is not None and text_match is None:
        return "binary", None

    assert text_match is not None
    assert binary_match is not None
    if binary_match.score > text_match.score:
        return "binary", None
    if text_match.score > binary_match.score:
        return "text", None

    return (
        None,
        (
            f"ambiguous: {rel_path} matches both "
            f"{text_match.pattern.source}:{text_match.pattern.line_no} "
            f"({text_match.pattern.pattern}) and "
            f"{binary_match.pattern.source}:{binary_match.pattern.line_no} "
            f"({binary_match.pattern.pattern})"
        ),
    )


def escape_gitattributes_pattern(rel_path: str) -> str:
    pattern = rel_path.replace("\\", "/")
    escaped: list[str] = []
    for char in pattern:
        if char == "*":
            escaped.append("[*]")
        elif char == "?":
            escaped.append("[?]")
        elif char == "[":
            escaped.append("[[]")
        else:
            escaped.append(char)

    escaped_pattern = "".join(escaped)
    needs_quote = (
        any(char.isspace() for char in escaped_pattern)
        or escaped_pattern.startswith(("#", "!"))
        or '"' in escaped_pattern
    )
    if not needs_quote:
        return escaped_pattern

    quoted = escaped_pattern.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{quoted}"'


def build_lfs_lines(rel_paths: Sequence[str]) -> list[str]:
    return [f"{escape_gitattributes_pattern(rel_path)} {LFS_ATTRIBUTES}\n" for rel_path in sorted(set(rel_paths))]


def replace_auto_section(gitattributes: Path, lfs_lines: Sequence[str]) -> None:
    if not gitattributes.exists():
        raise RuntimeError(f"missing .gitattributes: {gitattributes}")

    lines = gitattributes.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    begin_indexes = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == BEGIN_MARKER]
    end_indexes = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == END_MARKER]

    if len(begin_indexes) != 1 or len(end_indexes) != 1:
        raise RuntimeError(
            ".gitattributes must contain exactly one auto LFS begin marker "
            "and exactly one end marker"
        )
    begin = begin_indexes[0]
    end = end_indexes[0]
    if begin >= end:
        raise RuntimeError(".gitattributes auto LFS begin marker must appear before the end marker")

    new_lines = lines[: begin + 1] + list(lfs_lines) + lines[end:]
    gitattributes.write_text("".join(new_lines), encoding="utf-8")


def print_errors(errors: Sequence[str], max_errors: int) -> None:
    limit = len(errors) if max_errors == 0 else min(len(errors), max_errors)
    for error in errors[:limit]:
        print(f"ERROR: {error}", file=sys.stderr)
    remaining = len(errors) - limit
    if remaining > 0:
        print(f"ERROR: ... {remaining} more error(s) omitted", file=sys.stderr)


def print_progress(current: int, total: int, start_time: float) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    percent = 100.0 if total == 0 else current * 100.0 / total
    rate = current / elapsed
    print(
        f"进度: {current}/{total} ({percent:.1f}%), 用时 {elapsed:.1f}s, {rate:.0f} 文件/s",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    if args.threshold_kb < 0:
        print("ERROR: --threshold-kb 不能为负数", file=sys.stderr)
        return 1
    if args.progress_interval <= 0:
        print("ERROR: --progress-interval 必须大于 0", file=sys.stderr)
        return 1

    try:
        repo_root = find_repo_root()
        git_dir = find_git_dir(repo_root)
        text_patterns = build_pattern_set(load_type_patterns(repo_root, "FileTypes_Text.md"))
        binary_patterns = build_pattern_set(load_type_patterns(repo_root, "FileTypes_Binary.md"))

        rel_paths = iter_repo_files(repo_root)
        ignore_sources = [
            repo_root / ".ignore",
            git_dir / "info" / "exclude",
            git_dir / "hook" / "exclude",
            git_dir / "hooks" / "exclude",
        ]
        ignore_rules = load_ignore_rules(ignore_sources)
        custom_ignored = custom_ignored_paths(rel_paths, ignore_rules)
        scanned_paths = [path for path in rel_paths if path not in custom_ignored]

        threshold_bytes = args.threshold_kb * 1024
        lfs_paths: list[str] = []
        errors: list[str] = []
        show_progress = not args.no_progress
        scan_start_time = time.monotonic()
        last_progress = 0

        if show_progress:
            print(f"开始扫描 {len(scanned_paths)} 个文件...", file=sys.stderr)

        for index, rel_path in enumerate(scanned_paths, start=1):
            file_type, error = classify_file(rel_path, text_patterns, binary_patterns)
            if error is not None:
                errors.append(error)
            elif file_type == "binary":
                size = (repo_root / rel_path).stat().st_size
                if size > threshold_bytes:
                    lfs_paths.append(rel_path)

            if show_progress and index % args.progress_interval == 0:
                print_progress(index, len(scanned_paths), scan_start_time)
                last_progress = index

        if show_progress and last_progress != len(scanned_paths):
            print_progress(len(scanned_paths), len(scanned_paths), scan_start_time)

        if errors:
            print_errors(errors, args.max_errors)
            print(
                f"Checked {len(scanned_paths)} file(s); found {len(errors)} classification error(s).",
                file=sys.stderr,
            )
            return 1

        lfs_lines = build_lfs_lines(lfs_paths)
        print(f"Checked {len(scanned_paths)} file(s).")
        print(f"Skipped {len(custom_ignored)} extra ignored file(s).")
        print(f"Found {len(lfs_paths)} binary file(s) larger than {args.threshold_kb} KiB.")

        if args.dry_run:
            print("Dry run: .gitattributes was not modified.")
            for rel_path in sorted(set(lfs_paths)):
                print(rel_path)
            return 0

        replace_auto_section(repo_root / ".gitattributes", lfs_lines)
        print("Updated .gitattributes auto-added LFS section.")
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
