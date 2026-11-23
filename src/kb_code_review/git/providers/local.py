"""本地 Git 实现：直接调用 git CLI，不依赖任何代码托管平台。

适用场景：

- 本地开发与单元测试；
- CI 中已 checkout 工作副本，只需读取 diff / 文件内容；
- 作为平台实现的兜底（平台 API 限流或不可用时改用本地克隆）。

与平台强绑定的能力（账号、PR、评审评论、提交状态）统一抛
``UnsupportedFeatureError``。
"""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlsplit

from ..base import (
    GitCredentials,
    GitService,
    GitServiceError,
    ResourceNotFoundError,
    UnsupportedFeatureError,
)
from ..models import (
    Branch,
    ChangeType,
    Commit,
    Diff,
    FileDiff,
    PullRequest,
    PullRequestState,
    Repository,
    RepositoryRef,
    ReviewCommentRecord,
    ReviewResult,
    ReviewSubmission,
    TreeEntry,
    TreeEntryType,
    User,
)

__all__ = ["LocalGitService"]

_COMMIT_FORMAT = "%H%x00%P%x00%an%x00%ae%x00%aI%x00%cI%x00%B"

_CHANGE_TYPES: dict[str, ChangeType] = {
    "A": ChangeType.ADDED,
    "M": ChangeType.MODIFIED,
    "D": ChangeType.DELETED,
    "R": ChangeType.RENAMED,
    "C": ChangeType.COPIED,
    "T": ChangeType.TYPE_CHANGED,
}

_TREE_TYPES: dict[str, TreeEntryType] = {
    "blob": TreeEntryType.FILE,
    "tree": TreeEntryType.DIRECTORY,
    "commit": TreeEntryType.SUBMODULE,
}

_NOT_FOUND_HINTS = (
    "unknown revision",
    "bad revision",
    "invalid object name",
    "not a valid object name",
    "ambiguous argument",
    "did not match any file",
    "does not exist",
    "no such file",
    "bad file",
    "bad object",
    "unable to read tree",
)

_SSH_SCP_RE = re.compile(r"^(?:(?P<user>[^@/]+)@)?(?P<host>[^:/]+):(?P<path>[^:]+)$")


def _decode(data: bytes) -> str:
    """按文件系统语义解码，避免非 UTF-8 路径直接抛错。"""
    return data.decode("utf-8", "surrogateescape")


def _parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _RemoteInfo:
    host: str
    owner: str
    name: str
    scheme: str


@dataclass(frozen=True, slots=True)
class _ChangeEntry:
    """`git diff --name-status` 的一行。"""

    status: str
    old_path: str | None
    new_path: str | None

    @property
    def key(self) -> tuple[str, ...]:
        """与 numstat 对齐用的键：重命名/复制用 (旧, 新)，其余用单路径。"""
        if self.status in ("R", "C"):
            return (self.old_path or "", self.new_path or "")
        return (self.new_path or self.old_path or "",)


def _parse_remote_url(url: str) -> _RemoteInfo | None:
    """解析 https / ssh / scp 风格的远端地址。"""
    url = url.strip()
    if not url:
        return None
    if "://" in url:
        parts = urlsplit(url)
        scheme = parts.scheme
        host = parts.hostname or ""
        path = parts.path
    else:
        matched = _SSH_SCP_RE.match(url)
        if matched is None:
            return None
        scheme = "ssh"
        host = matched.group("host")
        path = matched.group("path")
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host or "/" not in path:
        return None
    *owner_parts, name = path.split("/")
    owner = "/".join(owner_parts)
    if not owner or not name:
        return None
    return _RemoteInfo(host=host, owner=owner, name=name, scheme=scheme)


def _parse_name_status(data: bytes) -> list[_ChangeEntry]:
    tokens = _decode(data).split("\0")
    entries: list[_ChangeEntry] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        if not status:
            index += 1
            continue
        index += 1
        letter = status[0]
        if letter in ("R", "C"):
            old_path, new_path = tokens[index], tokens[index + 1]
            index += 2
        else:
            old_path = new_path = tokens[index]
            index += 1
        entries.append(
            _ChangeEntry(
                status=letter,
                old_path=None if letter == "A" else old_path,
                new_path=None if letter == "D" else new_path,
            )
        )
    return entries


def _parse_numstat(data: bytes) -> dict[tuple[str, ...], tuple[int, int, bool]]:
    tokens = _decode(data).split("\0")
    stats: dict[tuple[str, ...], tuple[int, int, bool]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        additions, deletions, rest = token.split("\t", 2)
        if rest:
            key: tuple[str, ...] = (rest,)
        else:  # 重命名/复制：路径在两个独立 token 中
            key = (tokens[index], tokens[index + 1])
            index += 2
        binary = additions == "-" or deletions == "-"
        stats[key] = (
            0 if binary else int(additions),
            0 if binary else int(deletions),
            binary,
        )
    return stats


class LocalGitService(GitService):
    """基于本地工作副本的 Git 服务。

    Args:
        workdir: 工作副本中任意路径，会向上解析到仓库根目录。
        credentials: 本地仓库不使用，仅为与基类保持一致。
        remote: 远端名称，用于推导仓库信息与默认分支。
        git_executable: git 可执行文件路径。
        timeout: 单条 git 命令超时时间（秒）。
        max_concurrency: 并发拉取单文件 patch 的上限。
    """

    platform = "local"

    def __init__(
        self,
        workdir: str | Path,
        *,
        credentials: GitCredentials | None = None,
        remote: str = "origin",
        git_executable: str = "git",
        timeout: float = 120.0,
        max_concurrency: int = 8,
    ) -> None:
        self._git = git_executable
        self._timeout = timeout
        self._remote = remote
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._workdir = Path(workdir).expanduser()
        self._workdir = self._resolve_toplevel()
        self._remote_info = _parse_remote_url(self._remote_url_sync())

        if self._remote_info is not None:
            repository = RepositoryRef(
                owner=self._remote_info.owner,
                name=self._remote_info.name,
            )
        else:
            repository = RepositoryRef(
                owner=self._workdir.parent.name or "",
                name=self._workdir.name,
            )

        super().__init__(repository, credentials or GitCredentials())

    # ------------------------------------------------------------------ 构造辅助

    def _resolve_toplevel(self) -> Path:
        _, stdout, _ = self._sync_run("rev-parse", "--show-toplevel")
        toplevel = _decode(stdout).strip()
        if not toplevel:
            raise GitServiceError(f"{self._workdir} 不是 Git 仓库")
        return Path(toplevel)

    def _remote_url_sync(self) -> str:
        code, stdout, _ = self._sync_run(
            "config",
            "--get",
            f"remote.{self._remote}.url",
            check=False,
        )
        if code != 0:
            return ""
        return _decode(stdout).strip()

    # ------------------------------------------------------------------ 命令执行

    def _command(self, *args: str) -> list[str]:
        return [
            self._git,
            "--no-pager",
            "-C",
            str(self._workdir),
            "-c",
            "core.quotepath=false",
            *args,
        ]

    def _raise_for(self, returncode: int, stderr: bytes, args: tuple[str, ...]) -> NoReturn:
        text = stderr.decode("utf-8", "replace").strip()
        lowered = text.lower()
        if "not a git repository" in lowered:
            raise GitServiceError(f"{self._workdir} 不是 Git 仓库：{text}")
        if any(hint in lowered for hint in _NOT_FOUND_HINTS):
            raise ResourceNotFoundError(f"git {args[0]} 未找到对象：{text}")
        raise GitServiceError(f"git {args[0]} 执行失败（exit={returncode}）：{text}")

    def _sync_run(self, *args: str, check: bool = True) -> tuple[int, bytes, bytes]:
        try:
            proc = subprocess.run(
                self._command(*args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout,
            )
        except FileNotFoundError as exc:
            raise GitServiceError(f"未找到 git 可执行文件：{self._git}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitServiceError(f"git {args[0]} 执行超时（{self._timeout}s）") from exc
        if check and proc.returncode != 0:
            self._raise_for(proc.returncode, proc.stderr, args)
        return proc.returncode, proc.stdout, proc.stderr

    async def _execute(self, *args: str) -> tuple[int, bytes, bytes]:
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "LC_ALL": "C",
        }
        proc = await asyncio.create_subprocess_exec(
            *self._command(*args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise GitServiceError(f"git {args[0]} 执行超时（{self._timeout}s）") from None
        return proc.returncode or 0, stdout, stderr

    async def _run(self, *args: str) -> bytes:
        code, stdout, stderr = await self._execute(*args)
        if code != 0:
            self._raise_for(code, stderr, args)
        return stdout

    async def _try(self, *args: str) -> bytes | None:
        code, stdout, _ = await self._execute(*args)
        return stdout if code == 0 else None

    # ------------------------------------------------------------------ 内部工具

    async def _resolve_sha(self, ref: str) -> str:
        stdout = await self._try("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        sha = _decode(stdout).strip() if stdout else ""
        if not sha:
            raise ResourceNotFoundError(f"无法解析引用：{ref}")
        return sha

    async def _default_branch(self) -> str:
        stdout = await self._try(
            "symbolic-ref",
            "--quiet",
            "--short",
            f"refs/remotes/{self._remote}/HEAD",
        )
        if stdout:
            ref = _decode(stdout).strip()
            prefix = f"{self._remote}/"
            return ref[len(prefix) :] if ref.startswith(prefix) else ref

        stdout = await self._try("rev-parse", "--abbrev-ref", "HEAD")
        if stdout:
            branch = _decode(stdout).strip()
            if branch and branch != "HEAD":
                return branch
        return "main"

    async def _remote_url(self) -> str:
        stdout = await self._try("config", "--get", f"remote.{self._remote}.url")
        return _decode(stdout).strip() if stdout else ""

    async def _file_patch(self, base_sha: str, head_sha: str, entry: _ChangeEntry) -> str:
        paths = list(dict.fromkeys(p for p in (entry.old_path, entry.new_path) if p))
        async with self._semaphore:
            data = await self._run(
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--find-renames",
                base_sha,
                head_sha,
                "--",
                *paths,
            )
        return data.decode("utf-8", "replace")

    async def _diff_files(
        self,
        base_sha: str,
        head_sha: str,
        *,
        include_patch: bool,
    ) -> tuple[FileDiff, ...]:
        entries = _parse_name_status(
            await self._run(
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                "--no-color",
                "--no-ext-diff",
                base_sha,
                head_sha,
            )
        )
        if not entries:
            return ()

        stats = _parse_numstat(
            await self._run(
                "diff",
                "--numstat",
                "-z",
                "--find-renames",
                "--no-color",
                "--no-ext-diff",
                base_sha,
                head_sha,
            )
        )

        rows: list[tuple[_ChangeEntry, int, int, bool, str | None]] = []
        for entry in entries:
            additions, deletions, binary = stats.get(entry.key, (0, 0, False))
            rows.append((entry, additions, deletions, binary, None))

        if include_patch:
            patchable = [i for i, row in enumerate(rows) if not row[3]]
            patches = await asyncio.gather(
                *(self._file_patch(base_sha, head_sha, rows[i][0]) for i in patchable)
            )
            for index, patch in zip(patchable, patches, strict=True):
                entry, additions, deletions, binary, _ = rows[index]
                rows[index] = (entry, additions, deletions, binary, patch)

        return tuple(
            FileDiff(
                change_type=_CHANGE_TYPES.get(entry.status, ChangeType.UNKNOWN),
                old_path=entry.old_path,
                new_path=entry.new_path,
                patch=patch,
                is_binary=binary,
                additions=additions,
                deletions=deletions,
            )
            for entry, additions, deletions, binary, patch in rows
        )

    # ------------------------------------------------------------------ 账号（平台能力）

    async def get_authenticated_user(self) -> User:
        raise self._unsupported("账号信息（get_authenticated_user）")

    # ------------------------------------------------------------------ 仓库与引用

    async def get_repository(self) -> Repository:
        remote_url = await self._remote_url()
        info = self._remote_info or _parse_remote_url(remote_url)
        default_branch = await self._default_branch()

        if info is not None:
            web_url = f"https://{info.host}/{info.owner}/{info.name}"
        else:
            web_url = self._workdir.as_uri()

        if info is not None:
            clone_url_http = (
                remote_url
                if remote_url.startswith(("http://", "https://"))
                else f"https://{info.host}/{info.owner}/{info.name}.git"
            )
        else:
            clone_url_http = remote_url or self._workdir.as_uri()

        clone_url_ssh = remote_url if remote_url and info and info.scheme == "ssh" else None

        return Repository(
            ref=self.repository_ref,
            default_branch=default_branch,
            web_url=web_url,
            clone_url_http=clone_url_http,
            clone_url_ssh=clone_url_ssh,
            private=False,
        )

    async def get_branch(self, name: str) -> Branch:
        candidates = (f"refs/heads/{name}", f"refs/remotes/{name}", name)
        for candidate in candidates:
            stdout = await self._try(
                "rev-parse",
                "--verify",
                "--quiet",
                f"{candidate}^{{commit}}",
            )
            sha = _decode(stdout).strip() if stdout else ""
            if sha:
                return Branch(name=name, commit_sha=sha)
        raise ResourceNotFoundError(f"分支不存在：{name}")

    async def list_branches(self, *, search: str | None = None, limit: int = 100) -> list[Branch]:
        stdout = await self._run(
            "for-each-ref",
            "--format=%(refname)%00%(refname:short)%00%(objectname)",
            "refs/heads",
            "refs/remotes",
        )
        keyword = search.lower() if search else None
        branches: list[Branch] = []
        for line in _decode(stdout).splitlines():
            line = line.strip()
            if not line:
                continue
            full_ref, _, remainder = line.partition("\0")
            name, _, sha = remainder.partition("\0")
            if not name or not sha or full_ref.endswith("/HEAD"):
                continue
            if keyword and keyword not in name.lower():
                continue
            branches.append(Branch(name=name, commit_sha=sha))
            if len(branches) >= limit:
                break
        return branches

    async def get_commit(self, ref: str) -> Commit:
        stdout = await self._run("log", "-1", f"--format={_COMMIT_FORMAT}", ref, "--")
        return self._parse_commit(_decode(stdout).rstrip("\n"))

    async def list_commits(
        self,
        *,
        ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Commit]:
        args = ["log", f"--format={_COMMIT_FORMAT}%x1e", f"--max-count={max(1, limit)}"]
        if since is not None:
            args.append(f"--since={since.isoformat()}")
        if until is not None:
            args.append(f"--until={until.isoformat()}")
        args.extend([ref or "HEAD", "--"])

        stdout = await self._run(*args)
        commits: list[Commit] = []
        for record in _decode(stdout).split("\x1e"):
            record = record.strip("\n")
            if record:
                commits.append(self._parse_commit(record))
        return commits

    async def get_merge_base(self, base: str, head: str) -> str:
        base_sha = await self._resolve_sha(base)
        head_sha = await self._resolve_sha(head)
        code, stdout, stderr = await self._execute("merge-base", base_sha, head_sha)
        if code != 0:
            if code == 1:
                raise GitServiceError(f"{base} 与 {head} 没有共同祖先")
            self._raise_for(code, stderr, ("merge-base",))
        return _decode(stdout).strip()

    # ------------------------------------------------------------------ 变更与源码

    async def compare(
        self,
        base: str,
        head: str,
        *,
        use_merge_base: bool = True,
        include_patch: bool = True,
    ) -> Diff:
        base_sha = await self._resolve_sha(base)
        head_sha = await self._resolve_sha(head)

        merge_base_sha: str | None = None
        target = base_sha
        if use_merge_base:
            merge_base_sha = await self.get_merge_base(base_sha, head_sha)
            target = merge_base_sha

        files = await self._diff_files(target, head_sha, include_patch=include_patch)
        return Diff(
            base_sha=base_sha,
            head_sha=head_sha,
            merge_base_sha=merge_base_sha,
            files=files,
        )

    async def list_tree(
        self,
        ref: str | None = None,
        *,
        path: str = "",
        recursive: bool = True,
    ) -> list[TreeEntry]:
        args = ["ls-tree", "-z", "-l"]
        if recursive:
            args.append("-r")
        args.append(ref or "HEAD")
        if path:
            args.extend(["--", path])

        stdout = await self._run(*args)
        entries: list[TreeEntry] = []
        for token in _decode(stdout).split("\0"):
            if not token:
                continue
            meta, _, entry_path = token.partition("\t")
            fields = meta.split()
            if len(fields) < 3 or not entry_path:
                continue
            object_type = _TREE_TYPES.get(fields[1])
            if object_type is None:
                continue
            size: int | None = None
            if len(fields) >= 4 and fields[3].isdigit():
                size = int(fields[3])
            entries.append(
                TreeEntry(
                    path=entry_path,
                    type=object_type,
                    sha=fields[2],
                    size=size,
                )
            )
        return entries

    async def get_file_content(
        self,
        path: str,
        *,
        ref: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        target = ref or "HEAD"
        stdout = await self._run("cat-file", "blob", f"{target}:{path}")
        if b"\0" in stdout:
            raise UnsupportedFeatureError(f"{path} 是二进制文件，无法按文本读取")

        text = stdout.decode("utf-8", "replace")
        if line_start is None and line_end is None:
            return text

        start = max(1, line_start or 1)
        if line_end is not None and line_end < start:
            raise ValueError(f"line_end({line_end}) 小于 line_start({start})")
        lines = text.splitlines(keepends=True)
        if start > len(lines):
            return ""
        end = line_end if line_end is not None else len(lines)
        return "".join(lines[start - 1 : end])

    # ------------------------------------------------------------------ 合并请求（平台能力）

    async def get_pull_request(self, number: int) -> PullRequest:
        raise self._unsupported(f"合并请求（get_pull_request，#{number}）")

    async def list_pull_requests(
        self,
        *,
        state: PullRequestState = PullRequestState.OPEN,
        source_branch: str | None = None,
        limit: int = 50,
    ) -> list[PullRequest]:
        raise self._unsupported("合并请求列表（list_pull_requests）")

    async def get_pull_request_diff(self, number: int, *, include_patch: bool = True) -> Diff:
        raise self._unsupported(f"PR 变更集（get_pull_request_diff，#{number}）")

    async def get_pull_request_commits(self, number: int) -> list[Commit]:
        raise self._unsupported(f"PR 提交列表（get_pull_request_commits，#{number}）")

    # ------------------------------------------------------------------ 评审回写（平台能力）

    async def submit_review(
        self,
        number: int,
        submission: ReviewSubmission,
        *,
        commit_sha: str | None = None,
    ) -> ReviewResult:
        raise self._unsupported(f"评审回写（submit_review，#{number}）")

    async def list_review_comments(self, number: int) -> list[ReviewCommentRecord]:
        raise self._unsupported(f"评论列表（list_review_comments，#{number}）")

    async def reply_to_review_comment(
        self,
        number: int,
        comment_id: str,
        body: str,
    ) -> ReviewCommentRecord:
        raise self._unsupported(f"评论回复（reply_to_review_comment，#{comment_id}）")

    # ------------------------------------------------------------------ 可选能力

    async def create_branch(self, name: str, *, from_ref: str | None = None) -> Branch:
        start_point = from_ref or await self._default_branch()
        await self._run("branch", name, start_point)
        return await self.get_branch(name)

    # ------------------------------------------------------------------ 其他

    @staticmethod
    def _unsupported(feature: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"本地 GitService 不支持 {feature}，请使用对应平台的实现")

    def _parse_commit(self, record: str) -> Commit:
        sha, parents, name, email, authored_at, committed_at, message = record.split("\x00", 6)
        return Commit(
            sha=sha,
            message=message.rstrip("\n"),
            author=User(username=name, display_name=name or None, email=email or None),
            parent_shas=tuple(parents.split()),
            authored_at=_parse_datetime(authored_at),
            committed_at=_parse_datetime(committed_at),
        )
