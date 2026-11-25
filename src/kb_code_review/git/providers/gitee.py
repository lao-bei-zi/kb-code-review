"""Gitee（码云）OpenAPI v5 实现。

接口文档：https://gitee.com/api/v5/swagger （机器可读版本：``/api/v5/swagger_doc``）

认证
----
优先使用 ``Authorization: token <access_token>`` 请求头（已实测可用），
避免把 token 拼进 URL 造成日志/代理泄露。token 来源优先级：

1. 构造函数 ``token`` 参数；
2. ``GitCredentials.token``；
3. 环境变量 ``GITEE_ACCESS_TOKEN``。
"""

import asyncio
import base64
import binascii
import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from ..base import (
    AuthenticationError,
    GitCredentials,
    GitService,
    GitServiceError,
    RateLimitError,
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
    ReviewEvent,
    ReviewResult,
    ReviewSide,
    ReviewSubmission,
    TreeEntry,
    TreeEntryType,
    User,
)

__all__ = ["GiteeGitService"]

DEFAULT_API_BASE = "https://gitee.com/api/v5"
TOKEN_ENV = "GITEE_ACCESS_TOKEN"
MAX_PER_PAGE = 100
#: Gitee 的 PR 文件接口最多返回 300 条 diff
PR_FILES_LIMIT = 300
#: 防御性上限，避免分页异常时无限循环
MAX_PAGES = 200

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_CHANGE_TYPES: dict[str, ChangeType] = {
    "added": ChangeType.ADDED,
    "new": ChangeType.ADDED,
    "modified": ChangeType.MODIFIED,
    "changed": ChangeType.MODIFIED,
    "removed": ChangeType.DELETED,
    "deleted": ChangeType.DELETED,
    "renamed": ChangeType.RENAMED,
    "copied": ChangeType.COPIED,
}

_PR_STATES = {
    "open": PullRequestState.OPEN,
    "closed": PullRequestState.CLOSED,
    "merged": PullRequestState.MERGED,
}

_TREE_TYPES: dict[str, TreeEntryType] = {
    "blob": TreeEntryType.FILE,
    "file": TreeEntryType.FILE,
    "tree": TreeEntryType.DIRECTORY,
    "dir": TreeEntryType.DIRECTORY,
    "commit": TreeEntryType.SUBMODULE,
    "submodule": TreeEntryType.SUBMODULE,
}


def _parse_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _line_to_position(patch: str, line: int, side: ReviewSide = ReviewSide.NEW) -> int:
    """把文件行号换算成 Gitee 评论所需的 diff position。

    Gitee 的 ``position`` 是「目标行在 patch 文本中的序号（1-based）」，
    从 patch 第一行（通常是 ``@@`` 头）开始计数，而不是文件行号。

    Args:
        patch: 平台返回的 unified diff 文本（可能以 ``@@`` 开头）。
        line: 文件行号（1-based）。
        side: 定位到新侧（新增/上下文行）还是旧侧（删除行）。

    Raises:
        ResourceNotFoundError: 目标行不在该 patch 的可见范围内。
    """
    old_line = 0
    new_line = 0
    position = 0
    for raw in patch.splitlines():
        position += 1
        if raw.startswith("@@"):
            header = raw[2:].split("@@", 1)[0].strip()
            for part in header.split(" "):
                if part.startswith("-"):
                    old_line = _as_int(part[1:].split(",")[0])
                elif part.startswith("+"):
                    new_line = _as_int(part[1:].split(",")[0])
            continue
        if raw.startswith("\\"):  # \ No newline at end of file
            continue

        current = new_line if side is ReviewSide.NEW else old_line
        if current == line:
            return position

        if raw.startswith("-"):
            old_line += 1
        elif raw.startswith("+"):
            new_line += 1
        else:
            old_line += 1
            new_line += 1

    raise ResourceNotFoundError(f"行 {line} 不在 diff 范围内，无法在 Gitee 上定位评论")


class GiteeGitService(GitService):
    """基于 Gitee OpenAPI v5 的实现。

    Args:
        repository: 目标仓库（``owner`` 为个人/组织/企业的空间地址）。
        credentials: 凭证，``token`` 优先于环境变量。
        token: 显式指定 access token。
        base_url: 自建 Gitee 实例的 API 根地址，默认 ``https://gitee.com/api/v5``。
        timeout: 单次 HTTP 请求超时（秒）。
        max_retries: 429/5xx/网络错误的额外重试次数。
    """

    platform = "gitee"

    def __init__(
        self,
        repository: RepositoryRef,
        credentials: GitCredentials | None = None,
        *,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        resolved_token = (
            token
            or (credentials.token if credentials else None)
            or os.environ.get(TOKEN_ENV, "")
        ).strip()
        if not resolved_token:
            raise AuthenticationError(
                f"缺少 Gitee access token：请设置环境变量 {TOKEN_ENV} 或传入 token"
            )

        super().__init__(
            repository,
            credentials or GitCredentials(token=resolved_token),
            base_url=base_url,
        )
        self._api_base = (base_url or DEFAULT_API_BASE).rstrip("/")
        self._max_retries = max(0, max_retries)
        self._default_branch_cache: str | None = None
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "Authorization": f"token {resolved_token}",
                "Accept": "application/json",
                "User-Agent": "kb-code-review",
            },
        )

    # ------------------------------------------------------------------ 生命周期

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ HTTP

    @property
    def _repo(self) -> str:
        ref = self.repository_ref
        return f"/repos/{quote(ref.owner, safe='')}/{quote(ref.name, safe='')}"

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error")
            if message:
                return str(message)
        return str(payload)[:200]

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * (2**attempt), 8.0)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        url = path if path.startswith("http") else f"{self._api_base}{path}"
        last_error = "未知错误"

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, url, params=params, data=data)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt >= self._max_retries:
                    raise GitServiceError(f"请求 Gitee 失败：{exc}") from exc
                await asyncio.sleep(self._backoff(attempt))
                continue

            status = response.status_code
            if status < 400:
                if status == 204 or not response.content:
                    return None
                return response.json()

            message = self._error_message(response)
            if status == 404:
                if allow_404:
                    return None
                raise ResourceNotFoundError(f"Gitee 资源不存在：{method} {path}")
            if status == 401:
                raise AuthenticationError(f"Gitee 凭证无效或已过期：{message}")
            if status == 403:
                if "rate" in message.lower() or "limit" in message.lower():
                    last_error = message
                    if attempt < self._max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    raise RateLimitError(f"Gitee 限流：{message}")
                raise AuthenticationError(f"Gitee 拒绝访问（403）：{message}")
            if status == 429:
                last_error = message
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise RateLimitError(f"Gitee 限流：{message}")
            if status >= 500:
                last_error = message
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise GitServiceError(f"Gitee 服务异常（{status}）：{message}")

            raise GitServiceError(f"Gitee 请求失败（{status}）：{message}")

        raise GitServiceError(f"Gitee 请求失败：{last_error}")

    async def _paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        results: list[Any] = []
        page = 1
        while page <= MAX_PAGES:
            batch = await self._request(
                "GET",
                path,
                params={**(params or {}), "page": page, "per_page": MAX_PER_PAGE},
            )
            if not isinstance(batch, list) or not batch:
                break
            results.extend(batch)
            if len(batch) < MAX_PER_PAGE:
                break
            if limit is not None and len(results) >= limit:
                break
            page += 1
        if limit is not None:
            return results[:limit]
        return results

    # ------------------------------------------------------------------ 映射

    @staticmethod
    def _user(data: dict[str, Any] | None) -> User:
        data = data or {}
        return User(
            username=str(data.get("login") or data.get("name") or ""),
            display_name=data.get("name") or None,
            email=data.get("email") or None,
        )

    @staticmethod
    def _parent_sha(parent: Any) -> str:
        if isinstance(parent, dict):
            return str(parent.get("sha") or "")
        return str(parent or "")

    def _commit(self, data: dict[str, Any]) -> Commit:
        raw = data.get("commit")
        raw = raw if isinstance(raw, dict) else {}
        author = raw.get("author") if isinstance(raw.get("author"), dict) else {}
        committer = raw.get("committer") if isinstance(raw.get("committer"), dict) else {}
        platform_author = data.get("author") if isinstance(data.get("author"), dict) else {}

        return Commit(
            sha=str(data.get("sha") or ""),
            message=str(raw.get("message") or data.get("message") or "").rstrip("\n"),
            author=User(
                username=str(platform_author.get("login") or author.get("name") or ""),
                display_name=platform_author.get("name") or author.get("name") or None,
                email=author.get("email") or platform_author.get("email") or None,
            ),
            parent_shas=tuple(
                sha for sha in (self._parent_sha(p) for p in data.get("parents") or []) if sha
            ),
            authored_at=_parse_datetime(author.get("date")),
            committed_at=_parse_datetime(committer.get("date")),
            web_url=data.get("html_url") or None,
        )

    def _branch(self, data: dict[str, Any]) -> Branch:
        commit = data.get("commit")
        sha = commit.get("sha") if isinstance(commit, dict) else commit
        return Branch(
            name=str(data.get("name") or ""),
            commit_sha=str(sha or ""),
            protected=bool(data.get("protected")),
        )

    def _file_diff(self, data: dict[str, Any], *, with_patch: bool) -> FileDiff:
        """兼容两种返回形态。

        - compare 接口：``patch`` 是字符串，``status`` 为 ``added/modified/...``；
        - PR files 接口：``patch`` 是 dict（含 ``diff``/``old_path``/``renamed_file``），
          且 ``status`` 可能为 ``null``、``additions`` 为字符串。
        """
        raw_patch = data.get("patch")
        meta = raw_patch if isinstance(raw_patch, dict) else {}
        patch_text = meta.get("diff") if meta else (raw_patch if isinstance(raw_patch, str) else None)

        filename = str(data.get("filename") or meta.get("new_path") or "")
        old_path = meta.get("old_path") or data.get("previous_filename") or data.get("old_path") or filename
        new_path = meta.get("new_path") or data.get("new_path") or filename

        is_truncated = bool(meta.get("too_large")) or bool(data.get("truncated"))
        status = str(data.get("status") or "").lower()

        if meta.get("new_file") or status in ("added", "new"):
            change_type = ChangeType.ADDED
        elif meta.get("deleted_file") or status in ("removed", "deleted"):
            change_type = ChangeType.DELETED
        elif meta.get("renamed_file") or status in ("renamed", "copied"):
            change_type = ChangeType.RENAMED if status != "copied" else ChangeType.COPIED
        elif status:
            change_type = _CHANGE_TYPES.get(status, ChangeType.UNKNOWN)
        elif patch_text:
            change_type = self._infer_change_type(patch_text)
        else:
            change_type = ChangeType.UNKNOWN

        if change_type is ChangeType.ADDED:
            old_path = None
        elif change_type is ChangeType.DELETED:
            new_path = None

        return FileDiff(
            change_type=change_type,
            old_path=old_path or None,
            new_path=new_path or None,
            patch=patch_text if with_patch else None,
            # Gitee 对二进制文件不返回 patch，用「无 patch 且未被截断」近似判断
            is_binary=patch_text is None and not is_truncated,
            is_truncated=is_truncated,
            additions=_as_int(data.get("additions")),
            deletions=_as_int(data.get("deletions")),
        )

    @staticmethod
    def _infer_change_type(patch: str) -> ChangeType:
        if "new file mode" in patch:
            return ChangeType.ADDED
        if "deleted file mode" in patch:
            return ChangeType.DELETED
        if "rename from" in patch or "rename to" in patch:
            return ChangeType.RENAMED
        return ChangeType.MODIFIED

    def _comment_record(self, data: dict[str, Any]) -> ReviewCommentRecord:
        line = data.get("new_line")
        return ReviewCommentRecord(
            id=str(data.get("id") or ""),
            body=str(data.get("body") or ""),
            author=self._user(data.get("user") if isinstance(data.get("user"), dict) else {}),
            path=data.get("path") or None,
            line=_as_int(line) if line not in (None, "") else None,
            side=ReviewSide.NEW if data.get("path") else None,
            resolved=bool(data.get("resolved")),
            created_at=_parse_datetime(data.get("created_at")),
            web_url=data.get("html_url") or None,
        )

    # ------------------------------------------------------------------ 账号

    async def get_authenticated_user(self) -> User:
        return self._user(await self._request("GET", "/user"))

    # ------------------------------------------------------------------ 仓库与引用

    async def get_repository(self) -> Repository:
        data = await self._request("GET", self._repo)
        namespace = data.get("namespace") if isinstance(data.get("namespace"), dict) else {}
        owner = str(
            namespace.get("path")
            or (data.get("owner") or {}).get("login")
            or self.repository_ref.owner
        )
        name = str(data.get("path") or data.get("name") or self.repository_ref.name)

        html_url = str(data.get("html_url") or "")
        # Gitee 的 Project.html_url 实际返回的是 clone 地址（.git 结尾）
        web_url = html_url[: -len(".git")] if html_url.endswith(".git") else html_url

        return Repository(
            ref=RepositoryRef(owner=owner, name=name),
            default_branch=str(data.get("default_branch") or "master"),
            web_url=web_url,
            clone_url_http=f"{web_url}.git" if web_url else "",
            clone_url_ssh=data.get("ssh_url") or None,
            private=bool(data.get("private")),
            description=str(data.get("description") or "") or None,
        )

    async def _default_branch(self) -> str:
        if self._default_branch_cache is None:
            self._default_branch_cache = (await self.get_repository()).default_branch
        return self._default_branch_cache

    async def _resolve_sha(self, ref: str) -> str:
        ref = ref.strip()
        if _SHA_RE.match(ref.lower()):
            return ref.lower()

        data = await self._request(
            "GET",
            f"{self._repo}/commits/{quote(ref, safe='')}",
            allow_404=True,
        )
        if isinstance(data, dict) and data.get("sha"):
            return str(data["sha"])

        # 分支名带 "/" 时路径形式可能 404，退回用查询参数解析
        commits = await self._request(
            "GET",
            f"{self._repo}/commits",
            params={"sha": ref, "per_page": 1},
        )
        if isinstance(commits, list) and commits and commits[0].get("sha"):
            return str(commits[0]["sha"])
        raise ResourceNotFoundError(f"无法解析 Gitee 引用：{ref}")

    async def get_branch(self, name: str) -> Branch:
        data = await self._request(
            "GET",
            f"{self._repo}/branches/{quote(name, safe='')}",
            allow_404=True,
        )
        if isinstance(data, dict) and data.get("name"):
            return self._branch(data)

        sha = await self._resolve_sha(name)
        return Branch(name=name, commit_sha=sha)

    async def list_branches(self, *, search: str | None = None, limit: int = 100) -> list[Branch]:
        data = await self._paginate(f"{self._repo}/branches", limit=None)
        keyword = search.lower() if search else None
        branches: list[Branch] = []
        for item in data:
            branch = self._branch(item)
            if keyword and keyword not in branch.name.lower():
                continue
            branches.append(branch)
            if len(branches) >= limit:
                break
        return branches

    async def get_commit(self, ref: str) -> Commit:
        data = await self._request(
            "GET",
            f"{self._repo}/commits/{quote(ref, safe='')}",
        )
        return self._commit(data)

    async def list_commits(
        self,
        *,
        ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Commit]:
        params: dict[str, Any] = {}
        if ref:
            params["sha"] = ref
        if since is not None:
            params["since"] = since.isoformat()
        if until is not None:
            params["until"] = until.isoformat()

        items = await self._paginate(f"{self._repo}/commits", params=params, limit=max(1, limit))
        return [self._commit(item) for item in items]

    async def get_merge_base(self, base: str, head: str) -> str:
        base_sha = await self._resolve_sha(base)
        head_sha = await self._resolve_sha(head)
        data = await self._request(
            "GET",
            f"{self._repo}/compare/{base_sha}...{head_sha}",
            params={"straight": "false"},
        )
        merge_base = data.get("merge_base_commit") if isinstance(data, dict) else None
        sha = (merge_base or {}).get("sha")
        if not sha:
            raise GitServiceError(f"{base} 与 {head} 没有共同祖先")
        return str(sha)

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

        data = await self._request(
            "GET",
            f"{self._repo}/compare/{base_sha}...{head_sha}",
            # straight=false 为三点（merge-base）对比，与评审语义一致
            params={"straight": "false" if use_merge_base else "true"},
        )
        data = data if isinstance(data, dict) else {}
        merge_base = data.get("merge_base_commit") if isinstance(data.get("merge_base_commit"), dict) else {}

        files = tuple(
            self._file_diff(item, with_patch=include_patch) for item in data.get("files") or []
        )
        return Diff(
            base_sha=base_sha,
            head_sha=head_sha,
            # 直对比（straight）时 merge base 并未参与计算，与本地实现保持一致返回 None
            merge_base_sha=(
                str(merge_base["sha"]) if use_merge_base and merge_base.get("sha") else None
            ),
            files=files,
            is_truncated=bool(data.get("truncated")),
        )

    async def list_tree(
        self,
        ref: str | None = None,
        *,
        path: str = "",
        recursive: bool = True,
    ) -> list[TreeEntry]:
        target = ref or await self._default_branch()
        clean_path = path.strip("/")

        if not recursive:
            # 非递归用 contents 接口直接列一层，避免拉取整棵大树
            contents_path = f"{self._repo}/contents" + (f"/{clean_path}" if clean_path else "")
            data = await self._request(
                "GET",
                contents_path,
                params={"ref": target},
            )
            if isinstance(data, dict):
                raise GitServiceError(f"{clean_path or '/'} 不是目录")
            return [self._tree_entry_from_content(item) for item in data or []]

        data = await self._request(
            "GET",
            f"{self._repo}/git/trees/{quote(target, safe='')}",
            params={"recursive": 1},
        )
        prefix = f"{clean_path}/" if clean_path else ""
        entries: list[TreeEntry] = []
        for item in (data or {}).get("tree") or []:
            item_path = str(item.get("path") or "")
            if prefix and not item_path.startswith(prefix):
                continue
            entry_type = _TREE_TYPES.get(str(item.get("type") or "").lower())
            if entry_type is None:
                continue
            size = item.get("size")
            entries.append(
                TreeEntry(
                    path=item_path,
                    type=entry_type,
                    sha=str(item.get("sha") or ""),
                    size=_as_int(size) if entry_type is TreeEntryType.FILE and size else None,
                )
            )
        return entries

    def _tree_entry_from_content(self, data: dict[str, Any]) -> TreeEntry:
        entry_type = _TREE_TYPES.get(str(data.get("type") or "").lower(), TreeEntryType.FILE)
        size = data.get("size")
        return TreeEntry(
            path=str(data.get("path") or data.get("name") or ""),
            type=entry_type,
            sha=str(data.get("sha") or ""),
            size=_as_int(size) if entry_type is TreeEntryType.FILE and size else None,
        )

    async def get_file_content(
        self,
        path: str,
        *,
        ref: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> str:
        target = ref or await self._default_branch()
        data = await self._request(
            "GET",
            f"{self._repo}/contents/{quote(path.strip('/'), safe='/')}",
            params={"ref": target},
            allow_404=True,
        )
        if data is None:
            raise ResourceNotFoundError(f"文件不存在：{path}")
        if isinstance(data, list):
            raise GitServiceError(f"{path} 是目录，无法按文件读取")

        encoding = str(data.get("encoding") or "").lower()
        content = data.get("content") or ""
        if encoding == "base64" and content:
            try:
                raw = base64.b64decode(content)
            except (binascii.Error, ValueError) as exc:
                raise GitServiceError(f"解析 Gitee 文件内容失败：{path}") from exc
        else:
            # 超大文件 Gitee 不返回 content，退回 download_url
            download_url = data.get("download_url")
            if not download_url:
                raise GitServiceError(f"Gitee 未返回 {path} 的内容")
            response = await self._client.get(str(download_url))
            if response.status_code >= 400:
                raise GitServiceError(
                    f"下载文件失败（{response.status_code}）：{path}"
                )
            raw = response.content

        if b"\0" in raw:
            raise UnsupportedFeatureError(f"{path} 是二进制文件，无法按文本读取")

        text = raw.decode("utf-8", "replace")
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

    # ------------------------------------------------------------------ 合并请求

    def _pull_request(self, data: dict[str, Any]) -> PullRequest:
        head = data.get("head") if isinstance(data.get("head"), dict) else {}
        base = data.get("base") if isinstance(data.get("base"), dict) else {}
        state = str(data.get("state") or "").lower()
        return PullRequest(
            number=int(data.get("number") or 0),
            title=str(data.get("title") or ""),
            state=_PR_STATES.get(state, PullRequestState.CLOSED),
            author=self._user(data.get("user") if isinstance(data.get("user"), dict) else {}),
            source_branch=str(head.get("ref") or ""),
            target_branch=str(base.get("ref") or ""),
            head_sha=str(head.get("sha") or ""),
            base_sha=str(base.get("sha") or ""),
            web_url=str(data.get("html_url") or ""),
            description=str(data.get("body") or "") or None,
            draft=bool(data.get("draft")),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )

    async def get_pull_request(self, number: int) -> PullRequest:
        data = await self._request("GET", f"{self._repo}/pulls/{number}")
        return self._pull_request(data)

    async def list_pull_requests(
        self,
        *,
        state: PullRequestState = PullRequestState.OPEN,
        source_branch: str | None = None,
        limit: int = 50,
    ) -> list[PullRequest]:
        params: dict[str, Any] = {"state": state.value}
        if source_branch:
            params["head"] = source_branch
        items = await self._paginate(f"{self._repo}/pulls", params=params, limit=max(1, limit))
        return [self._pull_request(item) for item in items]

    async def get_pull_request_files(self, number: int, *, include_patch: bool = True) -> tuple[FileDiff, ...]:
        """PR 的文件变更。Gitee 单次最多返回 300 条 diff。"""
        items = await self._request("GET", f"{self._repo}/pulls/{number}/files")
        return tuple(
            self._file_diff(item, with_patch=include_patch) for item in items or []
        )

    async def get_pull_request_diff(self, number: int, *, include_patch: bool = True) -> Diff:
        pr = await self.get_pull_request(number)
        files = await self.get_pull_request_files(number, include_patch=include_patch)
        return Diff(
            base_sha=pr.base_sha,
            head_sha=pr.head_sha,
            files=files,
            # Gitee 没有单独的 merge-base 端点，PR 文件接口本身就是三点 diff
            merge_base_sha=None,
            is_truncated=len(files) >= PR_FILES_LIMIT,
        )

    async def get_pull_request_commits(self, number: int) -> list[Commit]:
        items = await self._request("GET", f"{self._repo}/pulls/{number}/commits")
        return [self._commit(item) for item in items or []]

    # ------------------------------------------------------------------ 评审回写

    async def submit_review(
        self,
        number: int,
        submission: ReviewSubmission,
        *,
        commit_sha: str | None = None,
    ) -> ReviewResult:
        if submission.event is ReviewEvent.REQUEST_CHANGES:
            raise UnsupportedFeatureError(
                "Gitee 没有“要求修改”的评审结论，请改用 COMMENT 或 APPROVE"
            )

        pr = await self.get_pull_request(number)
        head_sha = commit_sha or pr.head_sha
        comment_ids: list[str] = []

        patches: dict[str, str] = {}
        if any(comment.line is not None for comment in submission.comments):
            files = await self.get_pull_request_files(number)
            patches = {file.path: file.patch or "" for file in files}

        for comment in submission.comments:
            payload: dict[str, Any] = {"body": comment.body, "commit_id": head_sha}
            if comment.line is not None:
                patch = patches.get(comment.path)
                if patch is None:
                    raise ResourceNotFoundError(
                        f"PR #{number} 的变更中不存在文件：{comment.path}"
                    )
                payload["path"] = comment.path
                payload["position"] = _line_to_position(patch, comment.line, comment.side)
            else:
                # Gitee 的文件级评论需要 path+position，退化为带文件名的普通评论
                payload["body"] = f"**{comment.path}**\n\n{comment.body}"

            created = await self._request(
                "POST", f"{self._repo}/pulls/{number}/comments", data=payload
            )
            if isinstance(created, dict) and created.get("id"):
                comment_ids.append(str(created["id"]))

        if submission.body:
            created = await self._request(
                "POST",
                f"{self._repo}/pulls/{number}/comments",
                data={"body": submission.body},
            )
            if isinstance(created, dict) and created.get("id"):
                comment_ids.append(str(created["id"]))

        if submission.event is ReviewEvent.APPROVE:
            # Gitee 的“处理 PR 审查”接口即审查通过
            await self._request("POST", f"{self._repo}/pulls/{number}/review")

        return ReviewResult(comment_ids=tuple(comment_ids), web_url=pr.web_url)

    async def list_review_comments(self, number: int) -> list[ReviewCommentRecord]:
        items = await self._paginate(f"{self._repo}/pulls/{number}/comments")
        return [self._comment_record(item) for item in items]

    async def reply_to_review_comment(
        self,
        number: int,
        comment_id: str,
        body: str,
    ) -> ReviewCommentRecord:
        original = await self._request(
            "GET",
            f"{self._repo}/pulls/comments/{quote(str(comment_id), safe='')}",
            allow_404=True,
        )
        if not isinstance(original, dict):
            raise ResourceNotFoundError(f"评论不存在：{comment_id}")

        payload: dict[str, Any] = {
            "body": body,
            "in_reply_to_id": comment_id,
        }
        if original.get("path"):
            payload["path"] = original["path"]
            payload["position"] = original.get("position")
            payload["commit_id"] = original.get("commit_id")

        created = await self._request(
            "POST", f"{self._repo}/pulls/{number}/comments", data=payload
        )
        if isinstance(created, dict) and created.get("id"):
            return self._comment_record(created)
        raise GitServiceError(f"回复评论失败：{comment_id}")

    # ------------------------------------------------------------------ 可选能力

    async def create_branch(self, name: str, *, from_ref: str | None = None) -> Branch:
        start = from_ref or await self._default_branch()
        # 注意 Gitee 的参数命名：refs 是起点，branch_name 是新分支
        await self._request(
            "POST",
            f"{self._repo}/branches",
            data={"refs": start, "branch_name": name},
        )
        return await self.get_branch(name)
