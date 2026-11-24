"""``LocalGitService`` 的测试。

- 确定性用例：在 ``tmp_path`` 里构造仓库，覆盖重命名/删除/二进制/三点 diff 等边界；
- 集成用例：只读访问 ``~/projects/linux-c``，仓库不存在时自动 skip。

测试函数均为原生协程，由 pytest-asyncio 的 ``asyncio_mode = "auto"`` 自动收集。
"""

import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from kb_code_review.git.base import (
    GitCredentials,
    GitServiceError,
    ResourceNotFoundError,
    UnsupportedFeatureError,
)
from kb_code_review.git.models import (
    ChangeType,
    PullRequestState,
    ReviewSubmission,
    TreeEntryType,
)
from kb_code_review.git.providers.local import LocalGitService

LINUX_C = Path.home() / "projects" / "linux-c"


# --------------------------------------------------------------------------- 工具


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Tester")
    git(repo, "config", "user.email", "tester@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    return repo


def parse_name_status(output: str) -> list[str]:
    """解析 ``git diff --name-status -z``，返回变更后的路径（删除文件返回旧路径）。"""
    tokens = output.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        if not status:
            index += 1
            continue
        index += 1
        if status[0] in ("R", "C"):
            paths.append(tokens[index + 1])
            index += 2
        else:
            paths.append(tokens[index])
            index += 1
    return paths


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def scratch_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """构造一个有分叉、含 rename/delete/binary 变更的仓库。

    历史::

        init ──┬── main only            (main)
               └── feature work         (feature/x)
    """
    repo = init_repo(tmp_path_factory.mktemp("scratch") / "repo")
    write(repo / "pkg" / "a.py", "a1\na2\na3\n")
    write(repo / "pkg" / "b.py", "print('b')\n")
    write(repo / "README.md", "".join(f"line{i}\n" for i in range(1, 11)))
    (repo / "logo.bin").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00data")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    git(repo, "branch", "feature/x")

    write(repo / "main_only.txt", "main\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main only")

    git(repo, "checkout", "-q", "feature/x")
    git(repo, "mv", "pkg/a.py", "pkg/c.py")
    git(repo, "rm", "-q", "pkg/b.py")
    write(repo / "pkg" / "c.py", "a1\na2\na3\na4\n")
    write(repo / "new.txt", "new\n")
    (repo / "logo.bin").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00changed")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature work")
    git(repo, "checkout", "-q", "main")
    return repo


@pytest.fixture
def service(scratch_repo: Path) -> LocalGitService:
    return LocalGitService(scratch_repo)


@pytest.fixture
def linux_c_repo() -> Path:
    if not LINUX_C.exists():
        pytest.skip(f"缺少测试仓库：{LINUX_C}")
    try:
        LocalGitService(LINUX_C)
    except GitServiceError as exc:
        pytest.skip(f"{LINUX_C} 不是可用的 Git 仓库：{exc}")
    return LINUX_C


# --------------------------------------------------------------------------- 构造与仓库信息


async def test_platform_and_default_credentials(service: LocalGitService) -> None:
    assert service.platform == "local"
    assert service.repository_ref.name == "repo"

    credentials = GitCredentials()
    assert credentials.token is None
    assert credentials.username is None
    assert credentials.password is None


async def test_workdir_subdirectory_resolves_toplevel(scratch_repo: Path) -> None:
    nested = LocalGitService(scratch_repo / "pkg")
    assert nested.repository_ref.name == "repo"
    assert (await nested.get_file_content("README.md")).startswith("line1")


async def test_repository_without_remote_falls_back_to_file_url(
    service: LocalGitService,
) -> None:
    repo = await service.get_repository()
    assert repo.ref.name == "repo"
    assert repo.default_branch == "main"
    assert repo.web_url.startswith("file://")
    assert repo.web_url.endswith("/repo")
    assert repo.clone_url_http.startswith("file://")
    assert repo.clone_url_ssh is None
    assert repo.private is False


@pytest.mark.parametrize(
    ("remote_url", "owner", "clone_url_http", "clone_url_ssh"),
    [
        (
            "https://github.com/acme/kb.git",
            "acme",
            "https://github.com/acme/kb.git",
            None,
        ),
        (
            "git@github.com:acme/kb.git",
            "acme",
            "https://github.com/acme/kb.git",
            "git@github.com:acme/kb.git",
        ),
        (
            "ssh://git@gitlab.example.com:2222/group/sub/kb.git",
            "group/sub",
            "https://gitlab.example.com/group/sub/kb.git",
            "ssh://git@gitlab.example.com:2222/group/sub/kb.git",
        ),
    ],
)
async def test_repository_info_from_remote(
    tmp_path: Path,
    remote_url: str,
    owner: str,
    clone_url_http: str,
    clone_url_ssh: str | None,
) -> None:
    repo_path = init_repo(tmp_path / "demo")
    write(repo_path / "a.txt", "a\n")
    git(repo_path, "add", "-A")
    git(repo_path, "commit", "-q", "-m", "init")
    git(repo_path, "remote", "add", "origin", remote_url)

    repo = await LocalGitService(repo_path).get_repository()
    assert repo.ref.owner == owner
    assert repo.ref.name == "kb"
    assert repo.ref.full_name == f"{owner}/kb"
    assert repo.web_url.startswith("https://")
    assert repo.clone_url_http == clone_url_http
    assert repo.clone_url_ssh == clone_url_ssh


async def test_not_a_git_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(GitServiceError, match="不是 Git 仓库"):
        LocalGitService(tmp_path)


# --------------------------------------------------------------------------- 引用


async def test_get_branch_and_missing(service: LocalGitService, scratch_repo: Path) -> None:
    branch = await service.get_branch("feature/x")
    assert branch.name == "feature/x"
    assert branch.commit_sha == git(scratch_repo, "rev-parse", "feature/x").strip()

    with pytest.raises(ResourceNotFoundError, match="分支不存在"):
        await service.get_branch("no/such/branch")


async def test_list_branches_and_search(service: LocalGitService) -> None:
    names = {branch.name for branch in await service.list_branches()}
    assert names == {"main", "feature/x"}

    assert [b.name for b in await service.list_branches(search="FEAT")] == ["feature/x"]
    assert len(await service.list_branches(limit=1)) == 1


async def test_get_commit_fields(service: LocalGitService, scratch_repo: Path) -> None:
    commit = await service.get_commit("feature/x")

    assert commit.sha == git(scratch_repo, "rev-parse", "feature/x").strip()
    assert commit.message == "feature work"
    assert commit.author.email == "tester@example.com"
    assert commit.author.username == "Tester"
    assert commit.parent_shas == (
        git(scratch_repo, "rev-parse", "feature/x~1").strip(),
    )
    assert commit.authored_at is not None
    assert commit.authored_at.tzinfo is not None


async def test_list_commits_limit_ref_and_since(
    service: LocalGitService, scratch_repo: Path
) -> None:
    commits = await service.list_commits(ref="feature/x", limit=2)
    assert [c.sha for c in commits] == git(
        scratch_repo, "log", "--format=%H", "-2", "feature/x"
    ).split()
    assert all(c.author.email == "tester@example.com" for c in commits)

    future = datetime.now(timezone.utc) + timedelta(days=1)
    assert await service.list_commits(since=future) == []


async def test_missing_ref_raises(service: LocalGitService) -> None:
    with pytest.raises(ResourceNotFoundError):
        await service.get_commit("does-not-exist")


async def test_get_merge_base(service: LocalGitService, scratch_repo: Path) -> None:
    merge_base = await service.get_merge_base("main", "feature/x")
    assert merge_base == git(scratch_repo, "merge-base", "main", "feature/x").strip()


# --------------------------------------------------------------------------- diff


async def test_compare_three_dot_excludes_target_branch(
    service: LocalGitService, scratch_repo: Path
) -> None:
    diff = await service.compare("main", "feature/x")

    assert diff.base_sha == git(scratch_repo, "rev-parse", "main").strip()
    assert diff.head_sha == git(scratch_repo, "rev-parse", "feature/x").strip()
    assert diff.merge_base_sha == git(scratch_repo, "merge-base", "main", "feature/x").strip()
    assert diff.is_truncated is False

    paths = {file.path for file in diff.files}
    assert paths == {"pkg/b.py", "pkg/c.py", "new.txt", "logo.bin"}
    assert "main_only.txt" not in paths


async def test_compare_two_dot_includes_target_branch(service: LocalGitService) -> None:
    diff = await service.compare("main", "feature/x", use_merge_base=False)
    assert diff.merge_base_sha is None
    assert "main_only.txt" in {file.path for file in diff.files}


async def test_compare_change_types_and_patches(service: LocalGitService) -> None:
    files = {file.path: file for file in (await service.compare("main", "feature/x")).files}

    renamed = files["pkg/c.py"]
    assert renamed.change_type is ChangeType.RENAMED
    assert renamed.old_path == "pkg/a.py"
    assert renamed.new_path == "pkg/c.py"
    assert renamed.additions == 1
    assert renamed.deletions == 0
    assert renamed.patch is not None
    assert "rename from pkg/a.py" in renamed.patch
    assert "rename to pkg/c.py" in renamed.patch

    deleted = files["pkg/b.py"]
    assert deleted.change_type is ChangeType.DELETED
    assert deleted.old_path == "pkg/b.py"
    assert deleted.new_path is None
    assert deleted.deletions == 1
    assert "deleted file mode" in (deleted.patch or "")

    added = files["new.txt"]
    assert added.change_type is ChangeType.ADDED
    assert added.old_path is None
    assert added.new_path == "new.txt"
    assert added.additions == 1
    assert "+new" in (added.patch or "")

    binary = files["logo.bin"]
    assert binary.change_type is ChangeType.MODIFIED
    assert binary.is_binary is True
    assert binary.patch is None
    assert binary.additions == 0
    assert binary.deletions == 0


async def test_compare_include_patch_false(service: LocalGitService) -> None:
    diff = await service.compare("main", "feature/x", include_patch=False)
    assert diff.files
    assert all(file.patch is None for file in diff.files)


async def test_compare_identical_refs_returns_empty(service: LocalGitService) -> None:
    diff = await service.compare("main", "main")
    assert diff.files == ()
    assert diff.base_sha == diff.head_sha == diff.merge_base_sha


async def test_merge_base_without_common_ancestor(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "orphan")
    write(repo / "a.txt", "a\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "checkout", "-q", "--orphan", "other")
    git(repo, "rm", "-q", "-rf", ".")
    write(repo / "b.txt", "b\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")

    with pytest.raises(GitServiceError, match="没有共同祖先"):
        await LocalGitService(repo).get_merge_base("main", "other")


# --------------------------------------------------------------------------- 目录树与文件


async def test_list_tree_root_and_path_filter(service: LocalGitService) -> None:
    root = {entry.path: entry for entry in await service.list_tree(recursive=False)}
    assert set(root) == {"README.md", "logo.bin", "main_only.txt", "pkg"}
    assert root["pkg"].type is TreeEntryType.DIRECTORY
    assert root["pkg"].size is None
    assert root["README.md"].type is TreeEntryType.FILE
    assert root["README.md"].size == len("".join(f"line{i}\n" for i in range(1, 11)))

    recursive = {entry.path for entry in await service.list_tree(path="pkg")}
    assert recursive == {"pkg/a.py", "pkg/b.py"}


async def test_get_file_content_full_and_line_range(service: LocalGitService) -> None:
    full = await service.get_file_content("README.md")
    assert full == "".join(f"line{i}\n" for i in range(1, 11))

    assert await service.get_file_content("README.md", line_start=3, line_end=5) == (
        "line3\nline4\nline5\n"
    )
    assert await service.get_file_content("README.md", line_start=9) == "line9\nline10\n"
    assert await service.get_file_content("README.md", line_start=99) == ""
    assert await service.get_file_content("README.md", ref="feature/x") == full

    with pytest.raises(ValueError):
        await service.get_file_content("README.md", line_start=5, line_end=2)


async def test_get_file_content_errors(service: LocalGitService) -> None:
    with pytest.raises(UnsupportedFeatureError, match="二进制"):
        await service.get_file_content("logo.bin")

    with pytest.raises(ResourceNotFoundError):
        await service.get_file_content("no/such/file.txt")


# --------------------------------------------------------------------------- 平台能力


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(lambda s: s.get_authenticated_user(), id="authenticated-user"),
        pytest.param(lambda s: s.get_pull_request(1), id="pull-request"),
        pytest.param(lambda s: s.list_pull_requests(), id="pull-requests"),
        pytest.param(lambda s: s.get_pull_request_diff(1), id="pull-request-diff"),
        pytest.param(lambda s: s.get_pull_request_commits(1), id="pull-request-commits"),
        pytest.param(
            lambda s: s.submit_review(1, ReviewSubmission(body="summary")), id="submit-review"
        ),
        pytest.param(lambda s: s.list_review_comments(1), id="list-review-comments"),
        pytest.param(
            lambda s: s.reply_to_review_comment(1, "42", "reply"), id="reply-comment"
        ),
        pytest.param(lambda s: s.update_commit_status("HEAD", "success"), id="commit-status"),
    ],
)
async def test_platform_only_apis_raise(
    service: LocalGitService, invoke: Callable[[LocalGitService], Any]
) -> None:
    with pytest.raises(UnsupportedFeatureError):
        await invoke(service)


async def test_list_pull_requests_state_argument_is_ignored(
    service: LocalGitService,
) -> None:
    with pytest.raises(UnsupportedFeatureError):
        await service.list_pull_requests(state=PullRequestState.MERGED)


# --------------------------------------------------------------------------- 可选能力


async def test_create_branch_from_ref(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    write(repo / "a.txt", "a\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    service = LocalGitService(repo)

    branch = await service.create_branch("release/1.0", from_ref="main")
    assert branch.name == "release/1.0"
    assert branch.commit_sha == git(repo, "rev-parse", "main").strip()
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"

    with pytest.raises(GitServiceError):
        await service.create_branch("release/1.0", from_ref="main")


# --------------------------------------------------------------------------- 集成：linux-c


async def test_linux_c_repository_info(linux_c_repo: Path) -> None:
    service = LocalGitService(linux_c_repo)
    repo = await service.get_repository()
    assert repo.ref.name == "linux-c"
    assert repo.ref.owner
    assert repo.web_url.startswith("https://")
    assert repo.default_branch
    assert repo.default_branch in {b.name for b in await service.list_branches()}


async def test_linux_c_commit_history(linux_c_repo: Path) -> None:
    service = LocalGitService(linux_c_repo)
    commits = await service.list_commits(limit=5)
    assert [c.sha for c in commits] == git(linux_c_repo, "log", "--format=%H", "-5").split()
    assert all(c.author.email for c in commits)


async def test_linux_c_diff_matches_git_cli(linux_c_repo: Path) -> None:
    service = LocalGitService(linux_c_repo)
    diff = await service.compare("HEAD~1", "HEAD", use_merge_base=False, include_patch=False)

    expected_paths = parse_name_status(
        git(linux_c_repo, "diff", "--name-status", "-z", "--find-renames", "HEAD~1", "HEAD")
    )
    assert {file.path for file in diff.files} == set(expected_paths)

    numstat = git(linux_c_repo, "diff", "--numstat", "--find-renames", "HEAD~1", "HEAD")
    expected_additions = expected_deletions = 0
    for line in numstat.splitlines():
        additions, deletions, _ = line.split("\t", 2)
        if additions != "-":
            expected_additions += int(additions)
            expected_deletions += int(deletions)
    assert sum(file.additions for file in diff.files) == expected_additions
    assert sum(file.deletions for file in diff.files) == expected_deletions


async def test_linux_c_tree_handles_non_ascii_paths(linux_c_repo: Path) -> None:
    service = LocalGitService(linux_c_repo)
    entries = await service.list_tree(recursive=False)
    paths = {entry.path for entry in entries}
    assert "README.md" in paths
    assert "Http协议" in paths

    readme = await service.get_file_content("README.md")
    assert readme.strip()


async def test_linux_c_branch_lookup(linux_c_repo: Path) -> None:
    service = LocalGitService(linux_c_repo)
    current = git(linux_c_repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    branch = await service.get_branch(current)
    assert branch.commit_sha == git(linux_c_repo, "rev-parse", current).strip()
