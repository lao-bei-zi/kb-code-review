"""与代码托管平台无关的 Git 领域模型。

这些模型是各平台（GitHub / GitLab / Gitee / Gitea ...）原始响应的归一化结果：
上层评审逻辑只依赖这里的字段，平台差异全部收敛在 ``GitService`` 实现里。

约定：
- 时间统一为 UTC、带时区的 ``datetime``；
- 分支名 / 路径不带前缀（路径不带开头的 ``/``）；
- 行号采用「文件内行号」（1-based），与各平台评论接口语义一致；
- 所有模型不可变（``frozen=True``），可直接作为缓存键或并发共享。
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Branch",
    "ChangeType",
    "Commit",
    "Diff",
    "FileDiff",
    "GitModel",
    "PullRequest",
    "PullRequestState",
    "Repository",
    "RepositoryRef",
    "ReviewComment",
    "ReviewCommentRecord",
    "ReviewEvent",
    "ReviewResult",
    "ReviewSide",
    "ReviewSubmission",
    "TreeEntry",
    "TreeEntryType",
    "User",
]


class GitModel(BaseModel):
    """所有领域模型的基类。"""

    model_config = ConfigDict(frozen=True)


class ChangeType(StrEnum):
    """文件变更类型。"""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    TYPE_CHANGED = "type_changed"
    UNKNOWN = "unknown"


class PullRequestState(StrEnum):
    """合并请求状态。"""

    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class ReviewEvent(StrEnum):
    """提交评审时的动作，对应 GitHub 的 event / GitLab 的评论 + 审批。"""

    COMMENT = "comment"
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"


class ReviewSide(StrEnum):
    """评论定位到 diff 的新侧还是旧侧。

    ``NEW``：行号取 head 版本文件的行号；
    ``OLD``：行号取 base 版本文件的行号（用于删除行上的评论）。
    """

    NEW = "new"
    OLD = "old"


class TreeEntryType(StrEnum):
    """仓库目录项类型。"""

    FILE = "file"
    DIRECTORY = "directory"
    SUBMODULE = "submodule"


class User(GitModel):
    """平台账号。"""

    username: str
    display_name: str | None = None
    email: str | None = None


class RepositoryRef(GitModel):
    """仓库定位信息，作为 ``GitService`` 的必填配置。"""

    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class Repository(GitModel):
    """仓库元信息。"""

    ref: RepositoryRef
    default_branch: str
    web_url: str
    clone_url_http: str
    clone_url_ssh: str | None = None
    private: bool = False
    description: str | None = None


class Branch(GitModel):
    """分支。"""

    name: str
    commit_sha: str
    protected: bool = False


class Commit(GitModel):
    """提交。"""

    sha: str
    message: str
    author: User
    parent_shas: tuple[str, ...] = ()
    authored_at: datetime | None = None
    committed_at: datetime | None = None
    web_url: str | None = None


class TreeEntry(GitModel):
    """仓库目录项。"""

    path: str
    type: TreeEntryType
    sha: str
    size: int | None = None


class FileDiff(GitModel):
    """单个文件的变更。

    ``patch`` 为 unified diff 文本（不含文件头）；二进制、过大文件或被平台截断时为
    ``None``，此时 ``is_binary`` / ``is_truncated`` 会给出原因。
    """

    change_type: ChangeType
    old_path: str | None = None
    new_path: str | None = None
    patch: str | None = None
    is_binary: bool = False
    is_truncated: bool = False
    additions: int = 0
    deletions: int = 0

    @property
    def path(self) -> str:
        """变更后的路径；删除文件时回退到旧路径。"""
        return self.new_path or self.old_path or ""


class Diff(GitModel):
    """两个版本之间的差异集合，也是 AI 评审的主要输入。

    ``is_truncated`` 表示平台对文件数量或 patch 体积做了截断，评审方需要据此
    降低置信度或补充拉取。
    """

    base_sha: str
    head_sha: str
    files: tuple[FileDiff, ...] = ()
    merge_base_sha: str | None = None
    is_truncated: bool = False


class PullRequest(GitModel):
    """合并请求 / Pull Request。"""

    number: int
    title: str
    state: PullRequestState
    author: User
    source_branch: str
    target_branch: str
    head_sha: str
    base_sha: str
    web_url: str
    description: str | None = None
    draft: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ReviewComment(GitModel):
    """待提交的单条评论（增量/行级）。

    ``line`` 为 ``None`` 表示文件级评论，不定位到具体行；
    ``start_line`` 为 ``None`` 表示单行评论。
    """

    body: str
    path: str
    line: int | None = None
    side: ReviewSide = ReviewSide.NEW
    start_line: int | None = None
    start_side: ReviewSide | None = None


class ReviewSubmission(GitModel):
    """一次评审提交：一个总体评论 + 若干行级评论。"""

    body: str | None = None
    comments: tuple[ReviewComment, ...] = ()
    event: ReviewEvent = ReviewEvent.COMMENT


class ReviewResult(GitModel):
    """评审提交结果。"""

    review_id: str | None = None
    comment_ids: tuple[str, ...] = ()
    web_url: str | None = None


class ReviewCommentRecord(GitModel):
    """已存在的评论，用于增量评审时去重 / 追加回复。"""

    id: str
    body: str
    author: User
    path: str | None = None
    line: int | None = None
    side: ReviewSide | None = None
    resolved: bool = False
    created_at: datetime | None = None
    web_url: str | None = None
