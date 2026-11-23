"""代码托管平台 Git 服务的抽象基类。

设计目标
--------
1. 上层（评审调度、diff 解析、LLM 分析）只依赖 ``GitService``，不感知平台差异；
2. 接入新平台 = 新增一个子类 + 注册 ``platform``，调用方零改动；
3. 接口覆盖一次 AI 评审的完整闭环：
   ``拉取 PR -> 计算 diff -> 读取源码上下文 -> 回写评审评论``。

约定
----
- 所有 I/O 方法为 ``async``：平台侧都是 HTTP 调用，评审流程本身也是异步并发的；
  本地 git CLI 操作请在实现里用 ``asyncio.to_thread`` 包装。
- 实现方负责重试、限流与分页，调用方只关心「给我全量结果」。
- 无法支持的能力抛 ``UnsupportedFeatureError``，而不是静默返回空值。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Self

from .models import (
    Branch,
    Commit,
    Diff,
    GitModel,
    PullRequest,
    PullRequestState,
    Repository,
    RepositoryRef,
    ReviewCommentRecord,
    ReviewResult,
    ReviewSubmission,
    TreeEntry,
    User,
)

__all__ = [
    "AuthenticationError",
    "GitCredentials",
    "GitService",
    "GitServiceError",
    "RateLimitError",
    "ResourceNotFoundError",
    "UnsupportedFeatureError",
]


class GitServiceError(Exception):
    """Git 服务异常基类。"""


class AuthenticationError(GitServiceError):
    """凭证无效、无权限或过期。"""


class ResourceNotFoundError(GitServiceError):
    """仓库、分支、提交、PR、文件等资源不存在。"""


class RateLimitError(GitServiceError):
    """触发平台限流，且重试后仍失败。"""


class UnsupportedFeatureError(GitServiceError):
    """当前平台/实现不支持该能力。"""


class GitCredentials(GitModel):
    """访问凭证。具体用哪个字段由实现决定（如 GitHub 只用 ``token``）。"""

    token: str | None = None
    username: str | None = None
    password: str | None = None


class GitService(ABC):
    """代码托管平台的 Git 操作抽象基类。

    一个实例绑定一个仓库（``RepositoryRef``），避免每个方法都传 owner/name。
    """

    platform: ClassVar[str] = ""
    """平台标识（``github`` / ``gitlab`` / ``gitee`` ...），子类必须覆盖。"""

    def __init__(
        self,
        repository: RepositoryRef,
        credentials: GitCredentials,
        *,
        base_url: str | None = None,
    ) -> None:
        """
        Args:
            repository: 目标仓库。
            credentials: 平台凭证。
            base_url: 自建实例（GitLab CE、Gitea）的 API 根地址，``None`` 用平台默认值。
        """
        self._repository = repository
        self._credentials = credentials
        self._base_url = base_url

    # ------------------------------------------------------------------ 生命周期

    @property
    def repository_ref(self) -> RepositoryRef:
        return self._repository

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """释放底层 HTTP 连接等资源。默认无操作，需要时由子类覆盖。"""

    # ------------------------------------------------------------------ 账号

    @abstractmethod
    async def get_authenticated_user(self) -> User:
        """返回当前凭证对应的账号，用于启动时的连通性/权限自检。"""

    # ------------------------------------------------------------------ 仓库与引用

    @abstractmethod
    async def get_repository(self) -> Repository:
        """获取仓库元信息（含默认分支）。"""

    @abstractmethod
    async def get_branch(self, name: str) -> Branch:
        """按名称获取分支。不存在时抛 ``ResourceNotFoundError``。"""

    @abstractmethod
    async def list_branches(self, *, search: str | None = None, limit: int = 100) -> list[Branch]:
        """列出分支，实现方内部处理分页。"""

    @abstractmethod
    async def get_commit(self, ref: str) -> Commit:
        """按 sha / 分支名 / 标签获取单个提交。"""

    @abstractmethod
    async def list_commits(
        self,
        *,
        ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Commit]:
        """按时间范围倒序列出提交。``ref`` 为 ``None`` 时使用默认分支。"""

    @abstractmethod
    async def get_merge_base(self, base: str, head: str) -> str:
        """求两个引用的 merge base，返回其 sha。

        无专用端点时可用 ``compare`` 返回的 ``merge_base_sha`` 实现。
        """

    # ------------------------------------------------------------------ 变更与源码

    @abstractmethod
    async def compare(
        self,
        base: str,
        head: str,
        *,
        use_merge_base: bool = True,
        include_patch: bool = True,
    ) -> Diff:
        """比较两个引用。

        Args:
            base: 基线引用（分支 / 标签 / sha）。
            head: 目标引用。
            use_merge_base: ``True`` 时按三点 diff（base...head）计算，
                这是评审 PR 的正确语义——避免把目标分支的新提交算进变更。
            include_patch: ``False`` 时只返回变更文件列表，用于大 PR 的预筛。
        """

    @abstractmethod
    async def list_tree(
        self,
        ref: str | None = None,
        *,
        path: str = "",
        recursive: bool = True,
    ) -> list[TreeEntry]:
        """列出仓库目录树，``ref`` 为 ``None`` 时使用默认分支。"""

    @abstractmethod
    async def get_file_content(self, 
        path: str, 
        *, 
        ref: str | None = None, 
        line_start: int | None = None, 
        line_end: int | None = None,
    ) -> str:
        """读取指定版本的文件内容（UTF-8）。

        供评审方在提交未变更时获取完整上下文；二进制或超大文件由实现决定是否
        抛 ``UnsupportedFeatureError``。
        """

    # ------------------------------------------------------------------ 合并请求

    @abstractmethod
    async def get_pull_request(self, number: int) -> PullRequest:
        """按编号获取 PR/MR。"""

    @abstractmethod
    async def list_pull_requests(
        self,
        *,
        state: PullRequestState = PullRequestState.OPEN,
        source_branch: str | None = None,
        limit: int = 50,
    ) -> list[PullRequest]:
        """列出 PR。``source_branch`` 可用于按分支过滤，便于 Webhook 重放/兜底轮询。"""

    @abstractmethod
    async def get_pull_request_diff(self, number: int, *, include_patch: bool = True) -> Diff:
        """获取 PR 的变更集（等价于三点 diff，但优先走平台专用端点以复用 patch）。"""

    @abstractmethod
    async def get_pull_request_commits(self, number: int) -> list[Commit]:
        """获取 PR 内的提交列表。"""

    # ------------------------------------------------------------------ 评审回写

    @abstractmethod
    async def submit_review(
        self,
        number: int,
        submission: ReviewSubmission,
        *,
        commit_sha: str | None = None,
    ) -> ReviewResult:
        """提交评审：总体评论 + 行级评论 + 评审结论（评论/通过/要求修改）。

        Args:
            commit_sha: 行级评论锚定的 head sha；``None`` 时用当前 PR head，
                增量评审应显式传入以避免评论错位。
        """

    @abstractmethod
    async def list_review_comments(self, number: int) -> list[ReviewCommentRecord]:
        """列出 PR 上已有的评论，用于增量评审去重与状态回读。"""

    @abstractmethod
    async def reply_to_review_comment(
        self,
        number: int,
        comment_id: str,
        body: str,
    ) -> ReviewCommentRecord:
        """在已有评论下追加回复。"""

    # ------------------------------------------------------------------ 可选能力
    # 以下能力并非所有平台都提供，默认抛 UnsupportedFeatureError，
    # 子类按需覆盖；同时可据此实现「能力探测」。

    async def update_commit_status(
        self,
        sha: str,
        state: str,
        *,
        context: str = "ai-code-review",
        description: str | None = None,
        target_url: str | None = None,
    ) -> None:
        """回写提交状态（CI 门禁用）。``state`` 取值由平台决定，如
        ``success`` / ``pending`` / ``failure``。"""
        raise UnsupportedFeatureError(f"{type(self).__name__} 不支持提交状态回写")

    async def create_branch(self, name: str, *, from_ref: str | None = None) -> Branch:
        """创建分支（自动修复/回写补丁用）。``from_ref`` 为 ``None`` 时基于默认分支。"""
        raise UnsupportedFeatureError(f"{type(self).__name__} 不支持创建分支")
