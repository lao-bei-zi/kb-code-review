"""各来源的 ``GitService`` 实现。"""

from .local import LocalGitService

__all__ = ["LocalGitService"]
