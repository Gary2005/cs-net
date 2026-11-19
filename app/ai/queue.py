"""AI 分析的队列：**一次只算一个**，而且和提取队列分开。

为什么不复用 :data:`app.jobs.manager`
-----------------------------------
那个队列跑的是提取（一份 demo 30~90 秒，CPU 拉满）。把"点一个回合看看 AI 预测"
排到它后面，等于让用户等前面那份 demo 解析完；反过来，一次整场统计要几分钟，塞进
提取队列也会让新丢进来的 demo 一直排在后面。两条队列各有一个 worker，互不阻塞。

为什么还是只有 **一个** worker
----------------------------
模型只有一份（常驻显存/内存，见 :mod:`app.ai.service`），两个回合同时跑只会抢
显存、抢 CPU，还多占一份输入张量。一个 worker + `subject` 去重就够了：

* 同一个回合被点两次（两个浏览器、回放页 + 统计页）→ 第二次**看着**第一次跑
  （:meth:`JobManager.find_active`），不排第二个任务、不重复推理；
* 不同回合 → 严格按提交顺序一个算完再算下一个，算完就释放这个回合的张量
  （`service._tensors_of` 只留最近 3 个回合）。

排队本身几乎不占内存：一个 Job 对象 + 一个线程，真正吃内存的张量要等 worker
开始跑才构建。跨比赛切来切去时，唯一被拖着不放的是**正在算/排在前面**那场 demo
的 `data`（解码后的 .csz），等它算完就释放。

这个 `__init__` 之外的模块都懒 import：库页面每次轮询任务列表都会碰这个文件，
不能为了一个列表把 torch 拖进来。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..jobs import Job, JobManager

#: AI 队列（一个 worker，见模块说明）。
manager = JobManager(workers=1)

#: 任务类型：出现在任务抽屉里，和 `extract` / `scan` / `avatars` 并列。
KIND = "ai"


def submit(label: str, fn: Callable[[Job], Any], *, subject: Any = None) -> Optional[Job]:
    """把一段慢活排进 AI 队列；同一个 `subject` 已经在跑就返回 ``None``。"""
    # 懒启动：谁先提交谁把 worker 线程拉起来（幂等），这样 import 本模块不会
    # 顺手起线程 —— 测试里能干净地建/拆多个 app。
    manager.start()
    return manager.submit_unique(KIND, label, fn, subject=subject)


def find(subject: Any) -> Optional[Job]:
    """已经在排队/在跑的同一个任务（给"跟着看"用，不新增任务）。"""
    return manager.find_active(subject) if subject is not None else None
