# Repository Guidance

1. 开始任何任务前，完整阅读 `Module/MASTER_PLAN.md`；它是唯一有效的主任务记录。
2. P0、M01–M03、M04 MCC 和当前 MCC-only E05 已获授权；Finger DP 标记为
   `REWORK_REQUIRED / NOT_EVALUATED`。没有用户新的明确授权时，不把 DP、planner、GPIS
   或集成实验作为已验收结果发布。
3. 始终使用 Conda 环境 `handcomp`，其 Python 路径为
   `/home/ferry/data/Anaconda/envs/handcomp/bin/python`。
4. 严格按主计划的依赖和 Gate 推进；同一时间最多一个模块为 `IN_PROGRESS`。
5. Main method 与 Explicit baseline 必须保持主计划规定的隔离边界和公平比较条件。
6. 只有达到预先冻结的通过标准并登记证据，才能把模块标记为 `PASSED`。
7. 保留与当前任务无关的用户文件、数据和工作树修改。
