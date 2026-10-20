# 主任务记录：分模块验证与完整系统实验路线

> 本文件是仓库唯一的项目实施与实验主计划。开始任何模块前必须先检查本文件。

## 2026-10-20 B完整遍历恢复与仿真高力容忍

用户要求完整B，并允许适当忽略不准确的仿真高力。新增`b_full_mcc`，
恢复原B的多指起步/推进、参考手指更新及逐指转角限速，目标排序不再突变当前触点；
从共同初态、零覆盖，以单一冻结版本先遍历整瓶、再补扫遗漏区域。
超过12N只记录，超过40N连续200ms才停止；原8N覆盖排除及双几何守护保持。

当前状态`FRESH_SINGLE_VERSION_FULL_B_COMPLETE / PASSED_NATIVE_EVIDENCE_AUDIT`，
案例`generated/i04b_full_force_grace01_20261020`。4859.048秒完成8192/8192顶点及388/388目标；
2,429,524个原生状态连续重放、4,859,048次双几何守护通过，原始力峰9.0952N。
正常速度完整视频80分59.10秒，12倍速总览6分45秒。最长全手断触62.808秒，连续性仍需优化。
采用这些执行条件的新A尚未重跑，本次验证B完整功能。
见[完整运行说明](I04_B_FULL_RUNBOOK.md)和[本次记录](evidence/2026-10-20_I04_B_FULL.md)。

## 2026-10-20 B行为回归检查和完整视频

用户要求解释旧B已完成100%而新B仅17.62%的差异。检查确认新公共入口改变了接触门槛、滑动限速、
逐轮目标重投影和锚指状态维护。原48份控制源码保持相同，但覆写方法改变了B执行行为。
212.210s的目标2–3mm跳变已从同一物理/控制初态精确复现212.316s停止；仅改点交接的两个1s分支未触发停止，
但仍有≥8N观测，尚不代表安全或完整遍历修复。旧成功轨迹的版本接续与新从零补扫也存在调度差异。

本诊断轮完成了回归定位及A/B正常速度全程视频，保留历史失败和成功结果。
后续独立B已按上节修复参考交接、参考手指维护并完成从零整瓶遍历；
接下来仍需让A共用修复后的非对照部分，再从共同初态重新配对运行。
详见[回归记录](evidence/2026-10-20_I04_MCC_REGRESSION.md)及[完整视频](generated/i04_mcc_regression_review_20261020/review.html)。

## 2026-10-19 新I04A-MCC基线与首轮同条件B参照

按用户要求新增A-MCC运行。`matched_mcc`复用同一已核验MCC/测量/守护/原生执行核，
A仅接收手掌位姿，名义手指固定为公共初态手形；B保留逐指有限曲面目标与联合规划。
两案完整初态、公共参数/源码哈希和原生模型一致；实测2指/300ms/80%窗口共同冻结，不传入预测接触人数。
局部A/B各16秒原生核验通过；429项I04回归和2项恢复统计检查通过。当前仅seed7，DPRef暂缓。

A主案`generated/i04a_mcc_traverse01_20261019`在783.610s因连续16轮无新增覆盖停止：
面积指标10.1355%、1031/8192顶点、42/388目标；391805原生状态核验通过。
新B参照`generated/i04b_mcc_traverse01_20261019`在212.316s持续高力停止：
17.6220%、1950/8192顶点、71/388目标；106159原生观测（106158已执行命令）核验通过，力峰13.6442N。
两案状态、力、覆盖重放误差0，法向浮差≤3.34e−16，全部995926次双几何守护检查通过。
这证明原生记录可复现，不把B的持续高力停止视为安全性能成功。

共同前212.316s：A/B覆盖5.5667%/17.6220%，≥2指同时率7.5755%/48.6096%，≥3指0%/35.4170%，
四指0%/20.3527%；该前缀原始力峰6.8481N/13.6442N。B早期覆盖和多指参与更高，但出现高力保护停止。
两案都未完成整瓶；旧run10的100%不混入本表。完整原生轨迹、视频、恢复/覆盖与停止原因分别保留。
下一步是改进共享高力卸载和长时停滞，再冻结版本做seeds7/11/19；未来DPRef还需冻结腕序列调度重放。
详见[首轮记录](evidence/2026-10-19_I04_MATCHED_MCC.md)和[新运行说明](I04_MATCHED_MCC_RUNBOOK.md)。

## 2026-10-19 I04B新目标与I04A共同条件

用户明确将“手掌与四指联合规划、主动指腹接触、自然屈指、贴面优先换面和整瓶遍历”作为I04B新目标，
同时要求I04A/B的非对照部分一致。当前目标与共同条件以 `I04_AB_WHOLE_HAND_PROTOCOL.md` 为准。
I04A保留wrist-only任务输入及MCC/DPRef方法设计；DPRef继续暂缓。I04B增加逐指位置/朝向与联合几何规划。
两侧统一机器人/物体、初态、自然手形约束、指腹分类、MCC、力保护、传感历史恢复、执行预算、覆盖和核验。
A不接收B的逐指GT落点、法向、几何种子或完整关节参考。B几何重规划与公共历史恢复分别记录。
A-MCC与未来A-DPRef使用相同名义腕参考；A/B使用同源外层任务与公共规划框架，允许联合手指目标改变实际路径。

目标更新时状态为 `GOALS_AND_COMMON_CONTRACT_DEFINED / SHARED_RUNTIME_ALIGNMENT_PENDING / NEW_PAIRED_EVALUATION_NOT_RUN`；
后续首轮MCC实现与核验见上节，当前协议状态见`I04_AB_WHOLE_HAND_PROTOCOL.md`。
run10及下述原生核验继续作为B新目标的开发先导证据；不把历史A和run10写成已经完成的新公平对照。
下一实施顺序：新公共配置/执行层 → A/B目标适配器 → 统一推进与恢复窗口 → 信息边界和公共参数核验 →
局部/换面原生验证 → 冻结新版本、相同预算、seeds 7/11/19从零配对运行。已发布冻结源码和轨迹保留。

## 2026-10-19 整手协同整瓶遍历（原生全轨迹及审阅交付通过）

最终主案 `generated/whole_hand_surface_run10_20261019` 在5277.036s完成8192/8192原顶点与388/388目标覆盖。
2638518个2ms状态从原t0连续独立重放通过，全部5277036次实态/上一参考几何守护及266个分块哈希核对通过；
物理状态、力、覆盖误差0，法向浮差≤3.34e−16，unsafe=0、臂自碰力0。末段安全保持0.510s。
这是v4→v6→v7→v8准确续接的开发轨迹，三处边界为2260.000、4102.920、5273.000s，未拼合独立试验覆盖。
滑动控制阶段≥3指同时合格接触77.2113%、四指29.1176%；完整时间轴各指接触50.7130/57.2579/57.8054/47.9259%。
四指合格接触时实际切向材料运动9.088/10.128/10.388/8.152m。各指持续主动控制，不声称始终四指接触。
完整轨迹249段、70次换姿计划；最长全手有效接触中断146.156s，原始力峰10.000660N，换面连续性仍有明显改进空间。
47项回归及两例真实几何回归通过。完整24倍速回放219.95s、六方向1倍速节选24.20s已生成。
原始力≥8N共18段、累计0.146s、最长0.040s，均排除覆盖；未达到12N硬停止阈值。
审阅入口为 `generated/whole_hand_surface_run10_20261019/review.html`；另附3.95s瓶底正常速度特写，
仅局部相机隐藏机械臂/地面以消除遮挡，左侧保留原始全景，物理模型、运动和接触记录不变。
48份冻结源码、266个轨迹分块、三段视频编码/哈希及页面引用检查通过；六方向姿态、12张编码抽帧与三张指标图已目视检查。
交付证据为 `delivery_checks.json` 和 `manual_visual_review.json`；图像抽查不代替逐帧原生物理核验，也未声称浏览器页面布局测试。
100%仍指原25mm网格边测地接触邻域指标，不是实际皮肤压力足迹；仅验证固定Mustard、零重力、MCC。

2026-10-19发布整理：按用户要求将报告、三段正式视频、全部266个原生轨迹分块及核验来源一并纳入版本；
大体积MP4/NPZ使用Git LFS，文件清单及哈希见 `evidence/2026-10-19_I04_WHOLE_HAND_SURFACE.json`。
下载/运行说明见 `I04_WHOLE_HAND_RUNBOOK.md`，版本日志见 `CHANGELOG.md`。发布前完整I04回归424项通过。
新增隔离的可迁移入口，只在内存中重定位同字节资产；原冻结模型、控制器与原生证据不变。
前500步原生重放与独立进程双几何守护通过，q/qvel/命令误差0；此启动检查不替代原全轨迹独立核验。
零覆盖新实验入口另标 `v8-fresh-published`，只使用发布姿态作为几何种子，不继承覆盖，不称run10逐步重新规划复现。

用户接受probe08局部整手效果，并要求遍历整个物体。本轮已将同一四指共同MCC与自然屈指约束
接入完整Mustard表面调度；新入口和规划扩展隔离于已核验局部及旧单领触整瓶源码。
规划四指共同表面落点，优先连续贴面前沿，再尝试贴面换向，必要时执行严格几何检查过的短撤离/调姿。
不能要求每个曲率区域永久四指同时接触；四指目标始终参与，报告实际各指率、同时人数和全部未覆盖区域。
每次完整主案从唯一声明初态、零覆盖启动，不继承probe08或旧fresh02覆盖。保留原8192/388分母、
25mm测地覆盖、原生掌侧指腹分类、力和双命令几何守护；整瓶完成必须由完整原生轨迹独立核验。
先做多面几何可达与短连续跨面验证，再从零运行完整遍历；失败和局部进度不能拼成整瓶成功。
六方向代表点均找到严格几何可行多指手形。run01的43s/5段与run02的53s/5段均无抬手，
分别21500/26500原生状态独立核验通过，只是6.27%/6.61%的开发前缀。
run03在342s开发停止，18.0375%、1428/8192、71/388，22段/1次离面；171000原生状态独立核验通过。
检查瓶口极端点发现平面法向误差会误报多指可达：旧候选非锚指实际距网格19–26mm。
隔离v4增加真实有限曲面支撑距离、迭代重投影和接触人数优先选择；瓶底四指、瓶口/瓶肩三指几何可行。
四指路线使用重投影后的实际可达落点。1440组原生/扰动命令与原守护结果完全一致后，
启用精确q键的几何查询复用，独立审计仍逐步运行原守护。run04冻结v4继续从零完整遍历，旧前缀不拼接。
运行中发现全局种子会造成不必要的长换姿：t430.440同目标3033，原61.398s规划有当前腕姿2.784s候选。
隔离v5优先当前腕姿的多指解，贴面转向也使用有限曲面重投影后的落点和支持人数。
短换面probe02从声明的原生实态独立执行，2.786s完成撤离调姿，5.880s重新接触，随后2s保持；
3940原生状态独立重放通过，峰1.660279N、unsafe/臂自碰0，最长全手失联4.220s。
probe01因诊断脚本误用关节顺序而无效，已保留并标注；probe02映射后初始接触力与原记录严格一致。
run05冻结v5从零执行至1288s后开发停止，43.318845%、3769/8192、175/388；644000状态独立审计通过。
它与run04均含下述换姿超时问题，现集中完成修复版run08；run05覆盖不并入其他轨迹。
run04第11次调姿实际卸力暂停11.018s，超过原总预算中的8s余量，导致空中提前退出并保留旧接触目标。
已停止该运行并仅封存t0–2260s完整检查点前缀：70.667618%、5625/8192、275/388，1130000状态独立审计通过。
v6 `whole_hand_recovery.py` 分别累计路径执行和卸力暂停预算，转移失败后刷新当前有限曲面目标，
已卸载的手直接规划到新悬停；缺少路线时给主动接近实际控制时间。新增两项暂停时钟回归通过，原45项仍通过。
run08从run04准确原生终态继续：保留原t0初态、全部前缀控制/接触记录及版本来源，物理q/qvel/ctrl/warmstart无重置，
必须从t0跨版本连续独立重放，不把其他运行、probe或几何覆盖并入。此案明确标为续接开发轨迹，不称单版本从零运行。
续接后3.800s重新建立多指接触并继续滑动。run08最终在4102.920s因历史目标失败次数达到搜索上限退出：
97.833331%、7987/8192顶点、384/388目标，219段、33次换姿计划；2051460原生状态从t0连续独立审计通过，未完成整瓶。
该退出只表示规划搜索耗尽，不能解释为剩余区域物理不可达。run06/07仅初始化检查/文件复制失败，无原生运动。
v7 `whole_hand_completion.py` 从未覆盖顶点的18mm测地邻域挑选接触候选，允许1mm起的短滑动，
并使用本条已核验轨迹的真实多指姿态作为几何种子；候选及预测收益一律不计覆盖。
无路线时保持MCC接近状态，不反复重设悬停间隙；历史目标失败次数不再直接终止整瓶。
run09准确续接run08终态，保存两处控制器切换及递归来源，原始t0、物理状态、8192/388分母和实际接触判据保持。
新独立审计 `whole_hand_completion_audit.py` 从t0重放，并对所有版本的每个原生状态执行上一参考几何守护。
run09执行到5273.000s后为修复最后一个瓶底落点而开发停止：99.973042%、8191/8192、387/388；
2636500个原生状态及全部双几何守护通过独立连续重放，只有顶点2625仍未覆盖。
默认0.2mm间隙候选被原穿透判据拒绝；同一实态的2mm候选找到四指严格几何可行解及10.207s转移路径。
v8 `whole_hand_adaptive_completion.py` 仅在默认候选失败后尝试1/2/3mm几何间隙，仍优先多指解；
实际接近、MCC、力阈值、原几何守护及25mm接触覆盖均不变。run10精确续接run09终态补最后一点。
run10已在5277.036s达到原生完整覆盖：8192/8192、388/388，2638518个原生状态、249段、70次换姿计划。
最后续接段4.036s/2018状态单独核验通过，物理状态与命令无跳变，新增覆盖为真实接触，末段安全保持0.510s。
这一轮选取邻近落点2638完成补摸；不能把几何诊断的直接目标2625/10.207s路径当作实际执行路径。
整条轨迹从原t0的最终独立重放已经通过；回放、覆盖图、实测接触点和完整时间轴均已生成。
瓶口probe01另从严格校验的6mm悬停初态执行，5.646s完成接近、继续贴合4s；4823原生状态独立核验通过。
后4s至少三指75.95%、四指34.05%，四指各参与74.6/77.4/89.9/62.85%；峰2.766032N、unsafe/臂自碰0。
该局部瓶口证据不拼接到整瓶覆盖。
新协议见 `I04_WHOLE_HAND_SURFACE_PROTOCOL.md`；原始数据、源快照、控制和独立重放各案分开保存。

## 2026-10-18 整手协同贴合与自然屈指（局部通过；整瓶协同待接入）

用户最新明确：本轮重点不是失联，而是单指接触、其他手指被动伸直，以及第二根手指姿态僵直。
按中指理解第二根手指，目标改为整手主动贴合，不能继续以单领触指的整瓶覆盖代替此项验证。
旧fresh02单版本原生证据和运行源码保持。新隔离 `whole_hand_belly.py` / `whole_hand_belly_probe.py`
共同优化腕与四指，给各指附近表面独立目标；采用弯曲手形先验和软关节协同，允许按曲率调整，
不固定另外三指。新的长指任务范围保持小侧摆与正向屈伸，PIP/DIP正下界避免伸直等待；物理范围、
原始碰撞体、指腹分类、接触力与命令碰撞守护不放松。新增范围属于开发手形设计，不能直接称旧A/B公平性能对比。
先验证四指接近、滑动、接触人数/力分配及实际中指屈伸，再评估换面与整瓶；局部成功不继承旧整瓶覆盖。
最终局部主案 `generated/whole_hand_probe08_20261018` 冻结v1.6，从唯一6mm悬停初态和零覆盖执行，
25.988s达到 `COOPERATIVE_LOCAL_STROKE_COMPLETED`；5.848s进入共同滑动，扫描参考60mm。
腕与四指共享23DoF求解，采用连续名义腕路线、各指独立MCC、PIP/DIP软协同；先接触指交接不跳变。
拇指混合指腹/侧缘接触优先采用真实合格指腹法向，并沿表面小步避开接缝，原拒绝接触判据保持。
12994个原生2ms状态从唯一初态连续独立重放通过；所有已执行命令原严格几何与上一参考守护通过，
q/force/coverage误差0，法向浮差≤3.34e−16，unsafe=0，臂自碰力0，峰3.302455N，无规划抬手。
滑动阶段（含其间卸力/调整）至少三指同时合格接触92.6415%，四指59.8411%；
食/中/无名/拇指各99.0765/90.8044/93.8530/67.5670%，实际接触点沿扫描方向净移60.017/60.566/59.984/58.096mm。
中指滑动阶段实际MCP为18.691–28.663°，PIP为26.213–30.064°，DIP为10.247–15.875°，不再固定伸直等待。
接触交接允许个别短暂失联：300ms窗内≥3指占90%以上、每指参与60%以上、当前四指贴合；
此条件仅用于运动交接，接触/覆盖分类未放松。末窗独立复算通过，永久被动第四指的反例测试明确拒绝。
此前probe01–06超时/开发停止均保留原生审计，不称成功。probe07也完成局部试验，但不是新主案；
probe08四指率59.8%高于probe07的56.4%，三指率92.6%略低于93.5%，力峰3.302N高于2.822N，不能称全面改进。
局部覆盖仅4.327236%（396/8192顶点、17/388目标，仍为25mm测地邻域指标），**未完成新整瓶协同遍历**。
旧fresh02单领触整瓶结果不继承为新结果。38项相关回归通过；正常速度完整局部三视角回放及指标见主案 `review.html`。
详见 `I04_WHOLE_HAND_COOPERATIVE_BELLY_PROTOCOL.md`。当前活动模块保持整手协同；DP继续暂缓，G3/GPIS及一般多物体结论不解锁。

## 2026-10-18 自然手形与整瓶指腹遍历（单案例整瓶通过；连续性待优化）

最终主案 `generated/mustard_natural_fresh02_20261018` 已由冻结v1.11从声明悬停初态、零覆盖完整执行，
在4368.050s达到 `FULL_MEASURED_BELLY_SURFACE`：8192/8192原顶点、388/388目标全部覆盖，
末段150ms安全指腹保持及实际终态通过。2184025个2ms原生状态从唯一初态连续独立重放通过，
所有已执行命令原严格几何检查和相对上一参考的额外守护通过；q/force/coverage误差0，法向浮差≤3.34e−16。
原物体/物理碰撞体、覆盖分母、25mm网格边测地邻域判据均保持；100%不是皮肤压力足迹面积。
六方向均有真实安全指腹接触，包含瓶盖/瓶肩端和瓶底端。
长指实际侧摆最大绝对10.438°，PIP最低−0.180°（微小执行器偏差），已消除旧59°侧摆和−28°PIP反折。
全程至少一指安全接触63.636%，SLIDE阶段90.506%，CONTACT_PIVOT阶段81.575%；
125次规划器离面重定位，最长全手有效接触中断88.050s（3873.586–3961.636s）。不能称作不间断遍历。
峰12.13315N；≥8N共4段/52ms，均不计覆盖；>12N持续8ms，未触发原20ms硬停。臂自碰力0、unsafe_states=0。
此结果只验证单个Mustard、MCC、当前固定物体/零重力仿真条件；不解锁一般多物体性能、G3/GPIS或DP比较。
完整364.1s的12倍速视频、18s的正常速度局部回放、六方向实测点/覆盖图及原始证据见该case的 `review.html`。
`delivery_checks.json`核验了视频编码帧数/时长/哈希、全部页面引用、原始分块及当前34个运行源与冻结清单一致。

用户指出上一版只有局部覆盖且手形怪异。新隔离入口 `natural_belly_traversal.py` 从声明的
自然手形悬停初态开始，零继承覆盖；原 mustard 网格、256 件碰撞体及腹面逐接触判据保持。
查明 LEAP 长指规范顺序为侧摆、MCP 屈伸、PIP、DIP，旧 probe06 实际侧摆最高59.16°、
中指 PIP 最低−28.19°。新任务姿态约束长指侧摆±10.31°、PIP/DIP参考不反折，
按可达性选择领触指，其他手指保持轻展，拇指不强行与三指压同一个平面。
完整8192顶点/388目标按长轴蛇形带扫、双侧、瓶盖/瓶底排队；实际合格指腹接触独立累计。
相邻滑动段保留 MCC 状态和姿态参考；先尝试贴面小角度转腕，再尝试6/15/35/65mm必要离面。
几何查询改用原生FK+collision省去无用约束力求解；180个原/扰动姿态与旧严格守护的
原因、间隙、穿透和臂间隙全一致，Jacobian差0。28项新旧指腹/有界步进测试通过。
run01/03均触发8N开发保护，run04因参考路径从实测微小越界状态起步触发COMMAND_BOUND_VIOLATION，
均为阴性保留，不能拼作成功。run02完成80.002s、实测腹面3.99249%/300顶点/14目标、峰3.51221N；
40001个原生状态连续重放和所有已执行命令原严格几何核验通过，仍仅阶段性验证。
run05（v1.3）483.854s触发8N开发停止：18.27271%、1747顶点/65目标，峰8.05492N；
241928状态及全部已执行命令连续原生重放、原严格几何核验通过，未完成整瓶。
v1.4增加3N主动卸力和首个合格腹面接触交接，run06在184.498s因正间隙边界停止；
v1.5加入保持原阈值的主动退出间隙运动。run07被开发中断，只有600s检查点前缀，不作为完成证据。
run08（v1.6）从已审计run05的完整物理末态续接，1042s开发停止；24.51554%、2323顶点/91目标。
521000状态从原始t0连续重放，全状态/force/coverage误差0，法向浮差≤3.34e−16；全部实际命令严格几何通过。
不是单版本fresh结果。全程有效接触最长gap187.386s，换面接近规划仍有明显问题，未宣称连续性通过。
v1.7修正每段强制150ms稳定导致已到达位置停滞（590–645s最长安全接触bout仅76ms），
普通段按实测接触到位衔接；最终整瓶仍要求150ms稳定。换面先验检验悬停至贴面的完整几何路径，
优先近邻未覆盖前沿；30项姿态/指腹/有界步进回归测试通过。
run09从已审计run08物理末态继续，1286.562s开发停止，30.48598%、2708顶点/114目标。
卡点为0.95964mm正臂间隙缓冲区，未发生臂碰撞；不断原地重试不可行。
v1.8增加主动正间隙退出；同真实末态局部原生probe单步2ms将参考间隙增到1.03962mm，
指腹力0.97610→0.32217N，原严格命令几何通过。只属局部证据，不计整瓶覆盖。
换面比较路径时长，并可用未覆盖目标邻近40mm内的已覆盖点作有限落指中继；中继不授予覆盖。
冻结1069.5s长绕行样例原目标路径99.256s，邻近22–33mm目标几何路径3.33–7.00s，待原生执行。
40个腕IK算例仅省略scratch动力学后与原解逐值一致，30项相关回归通过。
run09的643281状态及所有原命令已独立连续重放通过。run10在1287.280s开发停止，未新增覆盖；
仅退出1mm边界没有给后续IK留足空间。v1.9增加IK臂间隙目标，但run11在1287.250s因
SHALLOW_ARM_SELF_COLLISION几何守护停止，未声称成功，也不继承到后续主案。
诊断显示旧“相对实测状态退出”规则可在执行器运动滞后时让参考距离逐步减少；新v1.10同时
检查相对上一条参考的原严格规则，并在MCC步内加入6mm软臂间隙目标，腕IK加入8mm软目标，
1mm原硬阈值不放松。run12从已审计run09末态继续，已越过旧卡点并在不新增lift情况下继续覆盖。
另启mustard_natural_fresh01_20261018，从零覆盖、声明悬停初态运行同一v1.10完整回归；
run12在1518.648s开发停止：46.21656%、3777/8192、169/388、22次累计离面；
759324状态和全部原始命令、其本版本命令相对旧参考的额外守护均独立核验通过。
fresh01在355.452s开发停止：23.75713%、1863/8192、91/388、5次离面、峰3.36742N；
177726状态连续独立核验通过，未完成整瓶。两案均出现非指腹3mm缓冲区附近的无路重试。
v1.11增加非指腹间隙主动退出（原生12步/24ms：3.005→5.877mm，无物理力，原双命令守护全过），
接近阶段按实际退距推进验证过的腕姿参考；无接触时从实测pad位置重建规划锚点，几何不授予覆盖。
旁指拒绝接触采用独立法向卸载并保持已让开指位；真实配对0.5s试验将中指侧缘力2.562→0.198N，
领触指腹面力始终≥0.891N；旧保持对照侧缘力仍2.562N，全部命令原双守护通过。
新增真实伺服滞后反例回归，现32项相关测试通过。run13从已核验run12末态继续，
fresh02另从零覆盖开始冻结相同v1.11，分别运行和审计；最终fresh02通过，见本节首段。
run13现于4650.720s因SUSTAINED_DANGEROUS_FORCE停止：99.35413%、8138/8192、388/388，
峰13.76195N，97次累计离面；2325361原生状态连续独立重放通过只证明记录一致，不代表整瓶通过。
失败保留，不将其末态继续接入合格整瓶结果。fresh02同一冻结版本独立从零完成。
补扫还观察到约5mm短prefix可能被5mm端点容差立即判为结束，覆盖没有虚增，但浪费路线尝试；
当前冻结版本未改。另在真实失效前状态做隔离配对卸力probe：补参考滞后的候选无改善，
实测法向速度制动40ms/每步期望法向参考增量至多0.6mm的候选将13.76195N样例降至6.91757N、50ms卸到1.5N以下；
另一12.13315N样例降至8.95829N、22ms卸到1.5N以下。原双几何/关节步长守护通过，无覆盖授予。
此为未接入整瓶冻结版本的局部候选，不能替代完整回归；原持续高力条件在配对基线中保持。
源与结果在generated/natural_belly_lag_release_probe_20261018；早期不含速度制动的阴性及诊断稿也保留。
源码、原生2ms分块状态、命令、MCC、接触分类、覆盖和检查点均保存在各独立case。详见新自然手形协议。

## 2026-10-17 指腹正面接触修正（进行中）

用户明确要求末节朝掌心一侧的指腹贴面。旧版“tip/pad geom 接触”只证明整个椭球代理命中，
不能证明指腹正面接触。独立重建旧145–190s、1240–1280s两窗8502个保存状态、18894条原生接触，
force/normal/point误差均0；中央正腹面接触为0。旧无名指1249.9s恢复例实际含背侧接触，
不能继续称作“指腹恢复成功”；历史轨迹和旧coverage保留，不继承为新腹面coverage。

新增隔离 `belly_contact.py`：固定pad +Z（末节body −X）掌侧轴；逐原生contact检查朝向≤30°、
中央椭圆归一化半径≤0.8、pad witness表面误差≤0.1mm及法向一致性。原始全部接触力照旧守护安全。
`belly_surface_planner.py`加入支撑点与姿态联合IK，物理尺寸/碰撞代理不改。
局部原生入口 `belly_surface_probe.py` 先尝试最小必要卸载、三指共同调姿，再主动逼近与滑动。
probe01调姿完成但全局缩放被单关节限位卡住，46.194s超时、腹面接触/coverage均0，明确阴性保留。
新 `belly_bounded_step.py` 直接求解多指盒约束，允许其他关节补偿；24项新单元测试通过。
最终probe06完成22.564s局部试验，12s滑动始终至少两指真实安全腹面接触。
食/中/无名指接触率99.833/99.867/95.867%，最长gap20/8/278ms，实际切向材料运动48.533/52.139/49.232mm。
峰2.965604N、臂自碰0；11282个原生状态从同一初态连续重放全状态/force/point误差0（normal浮差≤2.22e−16），
35源哈希通过；11282条实际命令每2ms原严格几何/关节边界检查全过，最大参考穿透1.564mm、实测0.552mm。
局部腹面面积2.28874%（193/8192、7/388），未宣称四指参与或整瓶连续遍历完成。
本轮主案 `generated/mustard_belly_probe06_20261017`，审阅页 `generated/new_i04_mustard_belly_review_20261017`。
probe02/03数值拒绝阴性、probe04命令几何保护停止、probe05无名指较长gap均保留，不拼接为完整结果。

## 2026-10-16 主动手指寻触与贴面换面（进行中）

最新v3.8-release fresh运行已1413.970s触发SUSTAINED_DANGEROUS_FORCE停止：
6582/8192、298/388、78.0576047%，174段/25lift，峰15.0838409083N，臂自碰0。
trace SHA6f0f75470f8ec66fed8dcd81a3d5f1f8efef8c77e80a4de7c37fc13b9c2ea226。
主动手指调形已产生真实恢复，但最新整瓶连续滑动验证未通过；不以300/900前缀或旧版全覆盖掩盖失败。
1413.920s开始卸力后仍增力；末11原生状态连续重放全部exact0，10个连续2ms观测>12N触发原硬停。
独立确认手指DLS逐关节限速/限位裁剪翻转卸力方向：1413.950s raw向外8.72µm，
裁剪后ring向内153.45µm，腕外撤60µm后仍净向内；真实下一步亦向内。冻结失败与红区保留。
隔离direction_preserving_force_release候选统一缩放整指原DLS，5边界测试通过；
pair02同1413.920初态，原26帧50ms精确复现15.08384N硬停，候选23帧44ms到3.84955N，
峰初态8.57852N。仅卸力段局部验证，无生产handoff/整瓶运行，未整合到冻结v3.8；
pair01漏mj_forward导致baseline不匹配已拒绝并保留。最终报告generated/fresh_v38_final_review_20261016。
pair02独立49原生状态/47实际命令/14历史锚点均误差0；候选22步含9步alpha0保持，thumb有单帧2ms失联。
仅证明这段卸力反例得到修正，不证明生产交接、普通MCC限位方向或整瓶通过。53源之外2加载依赖事后补记。

最新追加 v3 独立协调恢复入口 `coordinated_surface_slide.py`：有界手指求解、持续失联监视、
固定腕单指完整调形优先、小腕姿调整后备、实测首触即停路径、MCC 伺服预载连续交接。
短暂失联不判整瓶失败；原物理/力/碰撞边界保留。v2.2 run04已结束，未达到整瓶覆盖，见下文。
middle205 长失联配对 ON 14.182 s 重触、14.332 s 稳定，OFF 25 s 无重触；峰2.233 N/臂自碰0 N。
三对新probe共75006原生状态连续重放全误差0；腕小调整和食指bounded-only阴性结果保留。
v3 smoke02 30 s 中自然失联的 thumb 已主动调形恢复，21源及预算/传感审计通过，166项测试通过；
尚不是整瓶完成。run01 于14 s 开发终止修正预载路径预检与已有最终关节裁剪不一致的问题，
v3.1 保留原严格碰撞检查并明确记录裁剪量；详见主动手指协议新增章节。

v3.2 已接入最多32个邻域真实表面目标，以及保持支撑的3/6/12°转腕后从实际稳定末态再调手指；
两段共享60 s尝试上限/失联预算，恢复计入45 s区域无新增预算。严格几何与真实接触分开验证。
120 s smoke03完成并审计通过：16.75335%（1328/8192、64/388），16段、0抬手、峰8.34766 N、臂自碰0 N。
25源快照、201 FK/52原生传感及全部3个峰事件核验通过；203项I04测试通过。
index240新版组合ON 13.928 s首触/15.260 s稳定；OFF同初态35 s无重触，35002原生状态重放误差0。
ON稳定后仍有约6 ms短gap，15.11–35 s安全接触占96.47%；不能称完全不断触。无名指全程安全支撑。
run02从新acquisition启动，冻结entry SHA d620d6b024b87845dc33618829f40959d847200f7609dcc6cb04f17bc7ec3e48。
300s同初态对照发现拇指参与和全手移动占比回退，故617s开发主动停止；44.33763%（3228/8192、170/388），
147段/4抬手，峰11.6241 N/臂自碰0 N，25源及全案证据审计通过但不算行为成功。
目录generated/mustard_coordinated_run02_20261016；该案已停止，后续独立新版本见下文。
v3.3 priority已完成300s：安全移动接触91.9067%、最长全手gap0.05s，覆盖28.58063%，
峰9.60712N/臂自碰0，完整证据审计通过；0完整恢复请求，拇指最长失联226.12s，仍未解决逐指参与。
v3.4 context已完成300s：新实测几何context可重新开放耗尽的局部预算，保留真正失联episode和总次数。
已确认旧拇指跨157.95/196.19mm换区后仍被耗尽的预算锁住；新context需实测支撑/位姿变化/冷却，
不能靠普通stroke rebase刷新，须禁止A→B→A往返刷预算。26源/原生授权支撑窗/覆盖证据审计通过。
v3.4覆盖28.88597%，安全移动81.4967%，最长全手gap2.74s；index最长失联122.75→11.62s，
但thumb226.12→235.48s，middle5.43→54.79s，所以预算修复不能宣称整体行为改善。
v3.5 boundary已完成300s：SLIDE只排队，真实自然stroke端点才尝试≤8s固定腕单指路径；
覆盖29.79036%（2522/8192、117/388），72段/2抬手，峰11.01676N/臂自碰0。
27源和全部证据审计通过；13个新context、988个原生支撑帧、3次短调形均独立核验。
thumb17.702s、middle42.858s、index73.962s确认恢复，短调形合计15.318s。
同300s thumb接触56.4967%/最长gap60.85s，较v3.4明显改善；但index17.4067%/161.54s，
middle44.6367%/79.61s，整体移动77.4267%，均不能称全面改善。
最长全手gap9.42s在198.43–207.85s抬起/转移/接近阶段；短失联和必要转移不直接判失败。
249项I04测试通过。v3.6-prepared已完成整合，271项I04测试通过（60.491s）。
自然端点固定腕有限候选无解后，要求同一失联episode在至少2个真实context失败、失联≥30s、
同一其他手指原生稳定支撑150ms和60s逐指升级冷却，才允许尝试原锚定小转腕；
同context最多一次，真实选用anchor必须在稳定名单，原60s尝试/45s区域无增益限额不变。
原换面搜索失败后的安全外侧调手形/完整姿态退路亦已接入，几何永不授予覆盖。
精确46.96009s退路已由生产v3.6控制执行：47.728s稳定回触后继续SLIDE，无geometry defer/卸力/臂自碰。
终态注入集成验证独立记录，不称fresh整瓶。两项新端点集成及两项新退路集成均通过。
fresh长程run01于2026-10-16 10:48 UTC启动，零覆盖新acquisition，预算5000s，完成时提前结束。
目录generated/mustard_coordinated_prepared_run01_20261016/B-MCC_seed7；30源冻结，
entry SHA e1f2b0bfc875f24637e67b2e354e21bebb6c92a67b783cc1168e87b45a4407e3。
该旧v3.6 fresh案现已3483.300s结束：FULL_MEASURED_SURFACE，8192/8192、388/388，
585段、48次lift，独立最终覆盖重算与150ms保持核验通过；完整trace SHA
9c777cb9be4911bb3f8c3a82ac68d4b1640de4e5029da2fae2037f550d2fd1d3。
这是本次单次fresh执行的整瓶覆盖，不借用旧v2.2续接链；覆盖使用25mm原网格测地邻域。
但全程安全移动仅57.8423%、安全接触76.1037%，且有20.208068N真实高力异常，
最长全手离面119.64s、拇指最长失联666.25s；40次完整恢复中14次运行时确认、26次失败。
末段86个原生状态独立连续重放误差0，实际中指全170ms安全保持。故不能称连续性或安全已通过，
也不能冒称是新v3.8的结果。完整20倍速/原速恢复视频、原始3视图和证据在
generated/fresh_v36_final_review_20261016，v3.8终态另列在本节开头。
fresh第4次中指恢复实际失败：中心目标残差减小，但旋转指腹的物理间隙0.616→1.220mm。
已据此建立独立support Jacobian候选active_fingers_v4.py，原中心/新支撑点双8µm界和全部有限额度保留。
19项专项测试及7真实姿态×4指FK导数复核通过；同初态10s原生配对双阴性，10002帧重放误差0。
V4虽然少退0.414mm，但中指始终无首触；3D目标中的切向误差占优，候选不接入冻结整瓶源。
继续独立验证以真实分离间隙下降为优先的有限搜索，不能把静态可行步当实际恢复。
V5法向优先把中指gap降到0.386mm但仍阴性，并因近零梯度归一化产生来回摆动、耗满40mm。
V6保留梯度幅值与原阻尼，再做有限双基准FK进展检查；同负例距离消耗降为4.095mm，显式hold但不假报恢复。
同食指小撤离正例V6首触0.530s/稳定0.680s、其后4.47s实际连续safe；V3稳定0.878s。
两对15004原生状态独立重放误差0。另新卸力参考积分helper已在真实20.2N峰状态验证并接回MCC：
局部peak20.208→初始已有的12.931N，238ms确认重触，但仍有8.298N回弹与短gap，不称完全平稳。
独立新入口normal_surface_slide.py（v3.7-normal）保持原整瓶源冻结，接V6与reference_force_release。
14项新入口原生stage/helper测试通过；entry SHA92ba49b0bf1b352c06acf85497367c80f102e9e2b3fdc100ff711afc5636a766，
34源，30s fresh smoke在generated/mustard_normal_smoke01_20261016完成并通过独立证据审计：
6.69943%、507/8192、26/388、0lift、峰7.57196N；2441条proposal与205原生几何抽样核验。
同30s旧v3.6→新v3.7安全移动67.30→98.33%，最长同指滑动10.53→29.50s；
但thumb接触40.90→25.90%回退，仍需较长窗口判断逐指参与。
随后只扩展启动时已加载Module.*项目依赖快照，4项来源专项测试通过，控制行为不变。
新entry SHA9669c4469196e03e43bffb932522e0c572a345d316efd617eeefbcf1ec903dfc；
fresh generated/mustard_normal_run01_20261016/B-MCC_seed7已启动，预算5000s，先冻结300s同窗评估，
若通过则直接继续整瓶，避免重新执行前缀。旧v3.6的20.208N真实力峰保留，不算安全通过。
v3.7前120s暴露逐指参与退步：thumb接触69.98→6.48%，index41.23→32.83%，
覆盖17.301→13.201%；safe移动80.98→83.76%、最长同指滑动27.64→38.95s仍有改善。
这是寻触与卸力两项联合改动，不能直接归因其中一项。新独立release_surface_slide.py仅保留
V3原寻触并修改卸力参考，9项原生stage边界测试通过（18.200s）。开跑前独立检查发现
卸力分支previous_command变量名错误，修成实际last后新增2项真实8.2N原生分支测试通过，
60ms末3.8186N、helper28次调用/原生重放一致、全部保护保留。
entry SHA98b97a604f305ee41c1da62e3f51283581c05a667dff3e7545682bcdc055e1bf，
protocol v3.8-release，冻结父entry供实际控制循环单因素AST审计。
fresh generated/mustard_release_run01_20261016/B-MCC_seed7已启动，预算5000s、先120/300s对照。
v3.8单因素startup通过：51源、原v3.6共同30源相同，前50.734s/5074帧所有字段相同，
首次实际卸力后才分叉。300s完整前缀审计通过：2078/8192、103/388、27.25847%，
760原生调度支持帧重放误差0，native峰10.040116N/臂自碰0。
与v3.6同300s：safe移动77.4267→86.7733%，safe接触93.8367→96.15%，
最长bout31.31→48.05s；thumb接触56.50→73.67%、最长gap60.85→22.75s，
middle44.64→76.97%，index17.41→25.47%但仍有128.58s长失联。
覆盖29.7904→27.2585%较慢、全手gap9.42→10.70s略退步；总体支持沿原物理过程继续整瓶，
不改已冻结控制源。thumb155.496s恢复后30s safe99.9667%，index175.888s调形失败保留。
900s复核已通过来源/覆盖/原生支持与卸力命令审计：4406/8192、211/388、53.31719%，
峰仍10.040116N。相对旧v3.6同900s，安全移动70.3789→78.4089%、最长滑动51.18→68.77s；
安全接触92.6844→86.3611%、最长全手gap22.68→26.93s，覆盖54.9581→53.3172%，
所以结果是混合改进，不能称全面更连续。ring接触57.62→41.40%，thumb最长gap60.85→114.08s。
880.962s之后曾长墙钟等待，nonblocking调用栈定位普通换面RRT碰撞检查，随后返回并继续；
计算等待不计物理离面，生产规划缺少统一墙钟截止，诊断/未验证提案独立保存，不改当前运行。
v3.7的300s全案前缀审计已完成：safe移动77.4267→62.5533%，thumb接触56.4967→3.2367%，
全手gap9.42→15.43s、覆盖29.7904→22.9434%，故311s开发主动停止（非硬安全停止）。
终态23.13940%、2229/8192、91/388、101段/6lift、native峰9.03414N/臂自碰0，
全trace SHA959a0f5d1c8c0c5bce4d03788c01a31f474c22fbde7bf8ddb0c879a90b9d3785。
整瓶目标未达成，局部38.95s长滑动/较快食指恢复不掩盖整体回退；阴性完整保留。
原index组合采用旧寻触+BasicMCC，13.928 s短触但未稳定，35 s阴性记录保留；
发现短触反复清去抖与MCC关节限位裁剪方向翻转，真实V3+保载GraceMCC组合在当前case恢复成功；
原DLS裁剪机制仍未替换，保留其方向翻转诊断供后续优化，不能称该根因已修复。

用户追加：失联手指应主动接触，减少离面换面并最终摸遍 mustard。
当前实现 `active_fingers.py` 和连续路径的多指接力/已覆盖表面桥接/局部姿态调整，
本轮先验证 B-MCC；逐指 GT 表面目标是新增规划输入，不能套用旧 A/B 方法边界或称单因素消融。
协议 `I04_MUSTARD_ACTIVE_FINGER_SLIDING_PROTOCOL.md`。DP 延后。

独立 index seed7 配对撤离/恢复 probe 成功：腕参考不动，实际撤离 2.391 mm 后失接触，
ON 首次同指重触 0.786 s，含 150 ms 稳定确认共 0.936 s，峰值 2.178 N；OFF 在相同 5 s 窗内未恢复。
两组各 3300 帧连续原生重放误差为 0；只算局部恢复证据，不并入整瓶覆盖。
记录 `generated/i04_index_recovery_paired_seed7_20261016`，包含 1 倍速未剪辑回放。
Pilot01 120 s 无硬停止，18 段/1 次抬手/实测覆盖 12.977%；首次抬手仍有长空中路径。
独立审计通过，201 FK/51 传感重建误差 0。整个 mustard 的连续扫描仍在长程验证，
不能以 pilot 或历史 full-surface 的全覆盖替代本版本完成。

run02 长程出现瞬时 30.316 N 指腹冲击；独立核查根因为旧 2 mm 阈值漏掉浅层连杆自碰撞（约 208 N），
已中止，保留完整至 420 s 分块与诊断。v2.1 增加浅自碰撞保护和 tracking gate。
历史“无硬停止”仅对应历史判据，不能作为通过新增浅自碰撞标准的证据。
单指新 ON/OFF 配对恢复实验已完成并全帧原生重放核验通过，位置见协议。

run03 v2.1 在 180 s 开发主动终止、完整证据审计通过；无硬停止，指腹峰值 9.0844 N，实际臂自碰撞 0 N。
覆盖 20.0744%（1504/8192、78/388），全程安全切向移动 87.6278%，最长全指离面 9.29 s；不是整瓶完成。
最后 10.07 s 多次重规划仍无覆盖增长。v2.2 修复局部候选丢弃/中断不计预算问题，允许正常路径失败后
使用 ≥5 mm 已安全验证短桥；无进展时换区仍优先贴面过渡。94 项 I04 测试通过，缓存分支定向检查通过。
严格几何缓存替换 8 个旧姿态，387/387 合格；5775 的独立姿态搜索未找到，完整覆盖分母保留。
run04 v2.2 从相同新 acquisition 长程验证中，目录 `generated/mustard_active_sliding_run04_20261016`，不继承覆盖。
runtime SHA256 `ba1ee0fbdd02d8af3aec82763a62d3cac19c65265912455b16d88af52a3c9e79`。

run04 最终在2783.914 s以NO_SAFE_REPOSITION_AFTER_EXPANDED_SEARCH结束，99.78008%
（8181/8192、387/388），不是整瓶完成。13源、201 FK/52传感和6原生力峰审计通过；
峰11.0151 N、实际臂自碰0 N。安全移动接触81.8331%，最长严格同指滑动123.39 s，
该冻结版本未找到剩余11点的安全换区路径；运行失败与覆盖分母完整保留。v3.2独立run02也已停止。

独立终态连续续接已实际补齐原11点：原生clock从2788.914到3051.764（262.85s），
旧prefix+6段continuation联合账本8192/8192、388/388；续接峰3.48899N/臂自碰0。
目录generated/run04_terminal_completion_chain_20261016/complete_chain_evidence.json。
全链审计通过：131431原生帧连续重放误差0，最终实际安全保持152ms。
旧run04本体仍99.78008%，联合续接不作为fresh v3全程成功；v3.6主runner已接入新退路fallback并开始独立长程。

中指/无名指额外配对 probe ON 首次重触 1.494/1.022 s，OFF 均 5 s 未重触；全部连续原生复放通过。
三指六案共 19796 帧额外自碰撞力复核为 0 N，专项证据仍不并入整瓶覆盖。

## 2026-10-16 优先连续贴面滑动，受限时才抬起

用户要求规划时优先保证沿 mustard 连续滑动，必要时抬起调整；仅 MCC、DP 延后。
本轮 `NEW_I04_MUSTARD_CONTINUOUS_SLIDING / LOCAL_STROKES_VALIDATED`，
协议 `I04_MUSTARD_CONTINUOUS_SLIDING_PROTOCOL.md`。新增 `surface_slide.py`，
从当前实测接触和姿态生成 2 mm 密集表面路径，连续运输手腕朝向，默认切向 6 mm/s。
短暂失联先暂停切向并有界恢复；局部候选受阻、恢复超时或连续无进展后才重定位。
原始碰撞模型和安全阈值保留。抬起阶段停止向内压，过载时新增实测法向腕部卸力，
卸力后从当前状态重新规划。

最终 validation04：A/B × seeds 7/11/19，6 次均从新的共同 acquisition 完整运行 300 s，
没有硬安全停止；不是整瓶连续遍历完成。11 份运行源码 hash 一致，3 对初态逐元素相同。
57 项 I04 测试及 6 份独立审计通过；每案 201 帧 FK 和 51 帧接触动力学重建误差为 0。
validation03 的 A11/B19 两次持续高力停止保留为开发失败记录，不能与最终版本混合。

固定 seed7 的 B-MCC 例子：同初态、同 300 s 前缀，安全切向移动接触时间占比
旧 full-surface 34.29% → 新 41.75%；最长严格同指滑动段 18.35 s → 34.26 s，
新段切向跨度 195.21 mm。新 SLIDE 阶段的实测移动占比为 94.05%，
但全程仍有长空中转移，不能把局部 94.05% 写成全程连续率。
相同时间的覆盖 22.12% → 15.82%，说明连续段改善伴随覆盖速度取舍；规划和速度均有变化，
不是单项改动的消融实验。最终六案覆盖 10.26%–16.91%，完整表面连续扫描仍待验证。

报告：`generated/mustard_sliding_validation04_20261016/index.html`。
包含 seed7 B 的 1 倍速连续滑动特写、保留换面过程的 2 倍速完整回放、等时长对照、力与接触账本。
历史 full-surface v1.2 的整瓶离散触摸完成结论保留，不能迁移成本版本的整瓶连续滑动通过。

## 2026-10-14 mustard 全物体 MCC 遍历完成

用户指出 mustard 只摸了局部，尚未遍历摸一遍。本轮把全物体实际触摸覆盖作为完成目标；
本轮模块 `NEW_I04_MUSTARD_FULL_TRAVERSAL / FULL_TRAVERSAL_VERIFIED`。DP 继续延后。
原 v2 接触恢复实验和 18 次记录保留，不作为完整任务通过。旧路线 600 s development 诊断
仅覆盖 19.857%，不能以延长预算或接触率代替遍历完成。
已完成整瓶几何可达性、绕物体路径、到达后的局部接触恢复与完整实测覆盖审计。
新路径独立位于 `i04_geometric_traversal/full_surface.py`，不改写 v1/v2 冻结 runtime。
冻结协议 `i04-mustard-full-surface.v1.2`，source `20ea869`；A/B seeds 7/11/19 共 6 次，
全部从共同 5 s 初态连续运行，不继承开发覆盖。6/6 完成 100% 实测面积覆盖，
8192/8192 vertices、388/388 points；三组 q/qvel/ctrl 逐元素相同，31 项回归及六份独立账本审计通过。
覆盖仍为 25 mm mesh-geodesic 扫描分辨率，只有实际 pad contacts 增加覆盖；末段还确认安全接触稳定 150 ms。
Object 固定、原始 2.8 烘焙尺寸和全部 256-part 碰撞保持；GT 仅修复 4 个零体积重复/退化 faces，
没有因不可达删去真实区域。共享路径处理不可达外侧 offset 与正间隙缓冲退出，实际碰撞和力保护不变。

A/B 平均完成时间 6667.658 /5888.035 s，平均真实接触率 9.456% /13.335%，
瞬时峰值最大 12.204 /11.337 N；全部未触发持续 >12 N/20 ms 或实际碰撞停止。
完整扫描已成立，但转移仍长，安全接触 gap 最长 339.026 s；不能宣称连续贴面、低冲击或实时/实机性能。
本次全部观察到的安全接触 gap 最终恢复；没有 full-traversal recovery-OFF 消融，不能全部归因于显式层。
v1/v1.1 的初步/中断记录保留并排除在同版配对结果之外。

报告 `generated/new_i04_mustard_full_surface_v12_formal_20261013/index.html`，含完整 A/B 双视角回放、
六个面的真实稳定接触慢放、覆盖图和首次实际触摸时间图。
证据 `evidence/2026-10-14_NEW_I04_MUSTARD_FULL_TRAVERSAL.md/json`。

## 2026-10-13 mustard 接触恢复更新

用户追加要求：先使用 mustard，接触允许短暂中断，研究丢失后的恢复；仍仅运行 MCC。
本轮模块 `NEW_I04_MUSTARD_RECOVERY`，状态 `EVALUATED / FULL_COVERAGE_INCOMPLETE`。
冻结协议 `I04_MUSTARD_RECOVERY_PROTOCOL.md`；A/B 共享 measured-history recovery，
不把新增强结果标作旧 Pure Passive MCC。物理沿用 mustard 256-part V-HACD，
GT/覆盖包含完整三个 mesh components、388 required points，runtime 不重复 scale。
全指失联不是 episode FAIL；100 ms 去抖、150 ms 稳定确认、8 s 恢复超时后继续。
≥8 N 局部卸力与持续 >12 N/20 ms 等独立安全保护保留。
已完成 seeds 7/11/19 的自然/撤离 A/B 遍历与 hold 恢复开关消融，共 18 次真实运行，18/18 达到预算。
自然遍历平均接触率 A/B 为 99.57%/97.45%，平均覆盖 5.062%/3.631%。
撤离遍历均最终恢复；持续失联段恢复时间 A 16.394–16.478 s、B 4.048–5.562 s，
之后仍有新增覆盖。hold 隔离对照恢复开/关为 3/3、0/3，开启时稳定恢复 3.796–3.874 s。
两侧初态和撤离前完整保存历史逐元素一致。自然/挑战中所有记录的安全接触 gap 最终均稳定恢复，
但短 gap 有自发恢复，不能全部归因于显式恢复层。A 历史参考较旧、四指参与不足仍待改善；完整覆盖未完成。
28 项相关检查通过；source/manifest/model/256 parts/trace/paired-state/causal-recovery 审计全部 VERIFIED。
证据 `evidence/2026-10-13_NEW_I04_MUSTARD_RECOVERY.md`；
汇总 `generated/new_i04_mustard_recovery_review_20261013/index.html`，含原始记录、曲线与回放。
下方 Bunny v1 冻结协议与证据不改写，DPRef 延后。

## 2026-10-13 Bunny v1 历史实施更新

用户明确授权运行新 I04，随后指定仅运行 MCC、DPRef 延后。
当次模块：`NEW_I04_GEOMETRIC_MCC`，状态 `EVALUATED / FULL_COVERAGE_INCOMPLETE`。
有效设计为 `I04_REDESIGN_M06_M12_MIGRATION.md`；冻结运行协议为
`I04_GEOMETRIC_MCC_PROTOCOL.md`。A-MCC/B-MCC 共用 GT 几何、初始化、规划框架和
MCC/硬件安全参数。旧 G2/G3 和下方 2026-08-25 Mx 看板只保留为历史 Explicit 证据，
不阻塞新 I04。只新增功能层 runtime，历史实现和用户文件保留。
本轮不运行 DP、不训练新模型、不进入 GPIS。

已完成 3 seeds × A-MCC/B-MCC 共 6 次实际 MuJoCo 评测。
共同 60 s 总预算（3 s acquisition）；6/6 因持续高力提前停止，未完成完整表面覆盖。
A-MCC 实际 treatment 为 5.418–5.998 s，平均面积覆盖 4.771%；
B-MCC 为 14.536–34.988 s，平均面积覆盖 7.284%，包含全指失联。
这些末段覆盖来自不同执行时长；严格等时长前缀诊断单独保存。
21 项相关回归检查通过；source/trace/paired-state/method-boundary 审计通过。
证据：`evidence/2026-10-13_NEW_I04_MCC.md`；
报告：`generated/new_i04_mcc_formal_20261013/index.html`。
本轮 MCC-only 请求已完成评测；DPRef 继续 `DEFERRED_BY_USER`，不将任务标为 PASSED。

## 当前状态

| 项目 | 当前值 |
| --- | --- |
| 计划版本 | `v4.2` |
| 最近更新 | `2026-10-19` |
| 授权状态 | `I04B_GOAL_ADOPTED / AB_COMMON_CONDITIONS_REQUIRED / MCC_ONLY` |
| 实施状态 | `GOALS_DEFINED / SHARED_RUNTIME_ALIGNMENT_PENDING` |
| 当前活动模块 | `I04_AB_WHOLE_HAND_ALIGNMENT` |
| 当前 Gate | `I04B_DEVELOPMENT_COVERAGE_VERIFIED / NEW_PAIRED_EVALUATION_NOT_RUN / DP_DEFERRED` |
| 固定环境 | `handcomp` |
| 本轮 Python | `D:\Environment\Anaconda\envs\handcomp\python.exe` |

下文 2026-08 的 Mx、Explicit I04、G2/G3 状态和 Python/Linux 命令保留为历史证据；
新 I04 的目标与A/B共同条件以顶部2026-10-19更新及 `I04_AB_WHOLE_HAND_PROTOCOL.md` 为准；
已有运行的结论仍以其原协议和对应case原始证据为准。

本轮已完成 M0–M3 的 FR3+Leap Hand 扩展、M04 的 MCC 控制链和 central-palm mount 修正。
FR3+Leap 模型为 `7 arm + 16 finger = 23 DoF`；MCC 初始 hand q 精确取自原 DP 视频
`t=2.000 s`。mount site 位于 palm mesh XY 分数 `[0.506,0.541]`，interface alignment
error 约 `2.6e-10 m`，不再把偏置 body origin 安到电机区域。

两个 MCC 单元均完成 3 个冻结 paired episodes，因此执行状态为 `EVALUATED`。E05 只比较
策略性能并报告参考限制越界，不再给策略写 `PASS/FAIL` 或 `MET/NOT_MET`。本次正式协议关闭
gravity 以隔离接触控制，结果不能外推为 gravity-on 或硬件性能。

上一版 standalone DP 的数据、因果 observation schema、训练量和短时评测均未通过用户
验收。旧代码、数据、checkpoint、视频和数字不复用。其后的 DP-direct v1 协议冻结了 observation、
force TCN、relative chunk、authority QP、runtime guard、dataset contract 与 diffusion core
已实现并通过单测。此前 hypothetical object motion 与临时 IK/linear repair 的错误链及其
生成数据已清理，不再提供复现入口。新的 3 秒真实
forward physical→spatial inverse→FR3+LEAP physical replay 已通过 raw replay gate：时间不反转、
原 forward finger command 数值原序复用、replay force/contact 重新测量、finger repair 为零。
但该早期 forward collector 使用 simulator Fingertip MCC，所以只能作为 Diag-MCC pipeline
diagnostic。其后已用三条通过整轨迹 hard gate 的 12 s TRAIN episode 在 RTX 4090 D 上完成
Diag-MCC-only learning-chain 验证，并在独立 12 s EVAL episode 上完成 10.5 s
`Finger DP-direct + Wrist MCC` 闭环，`D-Gate=PASS`。随后 non-MCC forward oracle、Dataset-I
I-Pilot20、formal pool、nested I20/I100 CUDA training 与 I20 held-out physical validation 均已
完成；I20 已用于 Exp. 1 DP-direct E05。已有磁盘目录仍保留历史 `d20/d100/pilot20` 名称。

2026-08-24 架构现改为：Finger DP 生成 nominal multi-finger trajectory/reference，shared Finger
MCC 负责高频柔顺执行。旧 DP-direct 不删除，作为 Exp. 1 架构消融。新的双头 DPRef、Role
Interpreter、Dataset-I I20/I100 重标注、CUDA I100 training 和 Exp. 2 已实现并运行。2026-08-25
又加入普通 Plain whole-hand MCC 绝对参考，并把 shared stack 从旧 8 N-priority profile 调整为
contact-priority profile。当前四策略 aggregate 中，Plain 的 continuity/平均接触数/`N_c>=2`
supported traversal 为 `0.992/3.156/138.87 mm`；严格共享栈三者中 DPRef 最好，为
`0.988/2.450/126.09 mm`，但其四指同时接触率仅 `14.86%`。旧 `G1a=PASS` 仅作 pre-retune
provenance；E05 只报告接触性能及持续/多指高力诊断，不形成策略 Gate。validation 仍缺 RELEASE。
Exp. 3 已移出 E05，安排在 I05 之后作为最终 active-planner dimensionality ablation。

旧 fixed-palm `E05-PHY-v3` evaluator、协议、报告和生成结果已经退役。其通用复杂曲面与
MuJoCo scene primitives 仍由当前 FR3+LEAP E05-F/H 使用，因此作为共享环境代码保留，
不再形成独立 MCC 版本或独立实验结论。

2026-08-24 用户明确授权在 DP 尚未就绪的条件下，只沿 Explicit MCC baseline 实现
M06–M12。本轮因此冻结并执行 `M06_M12_MCC_BASELINE_PROTOCOL.md`：七个模块均完成模块级
`EVALUATED / MET`，但该授权只豁免实现顺序，不把 G1 写成通过，也不授权 I01、DP/Main
planner integration、G2/G3 或 GPIS。数值验收使用 analytic plane + deterministic
linearized validation backend；Bunny 仅作为复杂物体展示。

2026-08-24 用户随后明确授权启动 Bunny 物理 I01。`I01_BUNNY_PROTOCOL.md` 已在实现和正式
运行前冻结；该授权只允许 Geometry-Oracle + Explicit MCC baseline 的 I01-A/I01-B，
不使用 DP、不把历史 G1 改写为通过。正式 3-seed paired MuJoCo 结果为
`EVALUATED / MET`：fixed-contact 中位进度 `19.885 mm`，variable-contact 中位进度
`58.429 mm`，优势 `38.544 mm`；variable 3/3 达到 primary pass 并完成实测
`4 -> 3 -> 4` handover，因此本 Bunny 协议下 `G2=GO`。G3 未开始。

2026-08-24 用户在审阅 I01-A/I01-B 区别后明确授权继续评测 I02/I03。实现前已冻结
`I02_I03_BUNNY_PROTOCOL.md`：I02 只比较一次 `12 mm` committed REPOSITION 与
`3 x 4 mm` fresh-snapshot replanning；I03 只比较相同 Beam 是否启用 prediction-only
ShadowSucc terminal filter。两者继续使用 Bunny Geometry-Oracle + Explicit MCC baseline，
DP/DPRef/GPIS 关闭；正式结果产生前不预先宣告 G3。

正式 `3 seeds x 4 cells` 已完成且 source-stability 为 PASS。I02 的 LONG/SHORT 均 3/3
完成共同任务和实测 handover；支持距离中位数为 `101.767/101.808 mm`，末端预测误差中位数
为 `1.476/1.467 mm`。SHORT 虽有每条恰好 3 个 REPOSITION certificate + fresh barrier，
但未达到冻结的 `<=0.8*LONG+0.25 mm` 改善阈值，故 `I02=EVALUATED / NOT_MET`。I03 的普通
Beam 3/3 选择 `SLIDE(3)` 并在真实 barrier 后成为 dead end；ShadowSucc 3/3 选择
`SLIDE(1)`、0 dead end、3/3 handover，支持距离中位优势 `94.014 mm`，故
`I03=EVALUATED / MET`。G3 要求二者同时 MET，因此 `G3=NO_GO`，禁止进入 GPIS。

2026-08-24 用户进一步冻结 I04 的核心定义：I04 从 active exploration 中拆出，命名为
`Oracle Next-Point Whole-Hand Contact Traversal`。固定的是完整 Bunny surface graph、必访集合
和完成条件；每个 `g_k=(x_k*,n_k*,t_k_out,epsilon_g,epsilon_n)` 必须依据 M06 barrier 后的当前
真实手位置/contact state 与剩余节点在线选择，Oracle 不输出 target finger。I04 累计集成 I01
的 variable contact、I02 的短 prefix + real-root replan 和 I03 的 terminal viability；Explicit 与
DPRef/Role 各自决定 finger assignment、contact transition 和 handover，并共享 transactional
execution、真实 barrier、MCC 与 MAKE-before-BREAK guard。ARRIVE 只由真实 mesh-geodesic
contact 与法向一致性判定。当前只冻结核心协议，未授权实现、训练或运行，数值层仍为 TBD；
完整定义见 `I04_ORACLE_NEXT_POINT_PROTOCOL.md`。I04 不包含 uncertainty、frontier、information
gain 或 next-best-touch selection，这些仍只属于 GPIS active exploration。

2026-08-25 状态更新：I04 Explicit MCC development implementation 已累计接入 full-mesh
surface graph、M07–M12、M10 certificate、M06 barrier、MCC/guards、benchmark 和 visual demo。
三轮保存物理回归最多完成 `7/274` goals；90 s endurance 完成 `6/274`，接触连续率
`99.9839%`，但在双锚点后的 WRIST candidate optimization 停滞。因此 I04 当前是
`PHYSICAL_REGRESSION_INCOMPLETE`，不是 full traversal completed；DPRef/GPIS 仍关闭。完整续作
证据和下一步见 `I04_RESUME_CHECKPOINT_2026-08-25.md`。

## 核心推进原则

每个模块必须先单独证明自己工作，才能进入下一层集成：

```text
A. Low-level Control Validation
                 |
                 v
B. Geometry-Oracle Planning Validation
                 |
                 v
C. GPIS Unknown-Object Exploration
                 |
                 v
D. Main Method vs. Explicit Baseline
```

任何上游模块未通过时，不得通过调整下游模块掩盖问题。Full system 失败时，必须能
沿模块证据反查到具体接口、控制器、规划器、证书或执行器。

## Main 与 Explicit Baseline 的隔离边界

后续实现必须把两条方法隔离：

- **Main method**：`GPIS/Oracle Wrist-only Planner + Finger DP Reference Generator + shared
  Wrist MCC + shared Contact Force Coordinator + shared Fingertip MCC`；
- **Explicit baseline**：`Contact-Mode/whole-hand Planner + explicit fingertip trajectory/contact
  modes + the same Wrist MCC + the same Contact Force Coordinator + the same Fingertip MCC`；
- **共享部分**：状态契约、SurfaceModel 接口、RobotIO、runtime guards、对象、初始
  状态、wrist motion budget、Wrist MCC、Fingertip MCC、desired resultant-wrench convention、
  日志和 evaluator；
- **唯一核心变量**：Explicit 在线优化 finger trajectories/contact modes；Main 在线只优化
  wrist trajectory，并由 DPRef 生成 finger nominal references；
- **禁止隐藏分支**：显式 fingertip/contact-mode planner 不能成为 Main runtime fallback，DPRef
  也不能进入 Explicit 分支偷偷替换失败规划。

共享 Fingertip MCC 只解决高频法向力和小局部法向误差，不负责决定切向轨迹、可达性、碰撞、
接触拓扑或未来 viability。旧 DP-direct 分支只保留为 Exp. 1，不属于 Main runtime。

## 状态标记

主表只使用以下状态：

- `NOT_STARTED`：没有实现，也没有实验；
- `PROTOCOL_FROZEN`：接口、阈值、对象、seed 和 evaluator 已冻结，但尚未执行；
- `CORE_PROTOCOL_FROZEN`：问题定义和不可变语义已冻结，但数值阈值、fixture 或 schema 仍待
  单独冻结；该状态不授权实现或实验；
- `IN_PROGRESS`：已明确授权且正在实现或测试；
- `EVALUATED`：评测已完整执行；E05 comparative policies 只报告指标、相对优劣和参考限制越界；
- `EVIDENCE_ONLY`：已执行的预验证/诊断证据，不属于当前正式协议，不能解锁 Gate；
- `DEFERRED`：用户明确要求本轮不实现、不评测，且不作为本轮完成条件；
- `BLOCKED`：证据表明当前依赖或模块阻塞；
- `PASSED`：达到预先冻结的通过标准，并已登记证据；
- `FAILED`：模块/前置验证未达到冻结的有效性或安全标准，禁止推进依赖模块；已完整执行
  的比较实验不会仅因某个方法性能差而标成 `FAILED`；E05 只保持 `EVALUATED` 并描述性能。

不得把“代码能 import”“单次视频看起来正常”或“下游暂时能运行”记为 `PASSED`。

## 总任务看板

| 顺序 | ID | 模块/实验 | 依赖 | 状态 |
| ---: | --- | --- | --- | --- |
| 0a | P0 | 原 hand-only 统一测试环境、状态和日志 | 无 | `PASSED` |
| 0b | M0-FR3 | 23-DoF 全机器人状态/动作/wrench/log contract | P0 | `PASSED` |
| 1a | M01 | Oracle SurfaceModel 解析接口 | P0 | `PASSED` |
| 1b | M01-FR3 | live FR3 capsules 与 MuJoCo pad/object narrow phase | M0-FR3, M01 | `PASSED` |
| 2a | M02 | Fingertip MCC：Static/Sliding/Curved | P0, M01 | `PASSED` |
| 2b | M02-FR3 | moving-wrist 四指 MCC、signed coordinated error | M0-FR3, M02 | `PASSED` |
| 3a | M03 | hand-only Runtime Guards | P0 | `PASSED` |
| 3b | M03-FR3 | arm/finger 分组 guards 与全局/局部处置 | M0-FR3, M03 | `PASSED` |
| 4d | M04-DP-direct | Force-history direct Finger DP + non-MCC verified-inverse Dataset-I | M0-FR3, M03-FR3 | `EVALUATED`（I20/I100 trained；Exp. 1 ablation） |
| 4e | M04-DPRef | Learned multi-finger nominal trajectory/reference generator | M02-FR3, M04-H, M04-DP-direct data | `IMPLEMENTED / ROLE_COVERAGE_LIMITED` |
| 4r | M04-R | 23-DoF FR3+Leap MuJoCo plant | M0-FR3 | `PASSED`（central mount + exact 2 s posture） |
| 4w | M04-W | palm pose IK、wrench estimator 与 6D Wrist MCC | M04-R, M03-FR3 | `PASSED` |
| 4c | M04-C | Resultant/internal Contact Force Coordinator | M01-FR3, M02-FR3, M04-W | `PASSED` |
| 4h | M04-H | Wrist MCC + coordinated Fingertip MCC integration | M04-C | `PASSED` |
| 5pre | E05-PRE-FMCC | Fixed-palm Fingertip-MCC 物理预验证 | M01–M03 | `EVIDENCE_ONLY` |
| 5a | E05-F-MCC | 规定式 FR3 wrist + 四指 Fingertip MCC | M04-R, M02-FR3 | `EVALUATED`（见越界统计） |
| 5b | E05-H-MCC | FR3 Wrist MCC + coordinator + Fingertip MCC | M04-H | `EVALUATED`（Exp. 1） |
| 5c | E05-F-DP-direct | standalone Finger DP-direct diagnostic | M04-DP-direct, M03-FR3 | `NOT_STARTED`（非主结果） |
| 5d | E05-H-DP-direct | shared Wrist MCC + direct Finger DP + authority filter | M04-DP-direct, M04-W | `EVALUATED`（Exp. 1 ablation） |
| 5p | E05-H-PlainMCC | 普通解析 whole-hand MCC 绝对参考 | M04-H | `EVALUATED`（Exp. 2 absolute reference） |
| 5e | E05-H-PassiveMCC | shared Wrist/Finger MCC + passive nominal reference | M04-H | `EVALUATED`（Exp. 2） |
| 5f | E05-H-ReactiveMCC | shared Wrist/Finger MCC + causal heuristic reference | M04-H | `EVALUATED`（Exp. 2） |
| 5g | E05-H-DPRef+MCC | shared Wrist/Finger MCC + learned nominal reference/role | M04-DPRef | `EVALUATED`（Exp. 2；role coverage limited） |
| 6 | M06 | Transactional Prefix Executor | G1a for formal integration, M02, M03 | `EVALUATED / MET`（prior MCC-only module waiver） |
| 7 | M07 | ContactModeGraph | P0 | `EVALUATED / MET`（MCC-only module scope） |
| 8 | M08 | CheapCert | M01, M07 | `EVALUATED / MET`（MCC-only module scope） |
| 9 | M09 | ContinuousOptimize | M01, M02, M08 | `EVALUATED / MET`（linearized validation backend） |
| 10 | M10 | ExactPrefixAudit | M01, M03, M06, M09 | `EVALUATED / MET`（module scope） |
| 11 | M11 | Lazy Beam Search | M07–M10 | `EVALUATED / MET`（H2/H3 module scope） |
| 12 | M12 | Shadow Terminal Viability | M08, M11 | `EVALUATED / MET`（module scope） |
| 13 | I01 | Oracle Continuous Traversal | M01–M12 | `EVALUATED / MET`（Bunny MCC-only；`G2=GO`） |
| 14 | I02 | Receding-horizon prefix 对比 | I01 | `EVALUATED / NOT_MET`（3/3 均完成，但稳健性改善阈值未达到） |
| 15 | I03 | Terminal viability 对比 | I01, M12 | `EVALUATED / MET`（dead end `3 -> 0`） |
| 16 | I04 | Oracle Next-Point Whole-Hand Contact Traversal | I01–I03, M04-H, M04-DPRef | `EXPLICIT DEV IMPLEMENTED / FULL TRAVERSAL INCOMPLETE` |
| 17 | M13 | GPIS SurfaceModel | G3, M01 | `NOT_STARTED` |
| 18 | M14 | GPIS Active Exploration | M13 | `NOT_STARTED` |
| 19 | I05 | GPIS Main vs. GPIS Baseline | G4, M14 | `NOT_STARTED` |
| 20 | I06 / Exp. 3 | Final active-planner dimensionality ablation | I05 | `NOT_STARTED` |

## Phase 0：统一测试环境

### 目标

只冻结后续所有模块共享的环境、状态、动作、日志和 evaluator，不验证算法。

### 环境约束

- 固定使用 `handcomp`，不新建环境；
- 第一阶段物体固定，不做被推动物体的 SE(3) tracking；
- 对象按以下顺序增加复杂度：

```text
Plane -> Cylinder -> Sphere -> Box/Rounded Box -> Free-form Surface
```

- 在 Plane/Cylinder/Sphere 尚未通过前，不引入 ShapeNet 或复杂自由曲面。

### 统一状态

所有 planner/controller 从同一真实状态格式读取：

```text
z_t = (X_H, q, dq, f_tip, c_tip, E_fail, SurfaceModelVersion)
A_actual(t) = {i | c_tip[i] == CONTACT}
```

如需表面运动量，作为版本化可选字段 `v_surface` 加入，不能由不同模块自行解释。
`A_actual` 只来自真实测量；`A_expected`、预测 contact mode 和 optimizer state 均无权
覆盖它。

连续探索的硬条件是：

```text
for all t: |A_actual(t)| >= 1
```

不是要求四根手指始终全部接触。最后一个旧 anchor 只有在 replacement contact 已经
真实确认后才能 BREAK。

### 必须统一记录的日志

- monotonic timestamp、episode、step 和 seed；
- wrist pose/twist/wrench；
- finger q/dq、上一实际动作和当前命令；
- fingertip positions、forces、contact flags；
- `A_actual` 与预测 contact set；
- planned trajectory、committed prefix 和 prediction-only suffix；
- transaction id、executor state、micro-barrier state；
- blocked reason 与 blocked evidence；
- `SurfaceModelVersion`；
- planning/optimization/audit/execution latency；
- collision distance、joint margin、anchor margin、reach margin；
- safety override、contact event 和 certificate id。

执行状态 `RUNNING/DONE/BLOCKED/CANCELLED` 必须与 fingertip 状态
`CONTACT/FREE` 分开记录。

### Phase 0 通过条件

- schema、单位、坐标系、四元数顺序和采样率有版本号；
- Oracle、GPIS、DP、MCC、planner 和 evaluator 使用相同 contract；
- 一条短 mock episode 可以完整写出并重放日志；
- 所有阈值和 evaluator 在第一次正式实验前冻结。

### M0-FR3 扩展（已通过）

新增 `fr3-leap-modules.v1`，不破坏原 `hand-modules.v1`。它固定记录 `arm_q/dq[7]`、
`finger_q/dq[16]`、palm pose/twist、wrist wrench 的 frame/reference/acting-on/source、arm
external torque、四指 contact force/vector/point/normal/validity，以及 actuator/sensor/controller
状态。`A_actual` 仍只从 measured contact state 导出。JSON 与 JSONL round-trip、shape、单位、
NaN/validity 和 actual-contact authority 均有回归测试。

## Module 1：Oracle SurfaceModel

### 目标

先完全去掉未知几何误差，只验证 planning/control 所需的几何接口。

### 最小接口

```text
querySurface(x)
queryNormal(x)
queryClearance(link_or_swept_geometry)
sampleContactCandidates(finger_id)
queryUncertainty(x)
version
```

Oracle 使用 simulator/analytic ground truth：

```text
uncertainty(x) = 0
m_uncertainty = 1
version = fixed immutable id
```

### 独立测试

1. **Point-surface distance**：随机采样点，对比 analytic/simulator GT；
2. **Surface normal**：报告 `acos(n_pred dot n_GT)`；
3. **Link clearance**：随机 hand configuration，对比 simulator collision engine；
4. **MAKE candidates**：检查点在表面、法向正确、对应 fingertip 基本可达。

### 通过条件

误差阈值必须在运行前冻结。四类测试均通过后，才能把 Oracle 交给控制和规划模块。
这一阶段不做 exploration，也不将 analytic oracle 结果冒充未知物体能力。

展示层可额外使用统一放大的 Stanford Bunny/YCB mesh，但不得取代 Plane/Sphere/Cylinder/
Box/RoundedBox 的解析验收，也不得把展示 scale 当作 YCB 真实物理尺寸实验。

### M01-FR3 扩展（已通过）

`FullRobotGeometryAdapter` 从 live `MjData` 生成 world-frame FR3 link capsules，并把它们送入
同一 Oracle clearance contract；四个 fingertip belly pad 到 fixed object 的精确窄相距离
使用 MuJoCo `mj_geomDistance` 与 witness points。approximate capsule 只用于快速几何查询，
不冒充 exact trajectory certificate。

## Module 2：Fingertip MCC（Explicit 与 Main 共享执行层）

### 目标

给定正确 fingertip trajectory `p_plan_i(t)`，验证 analytical controller 能否跟踪
切向运动并稳定调节法向力。

### 控制形式

```text
M_i * dd(delta_d_i) + D_i * d(delta_d_i) + K_i * delta_d_i
    = f_des_i - f_i

p_cmd_i = p_plan_i + delta_d_i * n_i
```

MCC 只修改法向，不改变 planner 的切向意图。所有 offset、速度、加速度和 joint
command 必须限幅并记录。

### Module 2A：Static Contact

- 单指接触固定平面；
- 使用至少三个冻结的 `f_des`；
- 记录 steady-state error、overshoot、settling time、oscillation；
- 核心指标：`force RMSE` 与 `P(force > force_max)`。

### Module 2B：Tangential Sliding

- 使用已知切向轨迹 `p(t) = p0 + s(t) * tangent`；
- MCC 只调法向；
- 报告 tangential tracking error、force RMSE、contact loss 和 slip。

### Module 2C：Curved Surface

- 依次使用 cylinder 和 sphere；
- 法向 `n(t)` 持续变化；
- 验证 `f_n` 对 `f_des` 的跟踪、曲率切换稳定性和 contact continuity。

### 通过条件

2A、2B、2C 必须分别通过预冻结标准。单指或平面成功不能替代曲面验证。

### M02-FR3 扩展（已通过）

原 `FingertipMCC.step(...)` 保持兼容；新增 `step_force_error(...)` 接收 coordinator 已投影的
signed force error。四指 wrapper 使用 moving palm frame、逐指 direction、active-contact mask
和 contact transition reset，避免将固定 palm 假设带入 FR3。真实接触 pad 位于指尖指肚，
包括经过朝下初始化的拇指 pad。

## Module 3：Runtime Guards

### 目标

独立验证可观测条件下的 over-force、joint limit、known self-collision、no-progress 和
suspected non-tip blockage。不得假设存在 non-tip tactile sensor，也不得声称直接测得
未知物体与中间指节的碰撞点/法向。

### 必测案例

- **A — Free motion**：不得误报 `BLOCKED`；
- **B — Middle phalanx blockage**：`f_tip ~= 0`，但 command 有进展而 actual motion
  长时间接近零，应输出 `SUSPECTED_OBJECT_BLOCKAGE`；
- **C — Joint limit**：明确输出 `JOINT_LIMIT`；
- **D — Fingertip over-force**：在冻结 latency 内进入保护；
- **E — Known self-collision**：输出独立硬失败原因。

### 指标

blockage detection rate、false-positive rate、detection latency、over-force response
latency。`NO_PROGRESS` 与 `SUSPECTED_OBJECT_BLOCKAGE` 是局部证据，不得永久写成几何事实。

### M03-FR3 扩展（已通过）

全机器人 guard 独立累计 7-DoF arm stall 与四个 finger group stall，避免整体 norm 掩盖单指
阻塞；同时检查 stale/invalid sensor、tip over-force、wrist wrench、arm external torque、
known collision、arm/finger joint limit、controller limit 与 actuator saturation。处置明确区分
`FINGER_LOCAL` 和 `GLOBAL_SAFE_HOLD`，且仍不伪造未知 non-tip collision 的位置或法向。

## Module 4：FR3+Leap 整手 MCC

### Module 4DP-direct：已完成的 direct-controller 架构消融

`DP_CONTROLLER_V1_PROTOCOL.md` 现归档为 Exp. 1 的 DP-direct 协议。其 causal force history、
shared TCN、relative command chunk、Action Authority Filter、non-MCC Dataset-I、CUDA I20/I100
training 和 E05 结果均保留，用于回答“DP 能否直接替代 low-level Finger MCC”。

数据采用 spatial inversion，而非 temporal reversal。`Diag-MCC` 只验证 learning pipeline；正式
Dataset-I 使用 non-MCC privileged forward oracle，并严格区分 zero-repair `RAW_VERIFIED`、
`REPAIRED` 和 `REJECTED_DIAGNOSTIC`。已有磁盘路径中的 `d20/d100/pilot20` 保持不变，但文档
逻辑名称统一为 I20/I100/I-Pilot20。

Exp. 1 已完整评测。当前 DP-direct 的 contact continuity、平均 contact 数、force RMSE 和
traversal 均弱于 MCC，而 wrist force-z RMSE 较低；这不否定 learned finger trajectory
generation。它只说明当前 direct policy 不适合替代高频 analytical compliance。DP-direct
不再作为 Main runtime，也不再作为 planner integration Gate。

### Module 4DPRef：Learned Finger Reference Generator（已实现，role coverage limited）

新的 Main 将 DP 上移为 nominal multi-finger trajectory/reference generator：

```text
hand/contact state + future wrist plan -> shared encoder
 -> diffusion head: DPRef nominal finger trajectory
 -> categorical head: per-finger role intention
 -> shared Fingertip MCC compliant execution
```

active contact 的 nominal tangential/relative motion由 DPRef 生成，coordinated normal/internal
force 由 Finger MCC 调节，collective/resultant force 由 Wrist MCC 调节；free finger 的
reposition/MAKE nominal trajectory 可由 DPRef 完整生成。Role head 与 diffusion head 分开训练；
deterministic interpreter 只允许 `KEEP -> RELEASE -> FREE -> MAKE -> KEEP`。RELEASE 通过
`f_des -> 0` 平滑 ramp 卸力，最后接触 veto 与真实 contact confirmation 始终生效。

已实现 `dpref_dataset.py`、`dpref_policy.py`、`dpref_train.py`、`dpref_reference_sources.py`、
`reference_interpreter.py` 和 Exp. 2 evaluator。现有 I20/I100 RAW_VERIFIED 数据完成了 q_nom +
temporally confirmed role 重标注；旧 DP-direct checkpoint 没有复用。I100 在 RTX 4090 D 上完成
10,000 updates，continuous q_nom validation 已记录；validation 无 RELEASE 且 MAKE 只有 20 个
标签/60% accuracy，因此当前 checkpoint 的 role/hand-over 适用范围受限，不能宣称已泛化。

### Module 4R：FR3+Leap dynamic plant

已实现可控的 MuJoCo 整机模型：

- 7 个 FR3 position actuators 与 16 个 Leap Hand position actuators；
- `(nq,nv,nu)=(23,23,23)`，物体固定在 world，`nmocap=0`；
- flange 轴显式对准 central palm mesh，而不是偏置 body origin；四个 belly pad 绑定真实
  fingertip body；
- 初始拇指关节使 thumb belly 朝向并接触曲面，不再缺失拇指接触；
- 正式 E05 的 wrist trajectory 由 FR3 关节执行。

### Module 4W：FR3 wrist control 与 Wrist MCC

已实现 palm pose 的 damped-least-squares IK、规定式 wrist tracking、通用 6D admittance
Wrist MCC 和基于 arm constraint torque/palm Jacobian 的 wrench estimator。wrench 日志明确
frame、moment reference、acting-on 和符号；本次 E05 仅选择 collective-normal translation
方向，切向二维探索仍由 nominal trajectory 主导。

若 `f_des` 表示 hand-on-object force，则 `w_obj_des=G f_des`、
`w_hand_des=-G f_des`。当 `G f_des=0` 时，internal squeeze 与 zero wrist resultant 并不
冲突；Wrist MCC 不再默认对所有非零 fingertip force 追零。

### Module 4C：Contact Force Coordinator

实现使用 normal-force basis `f=B lambda` 与 effective map `H_A=G_A B_A`：

```text
normal-force error -> resultant component -> Wrist MCC
                   -> internal component  -> Fingertip MCCs
```

weighted generalized inverse 构造 resultant/internal projectors；`A_actual` 只由带迟滞的
真实 contact measurement 决定，未确认 contact 不进入承载 wrench 的 map。实现记录 rank、
singular values、condition、reconstruction error 和 internal leakage，并在 contact-set
transition 时 blend/reset，避免旧 projector 保留执行影响。

### Module 4H：整手集成

- `E05-F-MCC`：关闭 Wrist MCC，FR3 跟踪规定轨迹，四指 MCC 接收完整 local force error；
- `E05-H-MCC`：Wrist MCC 接收 resultant error，active fingers 只接收 internal error；
- 未确认接触的手指只允许局部 approach/recovery，不作为 wrench anchor；
- M03-FR3 guards 在同一 loop 中监测 arm/finger stall、force、wrench、torque、joint、sensor
  和 actuator saturation。

实现、符号和审计细节见 `WHOLE_HAND_COMPLIANCE_DESIGN.md`；复现入口见 `Module/README.md`。

## Experiment 5：E05 的两组控制层描述性评测

E05 现在明确包含两个不同问题；两组都只报告性能和参考限制越界，不设置策略
`PASS/FAIL` 或 `MET/NOT_MET`。完整 active-planner 对照另列为 I06 / Exp. 3。详细协议、
公平条件、指标与统一网页入口只以 `E05_EVALUATION_PLAN.md` 为准。

### Exp. 1：DP-direct 能否替代 Finger MCC（已完成）

```text
E05-H-MCC
vs.
E05-H-DP-direct
```

这是 low-level controller replacement ablation。三个 15 s paired episodes 已完整执行：

| aggregate 指标 | E05-H-MCC | E05-H-DP-direct |
| --- | ---: | ---: |
| mean contact continuity | 87.30% | 66.69% |
| mean contact count | 3.026 | 1.590 |
| mean force RMSE | 1.381 N | 2.232 N |
| worst peak force | 81.35 N | 103.02 N |
| mean Y traversal | 174.2 mm | 158.2 mm |
| controller P95 mean | 1.35 ms | 12.00 ms |

两边均已 `EVALUATED`。当前 DP-direct 没有表现出替代 MCC 的优势，因此保留为架构消融，不再
承担 main method 或 planner integration Gate。相对 8 N 参考限制，MCC/DP-direct worst peak
分别超出 `73.35/95.02 N`。旧协议 `E05_DP_CURRENT_PROTOCOL.md` 与
`DP_CONTROLLER_V1_PROTOCOL.md` 只用于复现 Exp. 1。

### Exp. 2：Plain + Passive / Reactive / DPRef+MCC（接触优先重评已执行）

```text
E05-H-PlainMCC = ordinary analytical whole-hand MCC absolute reference
E05-H-PassiveMCC = passive/hold nominal reference -> shared Finger MCC
E05-H-ReactiveMCC = causal heuristic reference -> same Finger MCC
E05-H-DPRef+MCC  = learned anticipatory reference -> same Finger MCC
```

后三者共享 Wrist MCC、Contact Force Coordinator、Finger MCC gains、force target、M03、wrist
trajectory、初态、对象、扰动、horizon 和 evaluator。唯一变量是 finger nominal reference 的
来源。Reactive 只可读取当前/过去 state 和 SurfaceModel，不可读取 future wrist plan。该实验
隔离 learned anticipation 是否改善 contact richness、handover 与 supported traversal，而不是
再次比较 DP 与 MCC 谁负责高频 force control。Plain 不经过新 Role/ForceSafety wrapper，只作
普通 MCC 绝对参考，不能参与 reference-source 因果归因。

四路均完成 nominal、low-friction、noisy-observation 三个 15 s 条件。Plain/Passive/Reactive/
DPRef 的 aggregate continuity 为 `0.992/0.972/0.973/0.988`，平均接触数为
`3.156/2.285/2.310/2.450`，`N_c>=2` supported Y 为
`138.87/89.35/86.90/126.09 mm`。

性能上，Plain 的绝对接触保持最好。在严格共享栈三者中，DPRef 的 continuity、平均接触数、
`P(N_c>=2/3)` 和 supported traversal 全部最好；相对最佳解析 source 分别提高 1.51 pp、0.139
个接触和 36.75 mm。但 DPRef 四指同时接触率 `14.86%`，低于 Reactive 的 `27.45%`，第四指
参与不足；validation 无 RELEASE 且 MAKE 覆盖很少，所以不能由当前结果声称 handover 泛化。
MuJoCo force 只作诊断：四种策略均无多指同时 `>8 N`，单个瞬时峰值不决定策略优劣。

统一网页与机器数据位于 `generated/e05_exp1_exp2_review/`。

### Shared low-level readiness provenance

- 旧 `G1a shared-compliance readiness = PASS` 只对应 2026-08-24 的 8 N-priority profile；
- 当前 contact-priority Exp.2 profile 不沿用该 verdict，也不以单个 MuJoCo peak 判定策略；
- Passive/Reactive/DPRef 的公平性来自同一次重评中完全共享的 interpreter/coordinator/MCC/M03；
- 原 `G1b` 策略性能 Gate 已退休；DPRef 的优缺点按连续数值报告，不再阻塞后续模块；
- M06–M12 的 MCC-only 模块级授权和结果不改变 G1a，也不等于完成正式 I01 traversal。

## Module 6：Transactional Prefix Executor

### 目标

在没有 planner 的情况下，人工提交 transaction，验证：

```text
commit short prefix -> execute -> micro barrier -> real snapshot
```

只有当前 transaction 的 committed prefix 有执行权限；prediction suffix 只能 warm start。

### Prefix 类型

- `WRIST_ADJUST`：wrist motion，anchor fingers 使用 Baseline MCC，free fingers hold；
- `FINGER_RECONFIGURE`：wrist hold，selected fingers 异步执行，anchors 使用 Baseline MCC。

### 独立测试

1. **Asynchronous completion**：先完成的 finger 进入 hold，必须等待 micro-barrier；
2. **Blocked finger**：被挡 finger 输出 `BLOCKED`，独立安全手指按规则结束；
3. **Transaction authority**：新 transaction 提交后，旧 transaction 永不恢复权限；
4. **SurfaceModelVersion**：transaction 的 model version 过期时必须拒绝；
5. **Timeout**：planner/executor timeout 使用 `SAFE_HOLD`，不恢复旧 suffix。

下一次 planner root 必须来自 barrier 后的新真实 snapshot。

## Module 7：ContactModeGraph

### 状态与 primitive

四指共有 `2^4 - 1 = 15` 个非空 contact sets。动作集合：

```text
WRIST_ADJUST
SLIDE(i)
REPOSITION(i)
MAKE(i)
BREAK(i)
```

### 纯单元测试

穷举 15 个 mode 与全部 primitive，要求 100% deterministic：

- 不产生 empty contact mode；
- MAKE 只作用于 free finger；
- BREAK 只作用于 contact finger；
- 一个 prefix 最多一次 topology change；
- 同一 prefix 不同时 MAKE 与 BREAK；
- WRIST_ADJUST 与 finger reconfiguration 不同时发生；
- PredictLegal 与 CommitLegal 分离；
- `CommitLegal(BREAK(i))` 依赖真实 replacement contact confirmation。

Handover 必须拆成独立 transaction，例如：

```text
REPOSITION(3) -> MAKE(3) -> BREAK(1)
```

## Module 8：CheapCert

### 目标

快速筛除明显不可行的 `(contact mode, action)`，但不签发执行证书。

### 检查与输出

- mode legality、failed-edge evidence；
- approximate IK/reachability；
- joint/anchor margins；
- coarse collision；
- trust region 与 uncertainty。

输出 `SURVIVE` 或 `REJECT`，以及：

```text
m_anchor, m_joint, m_collision, m_reach, m_uncertainty
```

### 定量验证

随机生成 1,000–10,000 个 candidate edges，以昂贵 exact solver/audit 为参考，报告
TP/FP/FN/TN。首要目标是低 false negative；可以放过垃圾，但不能大量误杀可行解。

## Module 9：ContinuousOptimize

### 第一版边界

暂不使用 information gain，只构造当前 mode edge 的连续轨迹：

- target tracking；
- collision/joint constraints；
- anchor preservation；
- smoothness；
- reachability；
- first-edge commit trust region。

### 分 primitive 测试

分别验证 `SLIDE`、`REPOSITION`、`MAKE`、`BREAK` 和 `WRIST_ADJUST`。对于未完成但仍
安全的 MAKE approach，状态为 `MAKE_PROGRESS(i)`；只有真实 contact 才能改变 topology。

### 指标

optimizer success rate、solve time、final target error、minimum clearance、minimum
joint margin、anchor margin 和 trust-region use。

## Module 10：ExactPrefixAudit

### 权限

只有本模块能签发 `ExecutionCertificate`。CheapCert、optimizer、beam node、suffix 和
ShadowSucc 永远没有执行权限。

### Audit 范围

检查整个 swept committed prefix：joint limit、self-collision、link clearance、phase
exclusivity、anchor assumptions、trust bounds、SurfaceModelVersion 与当前 CommitLegal。

### Adversarial tests

1. 起终点安全但中间碰撞：REJECT；
2. 中途 joint limit：REJECT；
3. replacement contact 未真实确认就 BREAK：REJECT；
4. SurfaceModelVersion 过期：REJECT；
5. 超过 trust region：REJECT；
6. prediction suffix 试图获得权限：REJECT。

## Module 11：Lazy Beam Search

### 目标

搜索有当前 progress 且未来仍有路的 contact-mode sequence，而不是单步贪心。

### 固定搜索顺序

1. shifted suffix warm start；
2. enumerate legal edges；
3. CheapCert；
4. optimize survivors；
5. 按 per-mode quota 保留 diverse top-B；
6. 重复至 horizon `H`。

初期 score 不含 information gain：

```text
S = w_progress * J_progress
  + w_contact  * J_contact
  - w_motion   * J_motion
  - w_risk     * J_risk
  - w_switch   * J_switch
```

### 验证

用 `H=2/3` 同时运行 exhaustive search 与 beam search，比较 score、最优 mode sequence
保留率、计算量和 diversity，以冻结 beam width 与 per-mode quota。

## Module 12：Shadow Terminal Viability

### 目标

拒绝当前合法但会把系统带入无安全 continuation 的 prefix。

### 测试

- **Viable**：`A={1}` 且 finger 2 存在 cheap-feasible MAKE，输出 `VIABLE`；
- **Dead end**：`A={1}` 且其他手指均 joint-limited/colliding/unreachable，输出
  `NONVIABLE`。

terminal state 至少需要一个非平凡 cheap successor；singleton contact mode 必须存在
至少一个 inactive finger 的 cheap-feasible MAKE。Shadow successor 只能预测，不能执行。

### M06–M12 本轮模块证据（2026-08-24）

冻结协议：[`M06_M12_MCC_BASELINE_PROTOCOL.md`](M06_M12_MCC_BASELINE_PROTOCOL.md)。
代码分别位于 `module_6_prefix_executor/` 到 `module_12_shadow_viability/`；统一 benchmark 为
`m06_m12_benchmark.py`，可视化为 `m06_m12_visual_demo.py`。

- M06：异步完成、blocked peer、transaction revocation、stale model 与 timeout 五类场景全部
  通过；micro-barrier 必须等待新的 timestamp；anchor command 使用现有 Fingertip MCC；
- M07：穷举 15 个非空 modes、131 个 legal primitive edges，重复枚举 deterministic；
- M08：4096 个 seeded candidates 对 nonlinear exact reference 的 false-negative rate 为 0；
- M09：`SLIDE/REPOSITION/MAKE/BREAK/WRIST_ADJUST` 各 32 个 cases，valid success rate 为 1；
  长 MAKE 正确输出 `MAKE_PROGRESS` 且不改变 topology；
- M10：正例签发 immutable certificate；中点碰撞、中途 joint limit、未确认 BREAK、stale
  version、trust exceed 与 suffix authority 六个 adversarial cases 全部拒绝；
- M11：beam width 8、per-mode quota 2 在 `H=2/3` 均保留 exhaustive optimum，score gap 为 0；
  只有 edge zero 可送 audit，后续 prefix 强制为 prediction suffix；
- M12：singleton viable/dead-end case 正确，按 finger 去重 successor，ShadowSucc 无证书权限。

机器结果与 provenance：
[`generated/m06_m12_mcc_baseline/summary.json`](generated/m06_m12_mcc_baseline/summary.json)；
审阅页：
[`generated/m06_m12_mcc_baseline/index.html`](generated/m06_m12_mcc_baseline/index.html)。集成
smoke 维持非空真实 contact set 并完成 `search -> audit -> MCC executor -> barrier snapshot`，
但它使用模块验证 plant，不是 I01/G2/G3 结果。

## Integration 1：Oracle Continuous Traversal

第一次完整集成 Explicit Baseline，但不做 active exploration。任务仅为沿 GT surface
尽可能连续移动。

### I01-A：Fixed Contact

强制 `|A|=4`，记录 `L_fixed`。尝试可重复地复现约 `47.97 mm` 的历史 planning
failure，但不得把它描述成物理 reach limit。

### I01-B：Variable Contact Mode

改为 `|A|>=1`，其余条件不变，记录 `L_variable`。核心假设是：

```text
fixed-contact infeasible != variable-contact infeasible
```

只有发现真实可行且具备 continuation 的 mode transition，才算支持该假设。预期证据
包括 `4->3->4`、`3->2->3` 等 handover，以及 `L_variable > L_fixed`。

### Gate G2

Variable mode 必须用受控实验真实突破 fixed-contact 局部可行域；轨迹切短本身不能
被解释为扩大 feasible set。

### I01 Bunny 正式结果（2026-08-24）

冻结协议：[`I01_BUNNY_PROTOCOL.md`](I01_BUNNY_PROTOCOL.md)；证据记录：
[`evidence/2026-08-24_I01_BUNNY_PHYSICS.md`](evidence/2026-08-24_I01_BUNNY_PHYSICS.md)；
审阅页：[`generated/i01_bunny_physics/index.html`](generated/i01_bunny_physics/index.html)。

- paired seeds 为 `7/11/19`，两组共享 Bunny、初态扰动、路径、时间律、MCC、guard 与 evaluator；
- fixed-contact 三次都因持续四指 mode 破坏触发 `FIXED_MODE_LOST`，实际进度中位数
  `19.885 mm`，但始终保持至少一指有效 Bunny 接触；这不是物理 reach limit，也不复现或替代
  历史 `47.97 mm` planning failure；
- variable-contact 三次实际进度为 `58.430/58.423/58.429 mm`，3/3 primary pass，3/3
  完成由 M10 certificate 与 M06 micro barrier 约束的实测 `4 -> 3 -> 4` handover；
- variable 非空接触比例为 `99.956%/99.933%/99.956%`，最大全接触丢失 gap `2 ms`，
  worst valid fingertip force `5.475 N < 8 N`；0 over-force tick、0 non-tip collision tick、
  0 authority violation；共签发 9 张 certificate、完成 9 个 barrier；
- `median(L_variable)-median(L_fixed)=38.544 mm >= 10 mm`，故本冻结 Bunny 协议的
  **G2 判定为 `GO`**。这条 I01 证据自身不决定 G1a/G3；后续独立评测现已得到
  历史 `G1a=ARCHIVED_PRE_RETUNE` 和当前 `G3=NO_GO`。E05 策略结果不再定义为 Gate。

物理碰撞使用从同一 Bunny mesh 生成的 `181 x 181` upper-envelope hfield；可视化和接触
残差审计使用完整变换后三角网格。正式实验为固定物体、gravity-off 的控制隔离仿真，不能
外推到完整非凸三角碰撞、gravity-on、硬件或 sim-to-real。

## Integration 2：Receding-Horizon Prefix

固定 variable contact mode，对比：

- long/full execution；
- short committed prefix + snapshot + replan。

报告 execution failure、blocked、collision、prediction error、completion distance 和
replanning latency。结论只能是 prefix execution 是否提高 closed-loop robustness，
不能声称其自动改变 feasible set。

## Integration 3：Terminal Viability

比较 Beam 与 Beam + ShadowSucc，报告 dead-end count、handover availability、minimum
joint margin、maximum traversal 和 recovery count。

### I02/I03 Bunny 正式结果（2026-08-24）

冻结协议：[`I02_I03_BUNNY_PROTOCOL.md`](I02_I03_BUNNY_PROTOCOL.md)；机器结果：
[`generated/i02_i03_bunny_physics/summary.json`](generated/i02_i03_bunny_physics/summary.json)；
审阅页：[`generated/i02_i03_bunny_physics/index.html`](generated/i02_i03_bunny_physics/index.html)。

- I02-LONG 与 I02-SHORT 均 3/3 达到 `>=100 mm` supported cumulative traversal、3/3
  `4 -> 3 -> 4`；无 over-force、non-tip collision、authority 或 suffix-command violation；
- SHORT 每条恰好有 3 张 REPOSITION certificate 与 3 个 fresh measured barrier；LONG/SHORT
  supported traversal 中位数为 `101.767/101.808 mm`；
- LONG/SHORT 最终 terminal prediction error 中位数为 `1.476/1.467 mm`。冻结改善阈值为
  `0.8 * 1.476 + 0.250 = 1.431 mm`，SHORT 未达到；两组 failure count 又同为 0，故
  **I02=`EVALUATED / NOT_MET`**。这表示短前缀机制有效，但没有证明所要求的稳健性增益；
- I03-BEAM 三次均选 `SLIDE(3)`，真实 terminal joint margin 约 `0.020 rad`，M12 诊断
  `NONVIABLE` 并记录 3 个 dead end；I03-SHADOW 三次均选 `SLIDE(1)`，真实 margin
  `0.0478–0.0480 rad`，有 distinct successors、0 dead end、3/3 handover；
- BEAM/SHADOW supported traversal 中位数为 `7.111/101.125 mm`，优势 `94.014 mm`，故
  **I03=`EVALUATED / MET`**。M12 execution authority 始终为 false；
- 因 I02 与 I03 未同时 MET，**G3=`NO_GO`**。该结论只属于固定已知 Bunny、gravity-off、
  Geometry-Oracle + Explicit MCC baseline，DP/DPRef/GPIS 均未参与。

### Gate G3 / Geometry-Oracle Baseline

只有 Oracle 下能够长期连续 traversal、维持 `|A_actual|>=1`、没有未解释的碰撞/关节/
force violation，并能报告完整 planning/certificate/executor 证据，才允许进入 GPIS。

Gate G3 报告至少包含：

- Safety：contact continuity、collision、joint/force violations；
- Coordination：MAKE/BREAK/handover success、average contacts；
- Planning：latency、nodes、optimizer/certificate success、blocked ratio；
- Viability：dead ends、successor-finger count、terminal margins。

## Module 13：GPIS SurfaceModel

### 替换原则

只替换：

```text
OracleSurfaceModel -> GPISSurfaceModel
```

planner、controller、executor、certificate 和 evaluator 接口保持不变。

GPIS 提供 mean surface、normal、uncertainty 和 version：

```text
surface: mu(x) = 0
normal:  grad(mu) / ||grad(mu)||
uncertainty: sigma^2(x)
```

collision safety 必须使用 uncertainty-aware margin，例如：

```text
d_safe = d0 + beta * sigma(x)
```

### 验证

先沿给定方向 traversal，不做 active exploration。Oracle 与 GPIS 共享 motion 和
evaluator，比较 collision、contact loss、normal error、candidate reachability 和
traversal distance。

### Gate G4

把 GT 替换为 GPIS 后，系统仍需在冻结安全标准下闭环运行。未通过 G4 时不得开始完整
active exploration 对比。

## Module 14：GPIS Active Exploration

第一版 information objective：

```text
J_info = sum_i sigma^2(x_i)
```

之后才考虑 `logdet(Sigma_X)`。比较 random/fixed exploration 与 uncertainty-guided
planner，报告 uncertainty reduction/time、uncertainty reduction/distance、coverage、
F-score、Chamfer 和 active-contact novelty。

## Integration 4：Oracle Next-Point Whole-Hand Contact Traversal

冻结核心协议见 [`I04_ORACLE_NEXT_POINT_PROTOCOL.md`](I04_ORACLE_NEXT_POINT_PROTOCOL.md)。
I04 不做 GPIS active exploration：固定的是 privileged-certified Bunny surface graph、必访节点
和完成条件。每个 surface goal 根据 barrier 后的真实手位置/contact state、剩余节点和共同
OracleRoute contract 在线选择，但不指定 target finger：

```text
g_k = (x_k*, n_k*, t_k_out, epsilon_g, epsilon_n)
```

Explicit contact-mode/finger planner 与 DPRef/Role 分支各自决定哪个 finger 到达、如何管理
workspace 以及如何 MAKE/BREAK；两边共享 surface graph/selector contract、transactional
prefix、真实 barrier、MCC 和 runtime guards。闭环状态分叉后允许目标顺序随各自真实 state
调整，但所有 required nodes 都必须保留到真实到达，不能丢弃难区域。ARRIVE 必须由 MuJoCo
实测 contact 的 mesh geodesic distance 与 surface-normal alignment 确认，`t_k_out` 只作为后继
方向输入，不进入 arrival condition。Oracle 的 privileged feasibility certificate 可以基于当前
真实 root 证明至少存在一个合法 finger/configuration/transition，但 finger identity 和 witness
不得进入 runtime goal。

执行链累计包含 I01 的 variable contact handover、I02 的 short committed prefix + barrier-state
replan 和 I03 的 terminal continuation。Explicit 使用完整 M07–M12 planner；DPRef proposal 至少
共享 M10 execution audit、M06 transaction/barrier、MCC 与 M03 guard。DPRef 分支中 M08/M12
究竟作为共同 filter 还是 diagnostic-only，留待 numerical/method-boundary protocol 冻结，不能
让 Explicit planner 暗中替 DPRef 选择 finger。

I04 比较的是 given-good-next-point 条件下的 whole-hand contact realization ability；它明确排除
uncertainty、frontier、information gain、next-best-touch selection 和 reconstruction。当前
Explicit development implementation 已冻结 mesh fixture、274-goal route、development tolerance、
trace 与 evaluator，并完成局部物理回归；这些数值仍不是最终 formal protocol。完整 required
set 尚未完成，DPRef goal conditioning/checkpoint 与 method boundary 也未实现。

## Integration 5：GPIS Main vs. GPIS Explicit Baseline

最终完整比较：

```text
Explicit = GPIS + Contact-Mode Planner + shared Wrist MCC
         + Contact Force Coordinator + Fingertip MCC
Main     = GPIS + Wrist-only Planner + Finger DP Reference Generator
         + shared Wrist MCC + Contact Force Coordinator + Fingertip MCC
```

报告：

- Reconstruction：F-score、Chamfer、coverage；
- Exploration：uncertainty reduction/sec、coverage/sec、path length、explored area；
- Contact：continuous-contact success、average contacts、contact loss、handover；
- Planning efficiency：mean/P95 latency、solver failure、nodes/candidates、real-time factor。

## Integration 6 / Exp. 3：最终 active-planner dimensionality ablation

Exp. 3 放在 I05 之后。它使用已经完成验证的 active exploration stack，在匹配的 SurfaceModel、
初态、对象、wrist motion budget、receding-horizon protocol、MCC、guard 和 compute budget 下，
只替换在线 planner 的决策空间：

```text
Explicit = optimize wrist + finger trajectories + contact modes -> shared MCC
Main     = optimize wrist only -> DPRef/Role generates finger references -> shared MCC
```

I05 回答两个完整 GPIS 系统最终效果如何；I06/Exp. 3 则作为最后一个受控消融，专门回答省掉
显式 finger/contact-mode planning 后，exploration/contact quality 损失多少以及 planning latency、
变量数和 solver failure 改善多少。DP-direct 不进入该实验。

## 分层论文问题

| 问题 | 对应实验 |
| --- | --- |
| Q0 — DP 能否直接替代高频 Finger MCC？ | Exp. 1：E05-H-MCC vs. E05-H-DP-direct（架构消融） |
| Q1 — learned anticipatory finger reference 是否优于 passive/reactive compliance？ | Exp. 2：Passive-Hold / Reactive-Heuristic / DPRef+MCC |
| Q2 — Variable contact modes 是否扩大 feasible region？ | Oracle Fixed vs. Variable Contact |
| Q2b — 已给定优质下一表面点时，哪种方法能更好地实现整手连续接触到达？ | I04：Oracle Next-Point Whole-Hand Contact Traversal |
| Q3 — 是否需要显式高维 finger/contact planning？ | I06 / Exp. 3：I05 后的 final active-planner dimensionality ablation |
| Q4 — 未知物体上能否高效闭环探索？ | GPIS Explicit vs. Main |

## 系统级 Go/No-Go Gates

| Gate | Go 条件 | No-Go 后动作 |
| --- | --- | --- |
| G1a（历史） | 2026-08-24 8 N-priority shared stack 的 safety/readiness 审计 | 只保留 provenance；不外推到 contact-priority Exp.2 profile |
| G2 | Variable mode 真实突破 fixed-contact 局部可行域 | 检查 mode legality/optimizer/audit，不引入 GPIS |
| G3 | Oracle 下长期连续 traversal 且无 dead end | 修复 planner/executor/certificate，禁止 GPIS 集成 |
| G4 | 换成 GPIS 后仍能安全闭环 | 修复 SurfaceModel/uncertainty margin，禁止 full comparison |

只有通过 G3，才投入主要精力进行 GPIS integration。

当前 Gate 记录：`G1a=ARCHIVED_PRE_RETUNE`、`G2=GO`（仅上述 Bunny I01 冻结协议）、
`G3=NO_GO`（I02 NOT_MET；I03 MET）、`G4=NOT_STARTED`。

## 获得授权后的严格实施顺序

```text
0.  [DONE] M0 full-robot environment/state/action/wrench/log contract
1.  [DONE] M01 Oracle + live FR3 geometry
2.  [DONE] M02 Fingertip MCC + moving-wrist four-finger wrapper
3.  [DONE] M03 hand-only + FR3 grouped Runtime Guards
4.  [DONE] M04-R FR3–Leap plant
5.  [DONE] M04-W 6D Wrist MCC + wrench estimator
6.  [DONE] M04-C/H resultant/internal coordinator and integration
7.  [EVALUATED] E05-F-MCC + E05-H-MCC historical control evidence and limit observations
8.  [DONE / REMOVED] Remove unaccepted legacy DP code and generated data
9.  [DONE] Freeze causal DP v1 architecture and teacher/evaluation protocol
10. [DONE / DIAGNOSTIC] Real forward + spatial-only raw physical replay Diag-MCC gate
11. [DONE / LONG GPU D-GATE PASS / DIAGNOSTIC] 3×12 s TRAIN + independent 10.5 s DP/Wrist-MCC closed loop
12. [DONE / I-GATE PASS] Non-MCC forward oracle + 12 s zero-repair raw replay
13. [DONE / RAW-GATE PASS] I-Pilot20 + RAW/REPAIRED/REJECTED pools
14. [DONE TO I100 / STOP AT I20] Nested I20/I100 CUDA training and object-disjoint held-out validation
15. [EVALUATED / DESCRIPTIVE ABLATION] Exp. 1 E05-H-MCC vs E05-H-DP-direct
16. [DONE / G1a ARCHIVED PRE-RETUNE] Validate the historical 8 N-priority shared stack
17. [DONE / ROLE COVERAGE LIMITED] Freeze DPRef contract; relabel/audit I20/I100; CUDA train
18. [EVALUATED / CONTACT-PRIORITY DESCRIPTIVE COMPARISON] Exp. 2 Plain + Passive / Reactive / DPRef+MCC
19. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] Transactional Prefix Executor
20. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ContactModeGraph
21. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] CheapCert
22. [EVALUATED / MET / LINEARIZED VALIDATION BACKEND] ContinuousOptimize
23. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ExactPrefixAudit
24. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] LazyBeamSearch
25. [EVALUATED / MET / MCC-ONLY MODULE SCOPE] ShadowSucc
26. [EVALUATED / MET / G2 GO / BUNNY MCC-ONLY] Oracle Continuous Traversal
27. [EVALUATED / NOT_MET / BUNNY MCC-ONLY] Receding-Horizon Prefix
28. [EVALUATED / MET / BUNNY MCC-ONLY] Terminal Viability
29. [EXPLICIT DEV IMPLEMENTED / PHYSICAL REGRESSION INCOMPLETE] Oracle Next-Point Whole-Hand Contact Traversal
30. [BLOCKED BY G3] GPIS SurfaceModel
31. [BLOCKED BY G3] GPIS Active Exploration
32. [BLOCKED BY G3] Full GPIS Main vs. Baseline
33. [AFTER I05] I06 / Exp. 3 final active-planner dimensionality ablation
```

不得因为某个后续模块更容易展示而跳过顺序或 Gate。M06–M12 是用户明确授权的 MCC-only
模块实现豁免；随后用户又单独授权并已完成 Bunny I01，所以该实验可独立给出 `G2=GO`。
用户随后单独授权的 I02/I03 已完成；当前 G3=NO_GO，所以不能进入 GPIS。I04 已从 active
exploration 中拆出，Explicit development implementation 已形成但完整 traversal 未完成；它不因
G3 状态被重新解释为 GPIS，DPRef/GPIS 也没有随之解锁。
E05 Exp.1/2 不是系统解锁 Gate；I06/Exp.3 固定在 I05 之后执行。

## 每个模块开始前的 Protocol Freeze

状态从 `NOT_STARTED` 改为 `IN_PROGRESS` 前，必须先登记：

```text
Module ID:
Authorization:
Owner:
Exact scope:
Inputs/outputs and schema version:
Objects and initial states:
Seeds and episode count:
Metrics:
Numerical pass/fail thresholds:
Evaluator version:
Allowed privileged information:
Expected artifact paths:
Timeout/resource budget:
```

没有冻结 numerical threshold 的实验不能事后根据结果选择通过标准。

## 每个模块完成后的 Evidence Record

只有下面内容完整时才能标记 `PASSED`：

```text
Module ID:
Status:
Commit/worktree provenance:
Environment and dependency versions:
Commands:
Input hashes:
Output artifacts:
Metrics and confidence intervals:
Failure cases:
Pass/fail decision:
Downstream modules unlocked:
```

视频和单个成功 episode 只能作为辅助证据，不能替代统计指标与失败案例。

## 主计划更新规则

1. 每次开始工作前先读取本文件的当前状态、依赖和 Gate；
2. 未获得用户明确授权时，只能修改计划文字，不能实施模块；
3. 一次最多有一个 `IN_PROGRESS` 模块；
4. 状态变化必须同时写入证据位置，不得只口头宣告；
5. 上游 `FAILED/BLOCKED` 时，下游保持 `NOT_STARTED`；
6. 任何架构变更先更新本计划并说明对公平比较和论文问题的影响；
7. 历史 suffix、预测 contact、shadow successor 和过期 SurfaceModel 均无执行权限。
