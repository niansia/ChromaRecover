# ChromaRecover

[English](README.md) | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md)

![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-3DA639)
![Experimental Alpha](https://img.shields.io/badge/Status-Experimental%20Alpha-E69500)
![Local-first](https://img.shields.io/badge/Privacy-Local--first-5B5BD6)
[![CI](https://github.com/niansia/ChromaRecover/actions/workflows/ci.yml/badge.svg)](https://github.com/niansia/ChromaRecover/actions/workflows/ci.yml)

ChromaRecover 是一套本地优先的 Python 工具，用于恢复并呈现由细微色彩差异承载的空间结构。它返回 top-k 掩码、叠加图、质量警告，以及明确的 `ok`、`uncertain` 或 `retry_recommended` 状态，不会被迫猜出文字答案。

> Alpha 范围：digital mode 是当前稳定的基线。实验性 Camera Alpha 提供保守的文档透视校正／展平、纸张与屏幕拍摄配置、多帧融合、反光无效区标记、轻度恢复假设，以及证据回映射到原始照片。`mode="auto"` 会识别明显的屏幕摩尔纹，否则回退到 digital。纸张照片在取得独立设备留出校准前，请明确选择 `camera-paper`。目前置信度采用保守设计，尚未使用独立设备数据完成校准。

| 直方图匹配的马赛克 | 连续证据 | 可审计叠加图 |
| --- | --- | --- |
| ![生成的多边形马赛克](examples/digital/demo_input.png) | ![恢复的连续证据](examples/digital/demo_result/evidence_01.png) | ![恢复叠加图](examples/digital/demo_result/overlay_01.png) |

这个由项目自行生成的案例，将 `820` 隐藏在某一种多边形颜色的局部密度中，并让随机负例具有相同数量的特殊色彩组件。证据图会把空间信号显示出来；此确定性案例返回 `ok`，而存在多个同样合理掩码的案例会保持 `uncertain`。这是 evidence-first 的预期行为，不是 OCR 承诺。

## 系统架构

![ChromaRecover evidence-first 系统架构](docs/assets/chromarecover-architecture.png)

恢复流程与持续红队／蓝队验证回路整合在同一张图中：输入安全、拍摄模型、多族群证据假设、保守决策及可审计产物。图中的对象都可以在[可编辑 PowerPoint 源文件](docs/assets/ChromaRecover-research-architecture.pptx)中单独调整。

## 为什么使用 ChromaRecover？

有些结构主要由相对色彩而非强烈亮度边缘编码，因此人眼、OCR 与通用视觉模型都可能难以可靠判断。ChromaRecover 会先展示视觉证据，再尝试赋予标签：

- 先恢复结构，再进行 OCR 或数字解读；
- 在本地 CPU 运行，不上传、不需要账号，也没有遥测；
- 证据不足时返回不确定，不凭空补画或猜答案。

**如果证据不足，ChromaRecover 会明确说明。**

## 安装

在第一个软件包注册表版本发布前，请下载或 clone 本仓库，进入根目录后安装已跟踪的源代码：

```
python -m pip install .
```

如需运行测试与贡献者工具：

```
python -m pip install ".[dev]"
```

Windows Python 3.10 在仓库路径包含非 ASCII 字符时，可能触发上游 editable-install `.pth` 编码问题。上面的非 editable 安装方式更可靠；开发环境建议使用 Python 3.11 以上。

## Python API

```python
from chromarecover import recover

result = recover("image.png", mode="digital", top_k=3)
result.save("result")
if result.best is None:
    print(result.status, "没有可辩护的候选")
else:
    print(result.status, result.best.decision_confidence)
```

`result.best` 的类型特意设为 `Candidate | None`：空白、接近单色或信息量过低的图像，会正常返回没有候选的 `uncertain`，而不是捏造结果。

数字解读是可选功能，而且只会在视觉证据存在后运行：

```python
result = recover("plate.png", semantics="auto")
print(result.semantics)  # 假设、置信度、接受理由、候选间一致性
```

它结合多字体形状距离、孔洞／拓扑与投影式字形分割。只有通过接受条件的读值会出现在公开 `hypothesis` 字段；未接受的模板标签只保留为候选内部的字符级诊断信息。数字假设不会修改掩码，也不会把缺失像素变成证据。

当三个候选掩码、至少两个独立视觉证据族群高度一致时，即使恢复状态仍是 `uncertain`，也可能接受可读的数字序列。这只验证形状，不能提升视觉状态、重排候选或宣称恢复缺失像素。

### 相机照片

相机模式默认会自动寻找可信四边形。如果文档边界不明确，请传入裁剪范围或原始图像坐标中的四个角点：

```python
result = recover("photo.jpg", mode="camera", roi=(120, 80, 1880, 1320))
result = recover(
    "photo.jpg",
    mode="camera",
    document_corners=[[130, 92], [1872, 65], [1904, 1318], [105, 1340]],
)
```

角点必须是四个互异、有限且能形成可用凸四边形的点。系统允许 10% 的小幅坐标边界，用于亚像素或略微超框估计；非有限值、重复点、退化或距离过远的点，会在透视变换前抛出 `InvalidInputError`。

褶皱与光照影响明显时使用 `camera-paper`；刷新条纹与摩尔纹明显时使用 `camera-screen`。通用 `camera` 会根据测量证据选择配置。自动展平只使用实际观测到的页面边界，几何不可信时会弃权。

如果反光或遮挡在多张照片之间移动，可以使用 burst fusion 从其他帧中真正被观测到的像素获取证据：

```python
from chromarecover import recover_burst

result = recover_burst(
    ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"],
    mode="camera-screen",
    return_debug=True,
)
```

相机校正只估计干扰场，不宣称能从单张照片恢复被截断、遮挡或严重失焦的信息。请检查 `result.quality.warnings`，并在系统建议时重新拍摄。

`ok` 始终使用 `best.decision_confidence` 所代表的同一个公开置信度；证据族群门控不能用另一个分数跨过状态阈值。相机与 auto 状态在取得具有独立授权、按来源组留出的拍摄数据前，仍然未完成校准。

`ok` 表示系统找到符合决策契约的一致色彩／空间结构；它**不表示**普通照片一定包含刻意隐藏的信息，也不保证语义解读正确。

## CLI

```
chromarecover image.png --out result --top-k 3 --semantics auto
chromarecover photo.jpg --mode camera --roi 120,80,1880,1320 --out result
chromarecover --burst one.jpg two.jpg three.jpg --mode camera-screen --debug --out result
```

输出目录会区分三种含义：`mask_01.png` 是选中的色彩支撑类别，`structure_01.png` 是其空间包络，`evidence_01.png` 是连续局部支撑；`overlay_01.png` 同时显示前两者。schema `0.5` 的 `result.json` 记录变换、质量与来源。全分辨率的支撑及结构掩码保持在原始图像坐标中。

为避免手机图像造成内存峰值，连续证据与叠加图限制在最长边 2048 像素的展示空间，并在 `artifact_spaces` 记录尺寸与缩放比例。`--debug` 还会保存校正后观测、无效区域、primitive 标签与恢复假设。

置信度也分层处理：`structure_confidence` 表示拍摄惩罚前的候选；`capture_confidence` 纳入输入质量；`decision_confidence` 再纳入歧义与最终状态。Alpha 期间，`overall_confidence` 只是 `decision_confidence` 的兼容别名。

## 设计保证

- Mask-first：OCR 或文件名不能决定恢复像素。
- Semantics-last：数字识别只能验证可读形状，不能生成色彩支撑或推翻弱证据弃权。
- 多假设：前景颜色、极性与面积不被写死。
- 分层相机路径：几何、光度干扰估计、原始／校正色彩证据、原生多边形 primitives、局部分布图、图分组与逆向映射都可审计。
- 证据保存：burst fusion 使用配准后的实际观测；单帧去模糊仅作为受限假设分支展示，不会被重新标记为 ground truth。
- 保守失败处理：可读但无法定论的输入不是异常。
- 本地优先：不上传、不需要账号、没有遥测。
- 可复现：每个结果都保存确定性 seed 与配置指纹。

详见[算法说明](docs/algorithm.md)与[项目契约](PROJECT_SPEC.md)。

## 限制与非目标

本工具不是医疗诊断工具，也不宣称能从单张相机照片重建物理上真实的颜色或原始像素。严重失焦、截断、反光、遮挡及采样不足都可能永久破坏信息；深度且不可展开的褶皱需要经过学习或校准的表面模型，不在这个 CPU Alpha 范围内。

默认单张图像安全上限为 50 megapixels，可覆盖 8064×6048 的手机原图。内存有限的系统仍应先裁剪目标区域，或使用高质量解码器只降采样一次。

文件系统输入在解码前限制为 100 MiB。Burst 会先检查所有输入，限制为 2–12 帧及合计 80 megapixels。嵌入式 ICC profile 在 LittleCMS 解析前限制为 4 MiB；过大的 profile 会被忽略并记录。本地函数库不假装提供安全的 process-internal 解码超时；网络服务必须按照[安全模型](docs/security-model.md)增加进程隔离、wall-time 与内存限制。

Alpha 的 inline typing 采用 best-effort。软件包包含 `py.typed`，但 OpenCV／NumPy 的完整静态类型门控属于 v0.4 工作；当前标记不表示所有内部数组操作都已经通过 mypy。

## Roadmap 与贡献

当前优先事项是具有独立授权的相机拍摄、更多 primitive families、更低内存的 burst processing，以及测量驱动的性能改进。已完成范围与发布门槛记录在[公开 roadmap](docs/roadmap.md)；新功能原则上必须同时附带 regression case 与 hard negative。

适合首次贡献的小型工作整理在[贡献者选题](docs/contributor-ideas.md)，其中部分已发布为 [`good first issue`](https://github.com/niansia/ChromaRecover/labels/good%20first%20issue) 与 [`help wanted`](https://github.com/niansia/ChromaRecover/labels/help%20wanted)。

## 开发检查

```
pytest
ruff check .
python benchmark/evaluate_synthetic.py --assert-thresholds
python benchmark/evaluate_nuisance.py --assert-thresholds
python benchmark/evaluate_mosaic.py --assert-thresholds
```

CI 测试 Python 3.10–3.13，并覆盖 NumPy 1.26／OpenCV 4.10 与 NumPy 2.x／OpenCV 4.14。OpenCV 5 与未来 NumPy major release 不在当前支持范围，必须先通过专门的完整测试与 benchmark 才会放宽版本上限。

当前 golden inventory 包含 20 个确定性生成案例；不宣称它们是真实相机照片。外部相机几何使用具有授权的 SmartDoc dataset 单独评估，而且不会提交第三方帧，详见[外部数据集说明](docs/external-datasets.md)。私有 blueprint 衍生 fixture 不存在于公开 tree 或工具中，也不是 CI 的必要条件。

ChromaRecover 采用 Apache-2.0 许可证。社区治理将在 Public Alpha 期间持续完善，并向 1.0 推进。
