# blade-build 反馈 (dogfooding)

> 用 nuthatch 这个真实工程压 blade-build,踩到的摩擦点随手记这里。
> 每条:现象 → 期望 → 是否可修 → 是否已提 issue/PR。

## 2026-07-11

### 1. blade 不在 PyPI,CI 安装方式不显然
- **现象**:CI 里 `pip install blade-build` 失败("No matching distribution")。blade 实际是"仓库放个 `blade` 启动脚本 exec `thirdparty/blade/blade`"(见 flare)。
- **期望**:官方给一段**标准 CI 安装片段**(clone + PATH,或一个 setup-blade 的 GitHub Action),让新项目照抄即可。
- **可修**:文档/Action 层面即可,低成本。
- **状态**:待提(blade-build docs / 或做个 `chen3feng/setup-blade` action)。

### 2. vcpkg 集成需要手动 pin baseline(runner 自带 vcpkg 太旧)
- **现象**:runner 预装的 vcpkg 版本库落后于 `vcpkg_config` 的 baseline,不 fetch+checkout 会解析不到 pin 的版本。flare 靠一个额外 CI 步骤 fetch+checkout baseline commit。
- **期望**:blade 的 vcpkg 集成能否**自动确保 vcpkg 在 baseline**(或文档明确要求这一步)?
- **可修**:文档明确 + 可能的自动化。
- **状态**:待观察(先照 flare 手动 pin)。
