# blade-build 反馈 (dogfooding)

> 用 nuthatch 这个真实工程压 blade-build,踩到的摩擦点随手记这里。
> 每条:现象 → 期望 → 是否可修 → 是否已提 issue/PR。

## 2026-07-11

### 1. blade 不在 PyPI,CI 安装方式不显然
- **现象**:CI 里 `pip install blade-build` 失败("No matching distribution")。blade 实际是"仓库放个 `blade` 启动脚本 exec `thirdparty/blade/blade`"(见 flare)。
- **期望**:官方给一段**标准 CI 安装片段**(clone + PATH,或一个 setup-blade 的 GitHub Action),让新项目照抄即可。
- **可修**:文档/Action 层面即可,低成本。
- **状态**:待提(blade-build docs / 或做个 `chen3feng/setup-blade` action)。

### 3. blade 输出目录 build_release/ 未被默认忽略,易误入 git
- **现象**:`blade build/test` 生成 `build_release/`(ninja 文件、vcpkg 编译产物、日志)。新项目若 `git add -A` 会把它整个提交进去(本项目一度污染了 main,44 个文件)。
- **期望**:blade `init`(或文档)给出一份**推荐 `.gitignore`**(含 `build_release/` `build_debug/` `blade-bin`),或 blade 在首次构建时自动写入 `.gitignore`。
- **可修**:低成本(文档 / `blade init` 模板)。
- **状态**:本项目已手动加 `.gitignore`;待给 blade 提建议。

### 2. vcpkg 集成需要手动 pin baseline(runner 自带 vcpkg 太旧)
- **现象**:runner 预装的 vcpkg 版本库落后于 `vcpkg_config` 的 baseline,不 fetch+checkout 会解析不到 pin 的版本。flare 靠一个额外 CI 步骤 fetch+checkout baseline commit。
- **期望**:blade 的 vcpkg 集成能否**自动确保 vcpkg 在 baseline**(或文档明确要求这一步)?
- **可修**:文档明确 + 可能的自动化。
- **状态**:待观察(先照 flare 手动 pin)。

### 4. vcpkg 多库端口:umbrella 的 transitive link 未被跟随
- **现象**:`vcpkg#ggml:ggml` 只链空壳 umbrella `libggml.a`,而真符号在 `libggml-base.a`(核心算子)和 `libggml-cpu.a`(compute)里 → 链接全未定义。CMake 侧 `ggml::ggml` 会 transitively 带上 base/cpu,但 blade 的 vcpkg 集成没跟随这个 INTERFACE_LINK_LIBRARIES。
- **期望**:blade 的 vcpkg 集成能否跟随 CMake target 的 transitive link interface?否则每个多库端口都要手动列全子库。
- **规避**:`thirdparty/ggml` 显式列 `vcpkg#ggml:{ggml-cpu,ggml-base,ggml}`。
- **状态**:已规避;值得给 blade 提(vcpkg 多库端口的通用问题)。

### 5. cc_check_undefined 是 warning,CI 侥幸绿(建议可配成硬失败)
- **现象**:上面 #4 的未定义符号,blade 的 `cc_check_undefined` 在**本地和 CI 都报了警**——这是 blade 的**亮点**(静态抓出漏声明的 deps)。但它只是 warning:CI 的宽松链接器把符号蒙混解析过去、测试还跑绿了,而本地严格 `ld64` 才真失败。结果 CI "侥幸绿" = 不可信。
- **期望**:提供一个开关把 `cc_check_undefined` 失败**升级为构建错误**(至少 CI 里),让 CI 与严格链接器一致。目前只看到 `allow_undefined` 白名单方向,没看到"变严格"的方向。
- **状态**:待确认 blade 的配置开关(cc_config?),配上后加进 CI。
