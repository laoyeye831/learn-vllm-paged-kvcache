# 录制 README 里的演示 GIF：vhs docs/demo.tape
# 依赖：vhs、ttyd、ffmpeg
#   winget install charmbracelet.vhs tsl0922.ttyd Gyan.FFmpeg.Essentials
#
# 节奏由两边共同决定：
#   - 这里（VHS）控制录制/截帧的节奏与画布大小
#   - demo 的 --paced 控制输出节奏（每个节拍插入 sleep），ctest 里默认关闭
# 画布按「整段输出不滚动」推算：演示约 76 行、最宽一行 88 列
Output docs/demo.gif

Set Shell "powershell"
# macOS / Linux 上把上一行换成：Set Shell "bash"

Set FontFamily "Cascadia Mono, Consolas, monospace"
Set FontSize 13
Set Width 800
Set Height 1240
Set Padding 20
Set Theme "Dracula"
Set Framerate 15
Set TypingSpeed 40ms

Type "./build/kvcache_demo --paced"
Enter
Sleep 20s
