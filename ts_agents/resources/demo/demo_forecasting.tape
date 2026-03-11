# Record a forecasting comparison demo with VHS: https://github.com/charmbracelet/vhs
# Run from repo root:
#   vhs demo/demo_forecasting.tape
#
# Outputs a demo GIF at: demo/assets/demo_forecasting.gif

Output demo/assets/demo_forecasting.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 700
Set TypingSpeed 60ms

Hide
Type "clear"
Enter
Show

Type "uv run ts-agents demo forecasting --no-llm --output-dir outputs/demo_forecasting --report-path outputs/demo_forecasting/forecasting_report.md"
Enter

# Forecasting demo is quick; 15s should be plenty
Sleep 15s

Type "ls -lh outputs/demo_forecasting"
Enter

Sleep 3s
