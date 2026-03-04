# Record a quick terminal demo with VHS: https://github.com/charmbracelet/vhs
# Run from repo root:
#   vhs demo/demo.tape
#
# Outputs a demo GIF at: demo/assets/demo.gif

Output demo/assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 700
Set TypingSpeed 60ms

Hide
Type "clear"
Enter
Show

Type "bash demo/run_demo.sh"
Enter

# Demo takes ~10s; wait long enough for it to finish plus margin
Sleep 15s

Type "ls -lh outputs/demo"
Enter

Sleep 3s
