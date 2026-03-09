# Record a WISDM activity-recognition demo with VHS: https://github.com/charmbracelet/vhs
# Run from repo root:
#   vhs demo/demo_wisdm.tape
#
# Outputs a demo GIF at: demo/assets/demo_wisdm.gif

Output demo/assets/demo_wisdm.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 700
Set TypingSpeed 60ms

Hide
Type "clear"
Enter
Show

Type "bash demo/run_demo_wisdm.sh"
Enter

# WISDM subset is larger than synthetic; allow extra time
Sleep 30s

Type "ls -lh outputs/demo_wisdm"
Enter

Sleep 3s
