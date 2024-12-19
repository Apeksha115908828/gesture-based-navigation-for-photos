import pyautogui
import subprocess

def activate_window():
    apple_script_file = "script.applescript"
    subprocess.run(["osascript", apple_script_file])

# Keycodes: 123 = Left Arrow, 124 = Right Arrow, 47 = .(period), 24 = +, 27 = -
def next_photo():
    activate_window()
    pyautogui.press('right') 

def previous_photo():
    activate_window()
    pyautogui.press('left') 

def mark_favorite():
    activate_window()
    pyautogui.press('.') 

def zoom_in():
    activate_window()
    pyautogui.hotkey('command', '+')

def zoom_out():
    activate_window()
    pyautogui.hotkey('command', '-')