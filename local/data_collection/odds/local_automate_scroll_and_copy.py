import pyautogui
import time


def scroll_and_copy():
    pyautogui.scroll(-1000)
    time.sleep(0.5)

    # Press Ctrl+C to copy
    pyautogui.hotkey("ctrl", "c")
    time.sleep(1)


if __name__ == "__main__":
    time.sleep(10)
    while True:
        scroll_and_copy()
