from pynput import mouse
import threading
import sys

class EmergencySystem:
    def __init__(self):
        self.click_count = 0
        self.timer_thread = None
        self.timer_expired = False
        self.listener = None  # Store the listener instance

    def run(self):
        print()
        print("You have triggered the EMERGENCY SYSTEM.")
        print("Waiting for user input...")
        print()

        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        self.listener.join()

    def on_click(self, x, y, button, pressed):
        if pressed:
            if self.timer_expired:
                return  # Ignore clicks after the timer has expired

            self.click_count += 1
            print(f"Click detected: {self.click_count}")

            if self.click_count == 1:
                print("10 Second timer has started,Tap once to cancel the emergency system.")
                self.start_timer()

            elif self.click_count == 4:
                self.cancel_timer()
                print("Connecting the emergency services...")
                sys.exit()  # Exit the program

    def start_timer(self):
        self.timer_thread = threading.Timer(10, self.timeout)
        self.timer_thread.start()

    def cancel_timer(self):
        if self.timer_thread:
            self.timer_thread.cancel()

    def timeout(self):
        print("Timeout! No thrice click detected within 10 seconds.")
        self.timer_expired = True
        self.listener.stop()  # Stop the listener

if __name__ == "__main__":
    emergency_system = EmergencySystem()
    emergency_system.run()
