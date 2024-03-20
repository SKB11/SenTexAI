from flask import Flask, render_template
from pynput import mouse
import threading

app = Flask(__name__)

class EmergencySystem:
    def __init__(self):
        self.click_count = 0
        self.timer_thread = None
        self.listener = None
        self.wait_time = 5  # Wait time for clicks after the first click

    def run(self):
        print("You have triggered the EMERGENCY SYSTEM.")
        print("Waiting for user input...")

        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        try:
            self.listener.join()
        except EmergencyExit as e:
            print(e)  # Print the user-friendly message

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.click_count += 1
            print(f"Click detected: {self.click_count}")

            if self.click_count == 1:
                print(f"{self.wait_time} Second timer started. Click two more times for help, failing which will be a false alarm.")
                self.start_timer()

            elif self.click_count == 3:
                self.cancel_timer()
                print("Contacting emergency services. Help on the way.")
                raise EmergencyExit("Emergency Signal Received")

    def start_timer(self):
        self.timer_thread = threading.Timer(self.wait_time, self.timeout)
        self.timer_thread.start()

    def cancel_timer(self):
        if self.timer_thread:
            self.timer_thread.cancel()

    def timeout(self):
        print("False alarm! No further clicks detected.")
        self.listener.stop()  # Stop the listener

class EmergencyExit(Exception):
    pass

emergency_system = EmergencySystem()

@app.route('/')
def index():
    return render_template('click_index.html')

@app.route('/trigger_emergency')
def trigger_emergency():
    try:
        emergency_system.run()
    except EmergencyExit as e:
        return "Received emergency signal. Help on the way."

if __name__ == "__main__":
    app.run(debug=True)

