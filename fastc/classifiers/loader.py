import sys
import threading
import time


class Loader:
    def __init__(self, description=''):
        self.description = description
        self.is_running = False
        self.animation_thread = None
        self.chars = ['|', '/', '-', '\\']

    def animate(self):
        while self.is_running:
            for char in self.chars:
                sys.stdout.write(f'\r{self.description} {char}')
                sys.stdout.flush()
                time.sleep(0.1)

    def start(self):
        self.is_running = True
        self.animation_thread = threading.Thread(target=self.animate)
        self.animation_thread.start()

    def stop(self):
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.description) + 2) + '\r')
        sys.stdout.flush()
