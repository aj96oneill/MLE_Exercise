import logging
import os
import sys

try:
    os.mkdir('./logs')
except OSError:  # Exists
    pass

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s [%(process)d - %(threadName)s] %(message)s',
      filename='./logs/server.log',
      filemode='a')

log = logging.getLogger()

class LoggerWriter():
    def __init__(self, level:str):
        self.level = level

    def write(self, message):
        message = message.replace("\n", "").strip()
        if message != "":
            if self.level == "info": log.info(message)
            elif self.level == "warning": log.warning(message) 
            elif self.level == "error": log.error(message) 
            elif self.level == "critical": log.critical(message)
            else: log.debug(message)
            
    def flush(self):
        pass

sys.stdout = LoggerWriter("debug")
sys.stderr = LoggerWriter("error")

class Logger():
    def debug(self, message:str) -> None:
        log.debug(message)
    def info(self, message:str) -> None:
        log.info(message)
    def warning(self, message:str) -> None:
        log.warning(message)
    def error(self, message:str) -> None:
        log.error(message)
    def critical(self, message:str) -> None:
        log.critical(message)
