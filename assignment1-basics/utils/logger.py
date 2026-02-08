import logging

class Logger:
    def __init__(
            self,
            log_name:str,
            log_file:str
            ):
        self.log_name = log_name
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        self.logger = \
                logging.getLogger(self.log_name)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return
        formatter = logging.Formatter(
                '%(asctime)s '
                '%(levelname)s:'
                '%(message)s',
                datefmt='%H:%M:%S'
        )
        file_handler = logging.FileHandler(\
                self.log_file, 
                encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("setup_logging: log init sucess")

