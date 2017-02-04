import time
import logging


class MetaBlockLogger:
    INDENT_FORMAT = '' #'%(tab)s'

    def __init__(self, logger=None):
        self.logger = logger
        self.indent_filter = None#IndentFilter()
        #self.logger.addFilter(self.indent_filter)

    def __call__(self, name):
        return BlockLogger(name, self.logger, indent_filter=self.indent_filter)


class IndentFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.

    Rather than use actual contextual information, we just use random
    data in this demo.
    """

    def __init__(self):
        self.tab = 0
        self.dash = False
        super().__init__()

    def indent(self):
        self.tab += 1

    def unindent(self):
        self.tab -= 1

    def filter(self, record):
        if self.tab > 0:
            record.tab = ' |  ' * (self.tab-1) + (' |--' if self.dash else ' |  ')
        else:
            record.tab = ''
        return True


class BlockLogger:
    def __init__(self, name, logger=None, indent_filter=None):
        self.logger = logger
        self.name = name
        self.indent_filter = indent_filter

    def __enter__(self):
        self.start = time.clock()

        if self.indent_filter:
            self.indent_filter.dash = True
            self.logger.info("BEGIN: {}".format(self.name))
            self.indent_filter.dash = False
            self.indent_filter.indent()
        else:
            self.logger.info("BEGIN: {}".format(self.name))
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        interval = self.end - self.start
        if self.indent_filter:
            self.indent_filter.unindent()
        self.logger.info("END  : {} | {:.2f}s".format(self.name, interval))