from typing import List

def __init__(self, tau: float = 0.5, rex: List[str] = None, keep_para: bool = True):
    self.tau = tau
    self.rex = rex or []
    self.keep_para = keep_para
    self.df_log = None
    self.logname = None
    self.logformat = None  # Add this line
    self.savePath = None   # Add this line