from typing import List
from torch.nn import Module
from pathlib import Path

class Logger():
    def __init__(self, models : List[Module], path : Path, attribs : dict) -> None:
        
        self.f = open(path, "w")
        
        # header
        self.log("---")
        for tag, values in attribs.items():
            if type(values) is list:
                self.log(f"{tag}: {', '.join(values)}")
            else:
                self.log(f"{tag}: {str(values)}")
        self.log("---\n")
        
        # model details
        self.log("# Models")
        for model in models:
            self.log(model.__str__())
        self.log("") 
        self.log("# Logs")
        self.f.flush()
        
    def log(self, text : str) -> None:
        self.f.write(text + "\n")
        
    def close(self) -> None:
        self.f.flush()
        self.f.close()