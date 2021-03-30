# -*- coding: utf-8 -*-
import os
import fnmatch

def scandir(path, pat, cb):
    for root, dirs, files in os.walk(path):
        head, tail = os.path.split(root)
        for file in files:
            if fnmatch.fnmatch(file, pat):
                fn = os.path.join(root, file)
                cb(fn)
                
            