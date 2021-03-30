# -*- coding: utf-8 -*-

# A set of array util functions

def peek_first(src):
    """
    Retrieve the first iterator element while 
    returning an iterator that will still generate the full sequence
    """
    first = next(src)
    
    def gen():
        yield first
        for d in src:
            yield d
    
    return first, gen()
