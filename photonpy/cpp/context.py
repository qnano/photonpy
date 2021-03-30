# -*- coding: utf-8 -*-

import ctypes
from .lib import SMLM

class Context:
    """
    Manage C++-side objects. 
    All native objects linked to the context will be destroyed when the context is destroyed.
    """
    def __init__(self, smlm:SMLM=None, debugMode=False):
        self.libOwner = smlm is None
        if self.libOwner:
            smlm = SMLM(debugMode)
        
        self.smlm = smlm
        lib = smlm.lib
        self.lib = lib
        
        self._Context_Create = lib.Context_Create
        self._Context_Create.argtypes=[]
        self._Context_Create.restype = ctypes.c_void_p
        
        self._Context_Destroy = lib.Context_Destroy
        self._Context_Destroy.argtypes = [ctypes.c_void_p]
        
        self.inst = self._Context_Create()

    def Create(self):
        return Context(self.smlm)

    def Destroy(self):
        if self.inst:
            self._Context_Destroy(self.inst)
            self.inst=None
            
            if self.libOwner:
                self.smlm.Close()
                self.smlm = None
                self.lib = None
            
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()


        