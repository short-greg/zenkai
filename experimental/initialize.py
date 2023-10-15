import os

initialized = False

def _initialize():

    global initialized
    if not initialized:
        initialized = True
        os.chdir('../')

_initialize()
