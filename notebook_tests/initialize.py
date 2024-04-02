import os

initialized = {}

def initialize():
    global initialized
    
    if 'x' not in initialized:
        os.chdir('../')
    initialized['x'] = True

initialize()
