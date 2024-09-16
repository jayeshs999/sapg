import os
# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
# This file opener *must* be used in a "with" block.
class AtomicOpen:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, lock, path, flags, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        
        if 'r' in flags:
            lock.acquire_read_lock()
        else:
            lock.acquire_write_lock()
            
        self.file = open(path, flags, *args, **kwargs)
        self.flags = flags
        self.lock = lock

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):        
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        self.file.close()
        if 'r' in self.flags:
            self.lock.release_read_lock()
        else:
            self.lock.release_write_lock()
        
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if (exc_type != None): return False
        else:                  return True