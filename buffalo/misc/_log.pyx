# cython: language_level=3, nonecheck=False
# distutils: language=c++


cdef extern from "buffalo/misc/log.hpp":
    cdef cppclass _BuffaloLogger "BuffaloLogger":
        void set_log_level(int)
        int get_log_level()


cdef class PyBuffaloLog:
    cdef _BuffaloLogger* obj

    def __cinit__(self):
        self.obj = new _BuffaloLogger()

    def __dealloc__(self):
        del self.obj

    def set_log_level(self, lvl):
        self.obj.set_log_level(lvl)

    def get_log_level(self):
        return self.obj.get_log_level()
