# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: pyimg.pyx
#
#                       Author: kewuaa
#                      Created: 2022-05-22 19:17:17
#                last modified: 2022-06-07 23:23:31
#******************************************************************#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.math cimport pow as c_pow
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos
from libc.math cimport exp as c_exp
from libc.math cimport log2 as c_log2
from libc.math cimport lround
from libc.math cimport hypot
from libc.math cimport fdim
from libc.math cimport pi
from cpython.mem cimport PyMem_Free
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Realloc
from cython.parallel cimport prange
cimport numpy as cnp
import numpy as np


cdef extern from '<complex.h>' nogil:
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)


cdef class IntMemory:
    cdef:
        Py_ssize_t _size
        int *_data
        int[:, ::1] view2d

    def __cinit__(self, Py_ssize_t size):
        self._data = <int *> PyMem_Malloc(size * sizeof(int))
        if self._data is NULL:
            raise MemoryError()

    def __init__(self, Py_ssize_t size):
        self._size = size

    def __dealloc__(self):
        PyMem_Free(self._data)


    def resize(self, Py_ssize_t size):
        self._data = <int *> PyMem_Realloc(self._data, size * sizeof(int))
        if self._data is NULL:
            raise MemoryError()

    cdef int[:, ::1] init_2dview(self, Py_ssize_t rows, Py_ssize_t cols):
        self.view2d = <int[: rows, : cols]> self._data
        return self.view2d


cdef class DoubleMemory:
    cdef:
        Py_ssize_t _size
        double *_data
        double[:, ::1] view2d

    def __cinit__(self, Py_ssize_t size):
        self._data = <double *> PyMem_Malloc(size * sizeof(double))
        if self._data is NULL:
            raise MemoryError()

    def __init__(self, Py_ssize_t size):
        self._size = size

    def __dealloc__(self):
        PyMem_Free(self._data)


    def resize(self, Py_ssize_t size):
        self._data = <double *> PyMem_Realloc(self._data, size * sizeof(double))
        if self._data is NULL:
            raise MemoryError()

    cdef double[:, ::1] init_2dview(self, Py_ssize_t rows, Py_ssize_t cols):
        self.view2d = <double[: rows, : cols]> self._data
        return self.view2d


cdef class DoubleComplexMemory:
    cdef:
        Py_ssize_t _size
        double complex *_data
        double complex[:, ::1] view2d

    def __cinit__(self, Py_ssize_t size):
        self._data = <double complex *> PyMem_Malloc(size * sizeof(double complex))
        if self._data is NULL:
            raise MemoryError()

    def __init__(self, Py_ssize_t size):
        self._size = size

    def __dealloc__(self):
        PyMem_Free(self._data)

    def resize(self, Py_ssize_t size):
        self._data = <double complex *> PyMem_Realloc(self._data, size * sizeof(double))
        if self._data is NULL:
            raise MemoryError()

    cdef double complex[:, ::1] init_2dview(self, Py_ssize_t rows, Py_ssize_t cols):
        self.view2d = <double complex[: rows, : cols]> self._data
        return self.view2d


cpdef cnp.ndarray[double, ndim=2] BGR2GRAY(double[:, :, ::1] img):
    """灰度转换.

    Args:
    img:
        三通道图像

    Returns:
        灰度图像
    """

    cdef:
        unsigned i, j, k
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        Py_ssize_t channels = img.shape[2]
        double transform_array[3]
        double[:, ::1] result_view
        cnp.ndarray[double, ndim=2] result = np.ones([rows, cols])
    transform_array = [0.0722, 0.7152, 0.2126]
    result_view = result
    with nogil:
        for i in prange(rows):
            for j in range(cols):
                for k in range(channels):
                    result_view[i, j] += img[i, j, k] * transform_array[k]
    return result


cdef fused double_or_double_complex_arr1d:
    double[:]
    double complex[:]


cdef void *dft(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """离散傅里叶变换.

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行变换的数组

    Returns:
        None
    """

    cdef:
        Py_ssize_t N = array.shape[0]
        Py_ssize_t i, j
        double complex J = -2j * pi / N
        double complex w
    result_view[:] = 0 + 0j
    for i in prange(N):
        w = J * i
        for j in range(N):
            result_view[i] += array[j] * cexp(j * w)


cdef void *idft(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """离散傅里叶变换逆过程.

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行逆变换的数组

    Returns:
        None
    """

    cdef:
        Py_ssize_t N = array.shape[0]
        Py_ssize_t i, j
        double complex J = 2j * pi / N
        double complex w
    result_view[:] = 0 + 0j
    for i in prange(N):
        w = J * i
        for j in range(N):
            result_view[i] += array[j] * cexp(j * w)
        result_view[i] /= N


cdef double complex[:] fft_by_recursion(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶变换(递归方式).

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        Py_ssize_t i
        Py_ssize_t half
        Py_ssize_t N = array.shape[0]
        double complex w, x, e, o
        double complex[:] array_e, array_o
    # 无法拆成奇偶对形式, 停止递归, 返回结果
    if N % 2 != 0:
        if N <= 1:
            result_view[0] = <double complex> array[0]
            return result_view
        else:
            dft(result_view, array)
            return result_view
    half = N / 2
    w = -2j * pi / N
    array_e = fft_by_recursion(result_view[: half], array[::2])
    array_o = fft_by_recursion(result_view[half:], array[1::2])
    # x1 + x2 = 0, x1² = x2²
    # P(x1) = Pe(x1²) + po(x1²)
    # p(x2) = pe(x2²) - po(x2²)
    # 根据公式依次带回
    for i in range(half):
        e = array_e[i]
        o = array_o[i] * cexp(w * i)
        result_view[i] = e + o
        result_view[i + half] = e - o
    return result_view


cdef void *fft_(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶变换(非递归方式).

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        Py_ssize_t N = array.shape[0]
        Py_ssize_t i, j, step, half_step
        int pos
        int bit = <int> c_log2(<double> N) - 1
        double complex e, o, w
    step = N >> bit
    # 位逆序置换
    for i in range(N):
        pos = 0
        for j in range(bit + 1):
            pos |= (i >> j & 1) << (bit - j)
        result_view[i] = array[pos]
    for _ in range(bit + 1):
        half_step = step / 2
        w = -2j * pi / step
        i = 0
        while i < N:
            for j in range(half_step):
                e = result_view[i + j]
                o = result_view[i + j + half_step] * cexp(w * j)
                result_view[i + j] = e + o
                result_view[i + j + half_step] = e - o
            i += step
        step *= 2


cdef double complex[:] ifft_by_recursion(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶逆变换(递归方式).

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行逆变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        Py_ssize_t i
        Py_ssize_t half
        Py_ssize_t N = array.shape[0]
        double complex w, x, e, o
        double complex[:] array_e, array_o
    if N % 2 != 0:
        if N <= 1:
            result_view[0] = <double complex> array[0]
            return result_view
        else:
            idft(result_view, array)
            return result_view
    half = N / 2
    w = 2j * pi / N
    array_e = ifft_by_recursion(result_view[: half], array[::2])
    array_o = ifft_by_recursion(result_view[half:], array[1::2])
    for i in range(half):
        e = array_e[i] / 2
        o = array_o[i] * cexp(w * i) / 2
        result_view[i] = e + o
        result_view[i + half] = e - o
    return result_view


cdef void *ifft_(double complex[:] result_view, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶逆变换(非递归方式).

    Args:
    result_view:
        存放结果的数组视图
    array:
        需要进行变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        Py_ssize_t N = array.shape[0]
        Py_ssize_t i, j, step, half_step
        int pos
        int bit = <int> c_log2(<double> N) - 1
        double complex e, o, w
    step = N >> bit
    for i in range(N):
        pos = 0
        for j in range(bit + 1):
            pos |= (i >> j & 1) << (bit - j)
        result_view[i] = array[pos]
    for _ in range(bit + 1):
        half_step = step / 2
        w = 2j * pi / step
        i = 0
        while i < N:
            for j in range(half_step):
                e = result_view[i + j] / 2
                o = result_view[i + j + half_step] * cexp(w * j) / 2
                result_view[i + j] = e + o
                result_view[i + j + half_step] = e - o
            i += step
        step *= 2


cpdef cnp.ndarray[double complex, ndim=1] fft(double[:] array):
    """一维离散傅里叶变换.

    Args:
    array:
        需要进行变换的数组

    Returns:
        返回变换结果
    """

    cdef:
        int N = array.shape[0]
        double complex[::1]result_view 
        cnp.ndarray[double complex, ndim=1] result = np.empty(N, dtype=complex)
    result_view = result
    with nogil:
        if c_log2(<double> N) % 1 == 0:
            fft_[double[:]](result_view, array)
        else:
            fft_by_recursion[double[:]](result_view, array)
    return result


cpdef cnp.ndarray[double complex, ndim=1] ifft(double complex[:] array):
    """一维离散傅里叶逆变换.

    Args:
    array:
        需要进行逆变换的数组

    Returns:
        返回逆变换结果
    """

    cdef:
        int N = array.shape[0]
        double complex[::1]result_view 
        cnp.ndarray[double complex, ndim=1] result = np.empty(N, dtype=complex)
    result_view = result
    with nogil:
        if c_log2(<double> N) % 1 == 0:
            ifft_[complex[:]](result_view, array)
        else:
            ifft_by_recursion[complex[:]](result_view, array)
    return result


cdef DoubleComplexMemory fft2_(double[:, :] array):
    """二维离散傅里叶变换.

    Args:
    array:
        需要进行变换的数组

    Returns:
        返回变换结果
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = array.shape[0]
        Py_ssize_t cols = array.shape[1]
        Py_ssize_t size = rows * cols
        DoubleComplexMemory temp = DoubleComplexMemory(size)
        DoubleComplexMemory result = DoubleComplexMemory(size)
        double complex[:, ::1] temp_view = temp.init_2dview(rows, cols)
        double complex[:, ::1] result_view = result.init_2dview(rows, cols)
    with nogil:
        if c_log2(cols) % 1 == 0:
            for i in prange(rows):
                fft_[double[:]](temp_view[i, ...], array[i, ...])
        else:
            for i in prange(rows):
                fft_by_recursion[double[:]](temp_view[i, ...], array[i, ...])
        if c_log2(rows) % 1 == 0:
            for i in prange(cols):
                fft_[complex[:]](result_view[..., i], temp_view[..., i])
        else:
            for i in prange(cols):
                fft_by_recursion[complex[:]](result_view[..., i], temp_view[..., i])
    return result


cpdef cnp.ndarray[double complex, ndim=2] fft2(double[:, :] array):
    cdef:
        DoubleComplexMemory result = fft2_(array)
    return np.array(result.view2d, dtype=complex)


cdef DoubleComplexMemory ifft2_(double complex[:, :] array):
    """二维离散傅里叶逆变换.

    Args:
    array:
        需要进行逆变换的数组

    Returns:
        返回逆变换结果
    """

    cdef:
        Py_ssize_t i
        Py_ssize_t rows = array.shape[0]
        Py_ssize_t cols = array.shape[1]
        Py_ssize_t size = rows * cols
        DoubleComplexMemory temp = DoubleComplexMemory(size)
        DoubleComplexMemory result = DoubleComplexMemory(size)
        double complex[:, ::1] result_view = result.init_2dview(rows, cols)
        double complex[:, ::1] temp_view = temp.init_2dview(rows, cols)
    with nogil:
        if c_log2(cols) % 1 == 0:
            for i in prange(rows):
                ifft_[complex[:]](temp_view[i, ...], array[i, ...])
        else:
            for i in prange(rows):
                ifft_by_recursion[complex[:]](temp_view[i, ...], array[i, ...])
        if c_log2(rows) % 1 == 0:
            for i in prange(cols):
                ifft_[complex[:]](result_view[..., i], temp_view[..., i])
        else:
            for i in prange(cols):
                ifft_by_recursion[complex[:]](result_view[..., i], temp_view[..., i])
    return result


cpdef cnp.ndarray[double complex, ndim=2] ifft2(double complex[:, :] array):
    cdef:
        DoubleComplexMemory result = ifft2_(array)
    return np.array(result.view2d, dtype=complex)

cdef void *fftshift_(double complex[:, :] array):
    cdef:
        Py_ssize_t rows = array.shape[0]
        Py_ssize_t cols = array.shape[1]
        Py_ssize_t center_row = rows / 2
        Py_ssize_t center_col = cols / 2
        DoubleComplexMemory temp = DoubleComplexMemory(rows * cols)
        double complex[:, :] temp_view = temp.init_2dview(rows, cols)
    temp_view[:] = array
    array[-center_row:, -center_col:] = temp_view[: center_row, : center_col]
    array[-center_row:, : cols - center_col] = temp_view[: center_row, center_col:]
    array[: rows - center_row, -center_col:] = temp_view[center_row:, : center_col]
    array[: rows - center_row, : cols - center_col] = temp_view[center_row:, center_col:]


cpdef cnp.ndarray[double complex, ndim=2] fftshift(double complex[:, :] array):
    fftshift_(array)
    return np.array(array)


cdef void *ifftshift_(double complex[:, :] array):
    cdef:
        Py_ssize_t rows = array.shape[0]
        Py_ssize_t cols = array.shape[1]
        Py_ssize_t center_row = rows / 2
        Py_ssize_t center_col = cols / 2
        DoubleComplexMemory temp = DoubleComplexMemory(rows * cols)
        double complex[:, :] temp_view = temp.init_2dview(rows, cols)
    temp_view[:] = array
    array[: center_row, : center_col] = temp_view[-center_row:, -center_col:]
    array[: center_row, center_col:] = temp_view[-center_row:, : cols - center_col]
    array[center_row:, : center_col] = temp_view[: rows - center_row, -center_col:]
    array[center_row:, center_col:] = temp_view[: rows - center_row, : cols - center_col]


cpdef cnp.ndarray[double complex, ndim=2] ifftshift(double complex[:, :] array):
    ifftshift_(array)
    return np.array(array)


cpdef cnp.ndarray[double, ndim=2] spacial_filter(double[:, :] img, double[:, :] kernel):
    """空域滤波器.

    Args:
    img:
        输入图像(单通道)
    kernel:
        滤波核

    Returns:
        返回滤波结果
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        Py_ssize_t rows_ = rows
        Py_ssize_t cols_ = cols
        Py_ssize_t kernel_rows = kernel.shape[0]
        Py_ssize_t kernel_cols = kernel.shape[1]
        Py_ssize_t size = rows * cols * sizeof(double)
        DoubleMemory kernel_padded = DoubleMemory(size)
        DoubleMemory img_padded
        DoubleComplexMemory fft_img, temp
        bint if_new_img = 0
        double temp_value
        double[:, :] temp_img
        double complex[:, ::1] temp_view1, temp_view2
    temp_value = c_log2(<double> rows)
    if temp_value % 1 != 0:
        rows_ = 1 << (<int> temp_value + 1)
        if_new_img = 1
    temp_value = c_log2(<double> cols)
    if temp_value % 1 != 0:
        cols_ = 1 << (<int> temp_value + 1)
        if_new_img = 1
    if if_new_img:
        img_padded = DoubleMemory(rows_ * cols_ * sizeof(double))
        temp_img = img_padded.init_2dview(rows_, cols_)
        temp_img[:] = 0.
        temp_img[: rows, : cols] = img
        img = temp_img
    kernel_padded.init_2dview(rows_, cols_)
    kernel_padded.view2d[:] = 0
    kernel_padded.view2d[: kernel_rows , : kernel_cols] = kernel[::-1, ::-1]
    temp = fft2_(kernel_padded.view2d)
    temp_view1 = temp.view2d
    fft_img = fft2_(img)
    temp_view2 = fft_img.view2d
    for i in range(rows_):
        for j in range(cols_):
            temp_view2[i, j] *= temp_view1[i, j]
    fft_img = ifft2_(temp_view2)
    return np.abs(fft_img.view2d[kernel_rows - 1: rows, kernel_cols - 1: cols])


cdef DoubleMemory init_gaussian_kernel(Py_ssize_t kernel_rows, Py_ssize_t kernel_cols, double sigma):
    """初始化高斯核, 对其进行赋值.

    Args:
    kernel_rows:
        核行数
    kernel_cols:
        核列数
    sigma:
        标准差

    Returns:
        None
    """

    cdef:
        DoubleMemory kernel = DoubleMemory(kernel_rows * kernel_cols)
        double[:, ::1] kernel_view = kernel.init_2dview(kernel_rows, kernel_cols)
        Py_ssize_t row_move_distance = kernel_rows / 2 + 1
        Py_ssize_t col_move_distance = kernel_cols / 2 + 1
        Py_ssize_t i, j
        double s = 0.
        double modulus
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            modulus = c_pow(<double> i - row_move_distance, 2.) + c_pow(<double> j - col_move_distance, 2.)
            kernel_view[i, j] = c_exp(-modulus / c_pow(sigma, 2.) / 2)
            s += kernel_view[i, j]
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            kernel_view[i, j] /= s
    return kernel


cpdef cnp.ndarray[double, ndim=2] gaussian_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols, double sigma):
    """高斯滤波.

    Args:
    img:
        输入图像(单通道)
    kernel_size:
        滤波核大小
    sigma:
        标准差

    Returns:
        返回滤波结果
    """

    cdef:
        DoubleMemory kernel
        cnp.ndarray[double, ndim=2] result
    kernel = init_gaussian_kernel(kernel_rows, kernel_cols, sigma)
    result = spacial_filter(img, kernel.view2d)
    return result


cpdef cnp.ndarray[double, ndim=2] gaussian_low_pass(double[:, :] img, double sigma):
    """高斯低通滤波器.

    Args:
    img:
        输入图像(单通道)
    sigma:
        标准差

    Returns:
        返回滤波结果
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        DoubleMemory kernel
        DoubleComplexMemory fft_img
        double center
        double[:, ::1] kernel_view
        double complex[:, ::1] fft_img_view
    fft_img = fft2_(img)
    fft_img_view = fft_img.view2d
    kernel = init_gaussian_kernel(rows, cols, sigma)
    kernel_view = kernel.view2d
    center = kernel_view[rows / 2 + 1, cols / 2 + 1]
    fftshift_(fft_img_view)
    for i in range(rows):
        for j in range(cols):
            fft_img_view[i, j] *= kernel_view[i, j] / center
    ifftshift_(fft_img_view)
    fft_img = ifft2_(fft_img_view)
    return np.abs(fft_img.view2d)


cpdef cnp.ndarray[double, ndim=2] mean_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols):
    """均值滤波.

    Args:
    img:
        输入图像(单通道)
    kernel_size:
        滤波核大小

    Returns:
        返回滤波结果
    """

    cdef:
        int size = kernel_rows * kernel_cols
        Py_ssize_t i, j, k, l
        Py_ssize_t rows = img.shape[0] - kernel_rows + 1
        Py_ssize_t cols = img.shape[1] - kernel_cols + 1
        double sum_ = 0.
        double[:, :] window_view
        double[:, ::1] result_view
        cnp.ndarray[double, ndim=2] result = np.empty([rows, cols])
    result_view = result
    with nogil:
        for i in range(rows):
            for j in range(cols):
                if j == 0:
                    window_view = img[i: i + kernel_rows, : kernel_cols]
                    sum_ = 0.
                    for k in range(kernel_rows):
                        for l in range(kernel_cols):
                            sum_ += window_view[k, l]
                else:
                    for k in range(kernel_rows):
                        sum_ -= img[i + k, j - 1] - img[i + k, j + kernel_cols - 1]
                result_view[i, j] = sum_ / size
    return result


cpdef cnp.ndarray[double, ndim=2] median_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols):
    """快速中值滤波.

    Args:
    img:
        输入图像(单通道)
    kernel_size:
        滤波核大小

    Returns:
        返回滤波结果
    """

    cdef:
        int cum_sum = 0
        int left, right
        int threshold = kernel_rows * kernel_cols / 2
        int histongram[256]
        int[::1] histongram_view = histongram
        Py_ssize_t rows = img.shape[0] - kernel_rows + 1
        Py_ssize_t cols = img.shape[1] - kernel_cols + 1
        Py_ssize_t i, j, k, l
        double[:, ::1] result_view
        double [:, :] window_view
        double median = 0.
        cnp.ndarray[double, ndim=2] result = np.empty([rows, cols])
    result_view = result
    with nogil:
        for i in range(rows):
            for j in range(cols):
                if j == 0:
                    window_view = img[i: i + kernel_rows, : kernel_cols]
                    cum_sum = 0
                    # 初始化直方图
                    histongram_view[:] = 0
                    # 更新直方图
                    for k in range(kernel_rows):
                        for l in range(kernel_cols):
                            histongram_view[<int> window_view[k, l]] += 1
                    # 通过计算累计直方图得到中值
                    for k in range(256):
                        cum_sum += histongram_view[k]
                        if cum_sum >= threshold:
                            median = <double> k + 1
                            break
                else:
                    # 减去最左边的一列, 加上最右边一列
                    for k in range(kernel_rows):
                        left = <int> img[i + k, j - 1]
                        histongram_view[left] -= 1
                        if left < median:
                            cum_sum -= 1
                        elif left == median:
                            for l in range(<Py_ssize_t> median, 256):
                                if histongram_view[l] != 0:
                                    median = <double> l
                                    break
                        right = <int> img[i + k, j + kernel_cols - 1]
                        histongram_view[right] += 1
                        if right < median:
                            cum_sum += 1
                    # 小于中值个数小于阈值则以当前中值为起点继续累加直到cum_sum超过阈值
                    if cum_sum < threshold:
                        for k in range(<Py_ssize_t> median, 256):
                            cum_sum += histongram_view[k]
                            if cum_sum >= threshold:
                                median = <double> k + 1
                                break
                    # 小于中值个数大于阈值则以当前中值减一为起点累减直到cum_sum回到阈值
                    elif cum_sum > threshold:
                        for k in range(<Py_ssize_t> median - 1, -1, -1):
                            cum_sum -= histongram_view[k]
                            if cum_sum <= threshold:
                                median = <double> k
                                break
                result_view[i, j] = median
    return result


ctypedef bint (*MaxMin)(double, double)


cdef bint bigger(double a, double b):
    return a > b


cdef bint smaller(double a, double b):
    return a < b


cdef DoubleMemory max_min_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols, bint filter_type):
    """最值滤波器.

    Args:
    img:
        输入图像(单通道)
    kernel_size:
        核大小
    filter_type:
        取最大或最小

    Returns:
        结果视图
    """

    cdef:
        int step = 1
        int histongram[256]
        int[::1] histongram_view = histongram
        Py_ssize_t i, j, k, l
        Py_ssize_t rows = img.shape[0] - kernel_rows + 1
        Py_ssize_t cols = img.shape[1] - kernel_cols + 1
        double base = 255.
        double pixel = 0.
        double v = 0.
        double left, right
        MaxMin compare_func = smaller
        double[:, :] window_view
        double[:, ::1] result_view
        DoubleMemory result = DoubleMemory(rows * cols)
    result_view = result.init_2dview(rows, cols)
    if filter_type:
        base = 0.
        step = -1
        compare_func = bigger
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                window_view = img[i: i + kernel_rows, : kernel_cols]
                histongram_view[:] = 0
                v = base
                for k in range(kernel_rows):
                    for l in range(kernel_cols):
                        pixel = window_view[k, l]
                        histongram_view[lround(pixel)] += 1
                        if compare_func(pixel, v):
                            v = pixel
            else:
                pixel = v
                for k in range(kernel_rows):
                    left = img[i + k, j - 1]
                    right = img[i + k, j + kernel_cols - 1]
                    histongram_view[lround(left)] -= 1
                    histongram_view[lround(right)] += 1
                    if compare_func(right, pixel):
                        pixel = right
                if compare_func(pixel, v):
                    v = pixel
                else:
                    while 1:
                        if histongram_view[lround(v)] > 0:
                            break
                        v += step
            result_view[i, j] = v
    return result


cpdef cnp.ndarray[double, ndim=2] max_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols):
    cdef:
        DoubleMemory result = max_min_filter(img, kernel_rows, kernel_cols, 1)
    return np.array(result.view2d)


cpdef cnp.ndarray[double, ndim=2] min_filter(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols):
    cdef:
        DoubleMemory result = max_min_filter(img, kernel_rows, kernel_cols, 0)
    return np.array(result.view2d)


cpdef cnp.ndarray[int, ndim=2] morphology(int[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols, bint flag):
    """形态学处理.

    Args:
    img:
        输入图像(单通道)
    kernel_rows:
        核行数
    kernel_cols:
        核列数
    flag:
        0腐蚀
        1膨胀

    Returns:
        返回结果
    """

    cdef:
        Py_ssize_t i, j, k, l
        Py_ssize_t rows = img.shape[0] - kernel_rows + 1
        Py_ssize_t cols = img.shape[1] - kernel_cols + 1
        int histongram[2]
        int[:, :] window_view
        int[::1] histongram_view = histongram
        cnp.ndarray[int, ndim=2] result = np.empty([rows, cols], dtype=np.int32)
        int[:, ::1] result_view = result
    with nogil:
        for i in range(rows):
            for j in range(cols):
                if j == 0:
                    histongram_view[:] = 0
                    window_view = img[i: i + kernel_rows, : kernel_cols]
                    for k in range(kernel_rows):
                        for l in range(kernel_cols):
                            if window_view[k, l] > 0:
                                histongram_view[1] += 1
                            else:
                                histongram_view[0] += 1
                else:
                    for k in range(kernel_rows):
                        if img[i + k, j - 1] > 0:
                            histongram_view[1] -= 1
                        else:
                            histongram_view[0] -= 1
                        if img[i + k, j + kernel_cols - 1] > 0:
                            histongram_view[1] += 1
                        else:
                            histongram_view[0] += 1
                if histongram_view[flag] > 0:
                    result_view[i, j] = 255 * flag
                else:
                    result_view[i, j] = 255 * (not flag)
    return result


cpdef cnp.ndarray[double, ndim=2] general_morphology(double[:, :] img, Py_ssize_t kernel_rows, Py_ssize_t kernel_cols, bint flag):
    cdef:
        DoubleMemory result = max_min_filter(img, kernel_rows, kernel_cols, flag)
    return np.array(result.view2d)


cdef double global_thresholding(double[:, :] img):
    """全局化阈值处理.

    Args:
    img:
        输入图像(单通道)

    Returns:
        阈值
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        int size = rows * cols
        int count = 0
        double s0 = 0., s1 = 0.
        double T0= 0., T1 = 0.
    with nogil:
        for i in prange(rows):
            for j in range(cols):
                s0 += img[i, j]
        T0 = s0 / size
        while 1:
            for i in prange(rows):
                for j in range(cols):
                    if img[i, j] > T0:
                        count += 1
                        s1 += img[i, j]
            T1 = (s1 / count + (s0 - s1) / (size - count)) / 2
            if fdim(T0, T1) < 3.:
                break
            T0 = T1
            s1 = 0.
            count = 0
    return T1


cdef double otsu_thresholding(double[:, :] img):
    """大津化二值算法.

    Args:
    img:
        输入图像(单通道)

    Returns:
        阈值
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        int T = 0
        int length1 = 0, length2
        int count = 0
        int size = rows * cols
        int histongram[256]
        int[::1] histongram_view = histongram
        double s0 = 0., s1 = 0.
        double classify_var, pixel
        double max_var = 0.
    histongram_view[:] = 0
    with nogil:
        for i in prange(rows):
            for j in range(cols):
                pixel = img[i, j]
                s0 += pixel
                histongram_view[<int> pixel] += 1
        for i in range(255):
            s1 += i * histongram_view[i]
            length1 += histongram_view[i]
            length2 = size - length1
            classify_var = length1 * length2 * c_pow(s1 / length1 - (s0 - s1) / length2, 2.)
            if classify_var > max_var:
                max_var = classify_var
                T = i
    return <double> T


cpdef cnp.ndarray[double, ndim=2] thresholding(double[:, :] img, bint flag):
    """图像阈值化.

    Args:
    img:
        输入图像(单通道)
    flag:
        阈值化方式

    Returns:
        返回二值化结果
    """

    cdef:
        Py_ssize_t i, j
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        double T = 0.
        double[:, ::1] result_view
        cnp.ndarray[double, ndim=2] result = np.empty([rows, cols])
    result_view = result
    if flag:
        T = otsu_thresholding(img)
    else:
        T = global_thresholding(img)
    for i in range(rows):
        for j in range(cols):
            if img[i, j] > T:
                result_view[i, j] = 255.
            else:
                result_view[i, j] = 0.
    return result


cpdef cnp.ndarray[double, ndim=2] canny(double[:, :] img, double low_threshold, double high_threshold):
    """canny边缘检测.

    Args:
    img:
        输入图像
    low_threshold:
        低阈值
    high_threshold:
        高阈值

    Returns:
        返回检测结果
    """

    cdef:
        Py_ssize_t i, j, k, l
        Py_ssize_t rows
        Py_ssize_t cols
        double pixel, point1, point2, slope
        double kernel[3][3]
        double[:, :] window_view
        double[:, ::1] kernel_view
        double[::1, :] kernel_T_view
        double[:, ::1] dx_view, dy_view, result_view, temp_view
        DoubleMemory temp
        cnp.ndarray[double, ndim=2] result
    kernel[0] = [-1, 0, 1]
    kernel[1] = [-2, 0, 2]
    kernel[2] = [-1, 0, 1]
    kernel_view = kernel
    kernel_T_view = kernel_view.T
    dx_view = spacial_filter(img, kernel_view)
    dy_view = spacial_filter(img, kernel_T_view)
    rows = dx_view.shape[0]
    cols = dx_view.shape[1]
    result = np.empty([rows, cols])
    result_view = result
    temp = DoubleMemory(rows * cols)
    temp_view = temp.init_2dview(rows, cols)
    with nogil:
        for i in prange(rows):
            for j in range(cols):
                result_view[i, j] = hypot(dx_view[i, j], dy_view[i, j])
        temp_view[:] = result_view
        for i in prange(1, rows - 1):
            for j in range(1, cols - 1):
                pixel = result_view[i, j]
                if dx_view[i, j] == 0.:
                    point1 = result_view[i, j - 1]
                    point2 = result_view[i, j + 1]
                elif dy_view[i, j] == 0.:
                    point1 = result_view[i - 1, j]
                    point2 = result_view[i + 1, j]
                else:
                    slope = dy_view[i, j] / dx_view[i, j]
                    if slope > 1.:
                        point1 = result_view[i + 1, j] * (1 + (result_view[i + 1, j + 1] - result_view[i + 1, j]) / slope)
                        point2 = result_view[i - 1, j] * (1 + (result_view[i - 1, j - 1] - result_view[i - 1, j]) / slope)
                    elif slope < -1.:
                        point1 = result_view[i - 1, j] * (1 + (result_view[i - 1, j - 1] - result_view[i - 1, j]) / slope)
                        point2 = result_view[i + 1, j] * (1 + (result_view[i + 1, j + 1] - result_view[i + 1, j]) / slope)
                    elif 0. < slope <= 1.:
                        point1 = result_view[i, j + 1] * (1 + (result_view[i + 1, j + 1] - result_view[i + 1, j]) * slope)
                        point2 = result_view[i, j - 1] * (1 + (result_view[i - 1, j - 1] - result_view[i, j - 1]) * slope)
                    else:
                        point1 = result_view[i, j - 1] * (1 + (result_view[i + 1, j - 1] - result_view[i, j - 1]) * slope)
                        point2 = result_view[i, j + 1] * (1 + (result_view[i - 1, j - 1] - result_view[i, j + 1]) * slope)
                if point1 > pixel or point2 > pixel:
                    temp_view[i, j] = 0.
        result_view[:] = temp_view
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                pixel = temp_view[i, j]
                if pixel <= low_threshold:
                    result_view[i, j] = 0.
                elif pixel > high_threshold:
                    result_view[i, j] = 255.
                else:
                    window_view = temp_view[i - 1: i + 2, j - 1: j + 2]
                    for k in range(3):
                        for l in range(3):
                            if window_view[k, l] > high_threshold:
                                result_view[i, j] = 255.
                            else:
                                result_view[i, j] = 0.
    return result[1: -1, 1: -1]


cpdef cnp.ndarray[int, ndim=2] hough(double[:, :] img, int threshold):
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t rows = img.shape[0]
        Py_ssize_t cols = img.shape[1]
        int r_max = <int> hypot(<double> rows, <double> cols) + 1
        Py_ssize_t r_length = r_max * 2
        int size = 49
        int count = 0
        IntMemory histongram = IntMemory(180 * r_length)
        int[:, ::1] histongram_view = histongram.init_2dview(r_length, 180)
        IntMemory temp = IntMemory(180 * r_length)
        int[:, ::1] temp_view = temp.init_2dview(r_length, 180)
        DoubleMemory result = DoubleMemory(size * 2)
        double theta, pixel
        double thetas[180]
        double[::1] thetas_view = thetas
        int[:, :] window_view
        double[:, ::1] result_view = result.init_2dview(2, size)
    histongram_view[:] = 0
    with nogil:
        for i in prange(180):
            theta = pi / 180 * i
            thetas_view[i] = theta
            for j in range(rows):
                for k in range(cols):
                    if img[j, k] > 0:
                        histongram_view[<int> (j * c_sin(theta) + k * c_cos(theta)) + r_max, i] += 1
        temp_view[:] = histongram_view
        for i in range(1, 180 - 1):
            for j in range(1, r_length - 1):
                pixel = temp_view[j, i]
                window_view = temp_view[j - 1: j + 2, i - 1: i + 2]
                for k in range(3):
                    if window_view[k, 0] > pixel or \
                            window_view[k, 1] > pixel or \
                            window_view[k, 2] > pixel:
                        histongram_view[j, i] = 0
                        break
    for i in range(180):
        for j in range(r_length):
            if histongram_view[j, i] > threshold:
                result_view[0, count] = thetas_view[i]
                result_view[1, count] = <double> (<int> j - r_max)
                count += 1
                if count >= size:
                    size *= 2
                    result.resize(size * 2)
                    result_view = result.init_2dview(2, size)
    return np.array(result_view[:, : count])

