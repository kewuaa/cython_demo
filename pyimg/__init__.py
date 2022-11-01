from functools import wraps
import asyncio

import numpy as np
import pyximport
pyximport.install(language_level=3)

from pyimg import pyimg


__all__ = ['BGR2GRAY', 'fft', 'ifft', 'fft2', 'ifft2', 'imgfilter', 'gaussian_low_pass']
_loop = asyncio.get_event_loop()
FILTER_MEDIAN, FILTER_MEAN, FILTER_MAX, FILTER_MIN = range(4)
THRESHOLD_GLOBAL, THRESHOLD_OTSU = range(2)
MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE, MORPH_TOPHAT, MORPH_BLACKHAT = range(6)


def async2sync(async_func):
    @wraps(async_func)
    def sync_func(*args, **kwargs):
        f"""{async_func.__doc__}"""

        return _loop.run_until_complete(async_func(*args, **kwargs))

    # sync_func.__doc__ = async_func.__doc__
    return sync_func


async def apply_over_channels(image: np.ndarray, func) -> np.ndarray:
    """并行处理图像的每个通道.

    Args:
    image:
        输入三通道图像
    func:
        需要对每个通道应用的函数

    Returns:
        返回结果
    """

    futures = [asyncio.get_running_loop().run_in_executor(None, func, image[..., i]) for i in range(image.shape[-1])]
    result = [await future for future in futures]
    return np.stack(result, axis=2)


def BGR2GRAY(image: np.ndarray) -> np.ndarray:
    """灰度转换.

    Args:
    image:
        输入三通道图像

    Returns:
        返回灰度图像
    """

    assert image.ndim < 3, '单通道图像无法灰度化'
    return pyimg.BGR2GRAY(image.astype(np.float64))


def fft(array) -> np.ndarray:
    """一维fft变换.

    Args:
    array:
        一维序列

    Returns:
        变换结果
    """

    return pyimg.fft(np.asarray(array, dtype=np.float64))


def ifft(array) -> np.ndarray:
    """一维fft逆变换.

    Args:
    array:
        一维序列

    Returns:
        变换结果
    """

    return pyimg.ifft(np.asarray(array, dtype=np.complex64))


async def _fft2(array: np.ndarray) -> np.ndarray:
    """二维fft变换.

    Args:
    array:
        二维数组

    Returns:
        变换结果
    """

    if array.ndim > 2:
        return await apply_over_channels(array.astype(np.float64), pyimg.fft2)
    return pyimg.fft2(array.astype(np.float64))


async def _ifft2(array: np.ndarray) -> np.ndarray:
    """二维fft变换.

    Args:
    array:
        二维数组

    Returns:
        变换结果
    """

    if array.ndim > 2:
        return await apply_over_channels(array.astype(np.complex64), pyimg.ifft2)
    return pyimg.ifft2(array.astype(np.complex64))


def fftshift(array: np.ndarray) -> np.ndarray:
    return pyimg.fftshift(array.astype(np.complex64))


def ifftshift(array: np.ndarray) -> np.ndarray:
    return pyimg.ifftshift(array.astype(np.complex64))


async def _imgfilter(image: np.ndarray,
                     *kernel_shape, kernel: np.ndarray = None, filter_type: int = None):
    """图像滤波器.

    Args:
    image:
        输入图像
    kernel_shape:
        滤波核形状, 针对非自定义核
    kernel:
        自定义滤波核
    filter_type:
        选择中值, 均值, 最值滤波器, 如输入自定义滤波核, 则该参数无效

    Returns:
        返回滤波后的图像
    """

    processor = None
    if kernel is not None:
        processor = lambda img: pyimg.spacial_filter(img, kernel.astype(np.float64))
    elif filter_type is not None:
        kernel_rows, kernel_cols, *_ = kernel_shape * 2
        all_types = 'median', 'mean', 'max', 'min'
        processor = lambda img: pyimg.__getattribute__(
                all_types[filter_type] + '_filter'
                )(img, kernel_rows, kernel_cols)
    else:
        raise
    img = image.astype(np.float64)
    if img.ndim > 2:
        return await apply_over_channels(img, processor)
    return processor(img)


async def _gaussian_filter(image: np.ndarray,
                          *kernel_shape, sigma: float = 1.4) -> np.ndarray:
    """高斯滤波.

    Args:
    image:
        输入图像
    kernel_shape:
        滤波核形状
    sigma:
        标准差

    Returns:
        返回滤波后的图像
    """

    img = image.astype(np.float64)
    kernel_rows, kernel_cols = kernel_shape * 2
    if img.ndim > 2:
        return await apply_over_channels(img, lambda _img: pyimg.gaussian_filter(_img, kernel_rows, kernel_cols, sigma))
    return pyimg.gaussian_filter(img, kernel_rows, kernel_cols, sigma)


async def _gaussian_low_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    """高斯低通滤波.

    Args:
    image:
        输入图像
    sigma:
        高斯标准差

    Returns:
        返回滤波后的图像
    """

    img = image.astype(np.float64)
    if img.ndim > 2:
        return await apply_over_channels(img, lambda img:pyimg.gaussian_low_pass(img, sigma))
    else:
        return pyimg.gaussian_low_pass(img, sigma)


def thresholding(image: np.ndarray, flag: int):
    """阈值分割.

    image:
        输入图像
    flag:
        选择全局阈值分割或otsu阈值分割

    Returns:
        返回二值图
    """

    if image.ndim > 2:
        img = pyimg.BGR2GRAY(image.astype(np.float64))
    else:
        img = image.astype(np.float64)
    return pyimg.thresholding(img, flag)


def canny(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """canny边缘检测.

    Args:
    image:
        输入图像
    low_threshold:
        低阈值
    high_threshold:
        高阈值

    Returns:
        返回边缘检测结果
    """

    if image.ndim > 2:
        img = pyimg.BGR2GRAY(image.astype(np.float64))
    else:
        img = image.astype(np.float64)
    return pyimg.canny(img, low_threshold, high_threshold)


def morphology(image: np.ndarray,
               *kernel_shape, flag: int, general: bool = True) -> np.ndarray:
    """形态学操作.

    Args:
    image:
        输入二值图像
    flag:
        选择需要进行的形态学处理

    Returns:
        返回结果
    """

    assert image.ndim < 3, '请输入正确的二值图'
    _flag = flag % 2
    if general:
        img = image.astype((np.float64))
        morph = pyimg.general_morphology
    else:
        img = image.astype(np.int32)
        morph = pyimg.morphology
    kernel_row, kernel_col, *_ = kernel_shape * 2
    result = morph(img, kernel_row, kernel_col, _flag)
    if flag < 2:
        return result
    result = morph(result, kernel_row, kernel_col, not _flag)
    if flag < 4:
        return result
    rows, cols = img.shape
    new_rows, new_cols = rows - kernel_row * 2 + 2, cols - kernel_col * 2 + 2
    row_pad, col_pad = kernel_row // 2 * 2, kernel_col // 2 * 2
    result -= img[row_pad: row_pad + new_rows, col_pad: col_pad + new_cols]
    return result * (-1) ** (_flag + 1)


def add_nosie(img: np.ndarray, noise_type: int,
              *, noise_point_num: int = None, SNR: float = None, mean: float = None, std: float = None) -> np.ndarray:
    """图像加噪.

    Args:
    img:
        输入图像
    noise_type:
        噪声类型
        examples:
            0: 随机噪声
            1: 椒盐噪声
            2: 高斯噪声
    noise_point_num:
        噪声点数量, 噪声类型为random时需要指定
    SNR:
        信噪比, 噪声类型为salt时需要指定
    mean:
        均值, 噪声类型为gaussian时需要指定
    std:
        标准差, 噪声类型为gaussian时需要指定

    Returns:
        加噪后的图像
    """

    img_with_noise = img.copy()
    if noise_type == 0:
        assert noise_point_num is not None, '请输入噪声点个数'
        rows, cols, _ = img_with_noise.shape
        img_with_noise[np.random.randint(0, rows, noise_point_num),
                       np.random.randint(0, cols, noise_point_num)] = 255
    elif noise_type == 1:
        assert SNR is not None, '请输入信噪比'
        threshold = 1 - SNR
        random_mask = np.random.choice((-1, 0, 1), size=img_with_noise.shape[: -1], p=[threshold / 2, SNR, threshold / 2])
        img_with_noise[random_mask > 0] = 255
        img_with_noise[random_mask < 0] = 0
    elif noise_type == 2:
        assert not (mean is None and std is None), '请输入均值和标准差'
        noise = np.random.normal(mean, std, size=img_with_noise.shape)
        img_with_noise = np.clip(img_with_noise / 255 + noise, 0, 1)
        img_with_noise = (img_with_noise * 255).astype(np.uint8)
    return img_with_noise


fft2 = async2sync(_fft2)
ifft2 = async2sync(_ifft2)
imgfilter = async2sync(_imgfilter)
gaussian_filter = async2sync(_gaussian_filter)
gaussian_low_pass = async2sync(_gaussian_low_pass)


if __name__ == '__main__':
    print(imgfilter.__doc__)

