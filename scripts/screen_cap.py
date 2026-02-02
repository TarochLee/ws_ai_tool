import Quartz
from PIL import Image
import numpy as np

def capture_screen(filename="pic.jpg"):
    # 1. 获取主显示器边界
    display_id = Quartz.CGMainDisplayID()
    bounds = Quartz.CGDisplayBounds(display_id)

    # 2. 从系统图形层抓屏幕图像
    image_ref = Quartz.CGWindowListCreateImage(
        bounds,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault
    )

    if image_ref is None:
        raise RuntimeError("Failed to capture screen")

    # 3. 取像素数据
    width = Quartz.CGImageGetWidth(image_ref)
    height = Quartz.CGImageGetHeight(image_ref)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
    data_provider = Quartz.CGImageGetDataProvider(image_ref)
    data = Quartz.CGDataProviderCopyData(data_provider)

    # 4. 转为 numpy array（BGRA）
    img = np.frombuffer(data, dtype=np.uint8)
    img = img.reshape((height, bytes_per_row // 4, 4))
    img = img[:, :width, :]  # 去掉 padding

    # 5. BGRA → RGB
    img = img[:, :, :3][:, :, ::-1]

    # 6. 保存为 jpg
    Image.fromarray(img).save(filename, "JPEG")
    print(f"saved: {filename}")

if __name__ == "__main__":
    capture_screen("pic.jpg")