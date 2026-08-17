---
sr-due: 2026-08-19
sr-interval: 3
sr-ease: 250
---
#review 

## remove_background

```python
def remove_background(image, red_image, cfg_line=None):
    """黑线背景去除 & 二值化。

    处理流程:
      1. 高斯模糊 —— 平滑纹理（如地砖接缝）
      2. Otsu 二值化 —— 画面有黑线时自动找阈值，否则回退到固定阈值
      3. 形态学开/闭运算 —— 去除噪声 + 连接断裂黑线
      4. 合并红色遮罩 —— 将黑线与红点区域合并

    Args:
        image: 灰度图 (H, W)
        red_image: 红色掩码图 (H, W)，0=非红, 255=红
        cfg_line: 线检测配置节（_SectionView），为 None 时使用默认值

    Returns:
        二值化掩码图 (H, W)，0=背景, 255=黑线/红色
    """
    # ---- 从配置提取参数，None 时回退到默认值 ----
    gaussian_blur_kernel = 15
    otsu_valid_max = 150
    fixed_threshold = 90
    morph_open_kernel = 7
    morph_close_kernel = 11

    if cfg_line is not None:
        gaussian_blur_kernel = cfg_line.gaussian_blur_kernel
        otsu_valid_max = cfg_line.otsu_valid_max
        fixed_threshold = cfg_line.fixed_threshold
        morph_open_kernel = cfg_line.morph_open_kernel
        morph_close_kernel = cfg_line.morph_close_kernel

    # 确保高斯核为奇数
    if gaussian_blur_kernel % 2 == 0:
        gaussian_blur_kernel += 1

    # 1. 高斯模糊
    blurred = cv2.GaussianBlur(image, (gaussian_blur_kernel, gaussian_blur_kernel), 0)

    # 2. Otsu 二值化，无效时回退固定阈值
    otsu_thr, otsu_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if otsu_thr <= otsu_valid_max:
        mask = otsu_mask
    else:
        _, mask = cv2.threshold(blurred, fixed_threshold, 255, cv2.THRESH_BINARY_INV)

    # 3. 形态学清理
    kernel_open = np.ones((morph_open_kernel, morph_open_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = np.ones((morph_close_kernel, morph_close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 4. 合并红色遮罩
    mask = cv2.bitwise_or(mask, red_image)

    return mask

```

## 二维码色块识别

```python

def _detect_qr_block_center(gray_img, last_qr_center=None, cfg_qr=None):
    """对整张灰度图做二维码黑块检测，返回符合条件的轮廓与二值 mask。

    Args:
        gray_img: 灰度图
        last_qr_center: 已废弃，仅为兼容调用方签名保留，不再参与任何过滤
        cfg_qr: 二维码检测配置节，为 None 时使用默认背景去除管线

    Returns:
        (blocks, mask): blocks 为通过筛选的轮廓列表，mask 为全图二值 mask
    """
    # 对整图做背景去除 / 阈值化
    # 背景去除的方式与remove_background参数不同
    if cfg_qr is None:
        mask = remove_background(gray_img, np.zeros_like(gray_img))
        work = mask
        min_area, max_ratio, rmin, rmax = 80, 0.45, 0.45, 2.2
    else:
        # 确保高斯核为奇数
        k = cfg_qr.block_blur_kernel + (cfg_qr.block_blur_kernel % 2 == 0)
        blur = cv2.GaussianBlur(gray_img, (k, k), 0)
        _, work = cv2.threshold(blur, cfg_qr.block_threshold, 255, cv2.THRESH_BINARY_INV)
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN,
                                np.ones((cfg_qr.block_open_kernel, cfg_qr.block_open_kernel), np.uint8))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE,
                                np.ones((cfg_qr.block_close_kernel, cfg_qr.block_close_kernel), np.uint8))
        mask = work
        min_area, max_ratio = cfg_qr.block_min_area, cfg_qr.block_max_area_ratio
        rmin, rmax = cfg_qr.block_aspect_ratio_min, cfg_qr.block_aspect_ratio_max

    # 通过 findContours 查找轮廓，再根据 面积和长宽比 筛选色块
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    img_area = work.shape[0] * work.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > img_area * max_ratio:
            continue

        _, _, w, h = cv2.boundingRect(contour)
        ratio = w / max(h, 1)
        if ratio < rmin or ratio > rmax:
            continue

        blocks.append(contour)

    return blocks, mask


def get_qr_block_center(gray_img, cfg_qr=None):
    """找黑白色块质心，用于低空辅助居中。

    对每个轮廓单独算矩再按面积加权合并 —— 避免 vstack 拼接多段轮廓
    产生自交多边形导致格林公式算出伪几何矩。

    Args:
        gray_img: 灰度图
        cfg_qr: 二维码检测配置节，为 None 时使用默认背景去除管线

    Returns:
        (cx, cy) 质心坐标（全图坐标），或 None
    """
    blocks, _mask = _detect_qr_block_center(gray_img, cfg_qr=cfg_qr)
    if not blocks:
        return None

    total_m00 = 0.0
    sum_m10 = 0.0
    sum_m01 = 0.0
    for c in blocks:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        total_m00 += m["m00"]
        sum_m10 += m["m10"]
        sum_m01 += m["m01"]
    if total_m00 == 0:
        return None

    cx = int(sum_m10 / total_m00)
    cy = int(sum_m01 / total_m00)

    h, w = gray_img.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        logger.warning(f"QR block center out of bounds: ({cx}, {cy}), "
                       f"image size=({w}, {h})")
        return None

    return cx, cy
```

## 红点识别

```python

def detect_red_by_rgb_diff(image, threshold=40, min_red=80, morph_open_kernel=3):
    """基于 RGB 通道差检测红色区域，返回二值 mask（纯算法，无绘制）。

    保留标量签名以向后兼容；新调用方应优先用 detect_red_candidate 传 cfg_red。
    """
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)
    diff_rg = r - g
    diff_rb = r - b
    mask = (
        np.logical_and.reduce(
            (diff_rg > threshold, diff_rb > threshold, r > min_red)
        ).astype(np.uint8)
        * 255
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_kernel, morph_open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def detect_red_candidate(frame, cfg_red):
    """红色检测 + 候选校验，返回 RedResult（纯算法，无绘制）。

    流程: RGB 差分 mask → 取最大轮廓 → minEnclosingCircle →
    按 min_enclosing_radius / max_y_position 校验是否为有效开启信号。
    校验逻辑与 main.py 原内联实现逐字段等价。
    """
    mask = detect_red_by_rgb_diff(
        frame,
        threshold=cfg_red.rgb_diff_threshold,
        min_red=cfg_red.min_red,
        morph_open_kernel=cfg_red.morph_open_kernel,
    )
    red_ratio = round(cv2.countNonZero(mask) * 100 / mask.size, 3)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return RedResult(mask, None, None, None, False, ["无红色候选"], red_ratio)

    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    reasons = []
    if radius <= cfg_red.min_enclosing_radius:
        reasons.append("半径不足")
    if not (0 < y < cfg_red.max_y_position):
        reasons.append("Y位置越界")
    valid = not reasons
    return RedResult(mask, x, y, radius, valid, reasons, red_ratio)

```

## decode

```python
    # 二维码识别
    def decode(self, gray_img):
        qr = self.cfg.qr_detection

        # 找到图像中的条形码并进行解码 (注意：用纯灰度图解码比彩色图快非常多)
        barcodes = pyzbar.decode(gray_img)
        self.current_barcodes = barcodes
        if barcodes:
            self.get_qr = True
            self.qr_count += 1
        elif self.get_qr:
            self.lost5 += 1

        if self.get_qr:
            time.sleep(0.03)

        # 循环检测到的条形码
        for barcode in barcodes:
            x, y, w, h = barcode.rect
            # 在灰度图上画白框 (255)
            cv2.rectangle(gray_img, (x, y), (x + w, y + h), 255, 5)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            text = "{} ({})".format(barcodeData, barcodeType)
            self.scan_content = barcodeData
            logger.info(f"二维码识别内容: {self.scan_content}")
            cv2.putText(
                gray_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2
            )
            points = barcode.polygon

            offset_x = sum(point.x for point in points) / len(points)
            offset_y = sum(point.y for point in points) / len(points)

            logger.info(f"二维码中心: ({offset_x}, {offset_y})")

            self.is_forward = False
            x_err = offset_x - qr.target_x
            y_err = offset_y - qr.target_y
            x_percent = abs(x_err) / 40
            y_percent = abs(y_err) / 40

            l = 0
            # 计算二维码偏移量调节飞机对准二维码
            if (
                abs(x_err) < qr.center_tolerance_x
                and abs(y_err) < qr.center_tolerance_y
            ):
                self.qr_stable_count += 1
                self.airplaneApi.move(0, 0, 0, 0)
                self.is_code_center = self.qr_stable_count >= qr.center_stable_frames
                logger.info(
                    f"qr stable {self.qr_stable_count}/{qr.center_stable_frames}"
                )
            elif x_percent >= y_percent:
                self.qr_stable_count = 0
                if x_err < -qr.center_tolerance_x:
                    l = self.clamp(int(qr.qr_align_kp * abs(x_err)), 45, 90)
                    self.airplaneApi.move_left(l)
                    logger.info(f"qr_left:{l}")
                    self.is_code_center = False
                elif x_err > qr.center_tolerance_x:
                    l = self.clamp(int(qr.qr_align_kp * abs(x_err)), 45, 90)
                    self.airplaneApi.move_right(l)
                    logger.info(f"qr_right:{l}")
                    self.is_code_center = False
            else:
                self.qr_stable_count = 0
                if y_err < -qr.center_tolerance_y:
                    l = self.clamp(int(qr.qr_align_forward_kp * abs(y_err)), 30, 70)
                    self.airplaneApi.move_forward(l)
                    logger.info(f"qr_forward:{l}")
                    self.is_code_center = False
                elif y_err > qr.center_tolerance_y:
                    l = self.clamp(int(qr.qr_align_forward_kp * abs(y_err)), 30, 70)
                    self.airplaneApi.move_backward(l)
                    logger.info(f"qr_back:{l}")
                    self.is_code_center = False
        if not barcodes:
            self.qr_stable_count = 0
            self.is_code_center = False
        logger.info(f"qr_cnt = {self.qr_count}, qr_lost {self.lost5}")
        # 二维码识别平均耗时0.04
```

## 巡线的切片处理

```python
def get_contour_center(contour):
    """获取轮廓的中心点（纯算法）。"""
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return [0, 0]
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return [x, y]


def slice_out(binary, num_slices, min_area=20):
    """水平切片，每片取最大轮廓质心，返回 SliceResult（纯算法，无绘制）。

    每片为全宽条带，故质心 x 为全图坐标、y 为片内局部坐标（调用方按
    sl*i 还原全图 y）。无效片质心为 [0,0]、轮廓为 None。

    修复: 最后一片延伸到图像底部，不再截断余数行。
    """
    h, w = binary.shape[:2]
    sl = h // num_slices
    centroids = []
    contours = []
    for i in range(num_slices):
        y1 = sl * i
        y2 = h if i == num_slices - 1 else y1 + sl
        crop = binary[y1:y2, 0:w]
        found = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = [0, 0]
        main_c = None
        if found:
            main_c = max(found, key=cv2.contourArea)
            if cv2.contourArea(main_c) > min_area:
                center = get_contour_center(main_c)
            else:
                main_c = None
        centroids.append(center)
        contours.append(main_c)
    return SliceResult(centroids, contours)


def draw_slices(binary, sl):
    """在二值图上按切片绘制轮廓与质心，返回 BGR 标注图（画图层，无算法）。

    输入 slice_out 的 SliceResult；逐片画绿轮廓 + 红质心圆，再垂直拼接，
    与旧 process+repack 的视觉效果一致。
    """
    h, w = binary.shape[:2]
    num_slices = len(sl.centroids)
    sl_h = h // num_slices
    bands = []
    for i in range(num_slices):
        y1 = sl_h * i
        y2 = h if i == num_slices - 1 else y1 + sl_h
        crop = binary[y1:y2, 0:w]
        color_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        c = sl.contours[i]
        if c is not None:
            cv2.drawContours(color_crop, [c], -1, (0, 255, 0), 2)
            cx, cy = sl.centroids[i]
            cv2.circle(color_crop, (cx, cy), 5, (0, 0, 255), -1)
        bands.append(color_crop)
    out = bands[0]
    for b in bands[1:]:
        out = np.concatenate((out, b), axis=0)
    return out
```
