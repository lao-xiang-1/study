# 按USB端口位置绑定摄像头设备

查看所有视频设备
ls -la /dev/video*

查看详细的设备信息
```
udevadm info --attribute-walk --name=/dev/video4
```

查看USB设备的物理位置
lsusb -t


查看设备的物理端口信息
```
udevadm info --query=property --name=/dev/video0 | grep ID_PATH
```

创建规则文件
sudo nano /etc/udev/rules.d/99-camera-by-location.rules


文件内添加以下信息（示例）
```
SUBSYSTEM=="video4linux", KERNELS=="2-1.4:1.0", ATTR{index}=="0", SYMLINK+="camera_left_main", GROUP="video", MODE="0666"
SUBSYSTEM=="video4linux", KERNELS=="2-1.4:1.0", ATTR{index}=="1", SYMLINK+="camera_left_secondary", GROUP="video", MODE="0666"
 
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-xhci-hcd.13.auto-usb-0:1:1.0", ATTR{index}=="0", SYMLINK+="camera_center_main", GROUP="video", MODE="0666" 
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-xhci-hcd.13.auto-usb-0:1:1.0", ATTR{index}=="1", SYMLINK+="camera_center_secondary", GROUP="video", MODE="0666"

SUBSYSTEM=="video4linux", KERNELS=="7-1:1.0", ATTR{index}=="0", SYMLINK+="camera_right_main", GROUP="video", MODE="0666"
SUBSYSTEM=="video4linux", KERNELS=="7-1:1.0", ATTR{index}=="1", SYMLINK+="camera_right_secondary", GROUP="video", MODE="0666"
```

重新加载udev规则
```
sudo udevadm control --reload-rules
sudo udevadm trigger
```

检查规则是否生效
ls -la /dev/video*
