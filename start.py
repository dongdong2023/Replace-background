# encoding=utf-8
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from mainwindow import Ui_MainWindow  # Ensure this import is correct
import cv2
import onnxruntime
import numpy as np
import yaml


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    global_color = {'红色': (0, 0, 255), '黑色': (0, 0, 0), '蓝色': (255, 0, 0), '白色': (255, 255, 255)}
    global_size = {'一寸': [413, 295], '二寸': [626, 413], '五寸': [1200, 840]}

    def __init__(self):
        super().__init__()
        self.runtime = onnxruntime.InferenceSession('cv_unet_image_matting_opset15_output_float_mask.onnx')
        self.setupUi(self)  # Initialize the UI from Ui_MainWindow
        # Additional initialization code can go here
        self.connect()
        self.load_config()

    def load_config(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.global_color = config['global_color']
            self.global_size = config['global_size']

        self.comboBox.addItems(self.global_color.keys())
        self.comboBox_2.addItems(self.global_size.keys())

    def connect(self):
        # Connect buttons to functions
        self.pushButton.clicked.connect(self.selectImage)
        self.pushButton_2.clicked.connect(self.save_image)
        self.pushButton_3.clicked.connect(self.run)

    def selectImage(self):
        # Open file dialog to select an image file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        if fileName:
            self.showImage(fileName)
            self.textBrowser.append(f'加载图片{fileName}成功')

    def showImage(self, fileName):
        # Create a QPixmap object and load the image file
        pixmap = QtGui.QPixmap(fileName)
        # Get the size of QLabel
        label_size = self.label.size()
        # Resize the image to fit QLabel
        pixmap = pixmap.scaled(label_size, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                               transformMode=QtCore.Qt.SmoothTransformation)
        # Display the image
        self.label.setPixmap(pixmap)

        # Display the file name
        self.label_2.setText(fileName)

    def run(self):
        if self.label_2.text():
            im_data = cv2.imread(self.label_2.text())
            mask = self.get_mask(im_data)
            color = self.comboBox.currentText()  # Get selected color
            size = self.comboBox_2.currentText()  # Get selected size
            background_image = self.change_background(mask, im_data, color)
            self.final_image = self.change_size(background_image, size)

            # Convert final image to QPixmap and display
            final_pixmap = self.convert_cvimage_to_qpixmap(self.final_image)
            # Get the size of QLabel
            label_size = self.label_3.size()
            # Resize the image to fit QLabel
            pixmap = final_pixmap.scaled(label_size, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                                         transformMode=QtCore.Qt.SmoothTransformation)
            self.label_3.setPixmap(pixmap)
            self.textBrowser.append(f'图片转为{color},{size}成功')
        else:
            QMessageBox.warning(self, "Warning", "请先选择图片！")

    def save_image(self):
        if self.final_image is not None:
            final_image_path = self.label_2.text().split('.')[0] + '_mask.jpg'
            cv2.imwrite(final_image_path, self.final_image)
            self.textBrowser.append(f'图片保存在{final_image_path}成功')
        else:
            QMessageBox.warning(self, "Warning", "请先运行处理图片！")

    def change_size(self, final_image, size):
        if size != '无':
            image_size = self.global_size[size]
            final_image = cv2.resize(final_image, tuple(image_size))
        return final_image

    def change_background(self, mask, im_data, color):
        background = np.zeros_like(im_data, np.uint8)
        background[:] = self.global_color[color]
        # Place foreground image onto new background based on mask
        masked_background = cv2.bitwise_and(background, background, mask=~mask)
        masked_foreground = cv2.bitwise_and(im_data, im_data, mask=mask)

        # Composite image
        final_image = masked_background + masked_foreground

        return final_image

    def get_mask(self, im_data):
        try:
            threshold = int(self.lineEdit.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "请输入有效的阈值！使用默认阈值。")
            threshold = 0  # 使用默认阈值
        input_data = im_data.astype(np.float32)
        output_img = self.runtime.run(None, {'input_image': input_data})
        mask = output_img[0].astype(np.uint8)
        mask[mask < threshold] = 0
        return mask

    def convert_cvimage_to_qpixmap(self, cv_image):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(cv_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        return QtGui.QPixmap.fromImage(q_img)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
