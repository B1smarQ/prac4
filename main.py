import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton, QHBoxLayout, QListWidget, QLabel, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self.cap = cap

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_signal.emit(frame)
            self.msleep(33)  

class BGSubThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, subtractor, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.subtractor = subtractor

    def run(self):
        learn_frames = 30
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            fgmask = self.subtractor.apply(frame)
            if frame_count > learn_frames:
                _, fgmask_thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fgmask_clean = cv2.morphologyEx(fgmask_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
                masked_frame = cv2.bitwise_and(frame, frame, mask=fgmask_clean)
                self.frame_signal.emit(masked_frame)
                self.msleep(33)
            frame_count += 1

class BGReplaceThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, subtractor, bg_image, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.subtractor = subtractor
        self.bg_image = bg_image

    def run(self):
        learn_frames = 30
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            fgmask = self.subtractor.apply(frame)
            bg_resized = cv2.resize(self.bg_image, (frame.shape[1], frame.shape[0]))
            mask = fgmask > 0
            result = np.where(mask[:, :, None], frame, bg_resized)
            if frame_count > learn_frames:
                self.frame_signal.emit(result)
                self.msleep(33)
            frame_count += 1

class MotionBlurThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, subtractor, blur_strength, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.subtractor = subtractor
        self.blur_strength = blur_strength

    def run(self):
        learn_frames = 30
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            fgmask = self.subtractor.apply(frame)
            mask = fgmask > 0
            if self.blur_strength % 2 == 0:
                k = self.blur_strength + 1
            else:
                k = self.blur_strength
            blurred_frame = cv2.GaussianBlur(frame, (k, k), 0)
            result = np.where(mask[:, :, None], blurred_frame, frame)
            if frame_count > learn_frames:
                self.frame_signal.emit(result)
                self.msleep(33)
            frame_count += 1

class OpticalFlowThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap, parent=None):
        super().__init__(parent)
        self.cap = cap

    def run(self):
        ret, prev_frame = self.cap.read()
        if not ret:
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.frame_signal.emit(bgr)
            prev_gray = gray
            self.msleep(33)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV")
        self.setGeometry(100, 100, 1200, 800)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.blur_strength = 5

        self.image_tab = QWidget()
        self.setup_image_tab()
        self.tab_widget.addTab(self.image_tab, "Image Similarity")

        self.video_tab = QWidget()
        self.setup_video_tab()
        self.tab_widget.addTab(self.video_tab, "Video Display")

        self.bg_sub_tab = QWidget()
        self.setup_bg_sub_tab()
        self.tab_widget.addTab(self.bg_sub_tab, "Background Subtraction")

        self.bg_replace_tab = QWidget()
        self.setup_bg_replace_tab()
        self.tab_widget.addTab(self.bg_replace_tab, "Background Replacement")

        self.motion_blur_tab = QWidget()
        self.setup_motion_blur_tab()
        self.tab_widget.addTab(self.motion_blur_tab, "Motion Blur")

        self.optical_flow_tab = QWidget()
        self.setup_optical_flow_tab()
        self.tab_widget.addTab(self.optical_flow_tab, "Optical Flow")

    def convert_cv_to_pixmap(self, cv_img, scale=300):
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaledToWidth(scale)

    def setup_image_tab(self):
        layout = QVBoxLayout()

        instructions = QLabel("Select 3-10 images, then click 'Find Similar' to display two most similar images.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.image_list = QListWidget()
        layout.addWidget(self.image_list)

        button_layout = QHBoxLayout()
        load_btn = QPushButton("Load Images")
        load_btn.clicked.connect(self.load_images)
        button_layout.addWidget(load_btn)

        find_similar_btn = QPushButton("Find Similar")
        find_similar_btn.clicked.connect(self.find_similar)
        button_layout.addWidget(find_similar_btn)

        layout.addLayout(button_layout)

        self.similar1_label = QLabel("No image")
        self.similar2_label = QLabel("No image")
        self.image_display_layout = QHBoxLayout()
        self.image_display_layout.addWidget(self.similar1_label)
        self.image_display_layout.addWidget(self.similar2_label)
        layout.addLayout(self.image_display_layout)

        self.similarity_label = QLabel("Similarity: Not calculated")
        layout.addWidget(self.similarity_label)

        self.image_tab.setLayout(layout)

    def setup_video_tab(self):
        layout = QVBoxLayout()

        instructions = QLabel("Load a video and click 'Play Video' to display it in real-time.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        button_layout = QHBoxLayout()
        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self.load_video)
        button_layout.addWidget(load_video_btn)

        play_btn = QPushButton("Play Video")
        play_btn.clicked.connect(self.play_video)
        button_layout.addWidget(play_btn)

        layout.addLayout(button_layout)

        self.video_display = QLabel()
        layout.addWidget(self.video_display)

        self.video_tab.setLayout(layout)

    def setup_bg_sub_tab(self):
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        load_bg_video_btn = QPushButton("Load Video")
        load_bg_video_btn.clicked.connect(self.load_bg_sub_video)
        button_layout.addWidget(load_bg_video_btn)

        start_bg_sub_btn = QPushButton("Start Background Subtraction")
        start_bg_sub_btn.clicked.connect(self.start_bg_subtraction)
        button_layout.addWidget(start_bg_sub_btn)

        layout.addLayout(button_layout)

        self.bg_sub_display = QLabel()
        layout.addWidget(self.bg_sub_display)

        self.bg_sub_tab.setLayout(layout)

    def setup_bg_replace_tab(self):
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        load_replace_video_btn = QPushButton("Load Video")
        load_replace_video_btn.clicked.connect(self.load_bg_replace_video)
        button_layout.addWidget(load_replace_video_btn)

        load_replace_bg_btn = QPushButton("Load Background Image")
        load_replace_bg_btn.clicked.connect(self.load_bg_image)
        button_layout.addWidget(load_replace_bg_btn)

        start_replace_btn = QPushButton("Start Replacement")
        start_replace_btn.clicked.connect(self.start_bg_replacement)
        button_layout.addWidget(start_replace_btn)

        layout.addLayout(button_layout)

        self.bg_replace_display = QLabel()
        layout.addWidget(self.bg_replace_display)

        self.bg_replace_tab.setLayout(layout)

    def setup_motion_blur_tab(self):
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        load_blur_video_btn = QPushButton("Load Video")
        load_blur_video_btn.clicked.connect(self.load_blur_video)
        button_layout.addWidget(load_blur_video_btn)

        start_blur_btn = QPushButton("Start Motion Blur")
        start_blur_btn.clicked.connect(self.start_motion_blur)
        button_layout.addWidget(start_blur_btn)

        layout.addLayout(button_layout)

        slider_label = QLabel("Blur Strength:")
        layout.addWidget(slider_label)

        from PyQt5.QtWidgets import QSlider
        from PyQt5.QtCore import Qt
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 20)
        self.blur_slider.setValue(5)
        self.blur_slider.valueChanged.connect(self.update_blur_strength)
        layout.addWidget(self.blur_slider)

        self.blur_display = QLabel()
        layout.addWidget(self.blur_display)

        self.motion_blur_tab.setLayout(layout)

    def setup_optical_flow_tab(self):
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        load_flow_video_btn = QPushButton("Load Video")
        load_flow_video_btn.clicked.connect(self.load_flow_video)
        button_layout.addWidget(load_flow_video_btn)

        start_flow_btn = QPushButton("Start Optical Flow")
        start_flow_btn.clicked.connect(self.start_optical_flow)
        button_layout.addWidget(start_flow_btn)

        layout.addLayout(button_layout)

        self.flow_display = QLabel()
        layout.addWidget(self.flow_display)

        self.optical_flow_tab.setLayout(layout)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image files (*.jpg *.png *.jpeg)")
        if len(files) < 3 or len(files) > 10:
            QMessageBox.warning(self, "Warning", "Select between 3 and 10 images.")
            return
        self.image_paths = files
        self.image_list.clear()
        self.image_list.addItems([f.split('/')[-1] for f in files])

    def find_similar(self):
        if not hasattr(self, 'image_paths') or len(self.image_paths) < 2:
            QMessageBox.warning(self, "Warning", "Load at least 2 images first.")
            return
        images = [cv2.imread(path) for path in self.image_paths]
        if any(img is None for img in images):
            QMessageBox.error(self, "Error", "Failed to load some images.")
            return
        hists = [cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten() for img in images]
        for hist in hists:
            cv2.normalize(hist, hist)
        max_sim = -1
        max_pair = (0,1)
        for i in range(len(hists)):
            for j in range(i+1, len(hists)):
                sim = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (i,j)
        self.similar1_label.setPixmap(self.convert_cv_to_pixmap(images[max_pair[0]]))
        self.similar2_label.setPixmap(self.convert_cv_to_pixmap(images[max_pair[1]]))
        self.similarity_label.setText(f"Similarity score: {max_sim:.4f} (using histogram correlation)")

    def load_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video files (*.mp4 *.avi *.mov)")
        if not file:
            return
        self.video_cap = cv2.VideoCapture(file)
        if not self.video_cap.isOpened():
            QMessageBox.error(self, "Error", "Could not open video.")
            return

    def play_video(self):
        if not hasattr(self, 'video_cap') or not self.video_cap.isOpened():
            QMessageBox.warning(self, "Warning", "Load a video first.")
            return
        self.video_thread = VideoThread(self.video_cap, self.video_display)
        self.video_thread.frame_signal.connect(self.update_video_display)
        self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def update_video_display(self, frame):
        pix = self.convert_cv_to_pixmap(frame, scale=500)
        self.video_display.setPixmap(pix)

    def load_bg_sub_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video files (*.mp4 *.avi *.mov)")
        if not file:
            return
        self.bg_sub_cap = cv2.VideoCapture(file)
        if not self.bg_sub_cap.isOpened():
            QMessageBox.error(self, "Error", "Could not open video.")
            return
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)

    def start_bg_subtraction(self):
        if not hasattr(self, 'bg_sub_cap') or not self.bg_sub_cap.isOpened():
            QMessageBox.warning(self, "Warning", "Load a video first.")
            return
        self.bg_sub_thread = BGSubThread(self.bg_sub_cap, self.bg_subtractor)
        self.bg_sub_thread.frame_signal.connect(self.update_bg_sub_display)
        self.bg_sub_thread.start()

    @pyqtSlot(np.ndarray)
    def update_bg_sub_display(self, fgmask):
        pix = self.convert_cv_to_pixmap(fgmask, scale=500)
        self.bg_sub_display.setPixmap(pix)

    def load_bg_replace_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video files (*.mp4 *.avi *.mov)")
        if not file:
            return
        self.bg_replace_cap = cv2.VideoCapture(file)
        if not self.bg_replace_cap.isOpened():
            QMessageBox.error(self, "Error", "Could not open video.")
            return
        self.bg_replace_subtractor = cv2.createBackgroundSubtractorMOG2()

    def load_bg_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Image files (*.jpg *.png *.jpeg)")
        if not file:
            return
        self.bg_image = cv2.imread(file)
        if self.bg_image is None:
            QMessageBox.error(self, "Error", "Could not load background image.")
        else:
            QMessageBox.information(self, "Success", "Background image loaded.")

    def start_bg_replacement(self):
        if not hasattr(self, 'bg_replace_cap') or not self.bg_replace_cap.isOpened():
            QMessageBox.warning(self, "Warning", "Load a video first.")
            return
        if not hasattr(self, 'bg_image') or self.bg_image is None:
            QMessageBox.warning(self, "Warning", "Load a background image first.")
            return
        self.bg_replace_thread = BGReplaceThread(self.bg_replace_cap, self.bg_replace_subtractor, self.bg_image)
        self.bg_replace_thread.frame_signal.connect(self.update_bg_replace_display)
        self.bg_replace_thread.start()

    @pyqtSlot(np.ndarray)
    def update_bg_replace_display(self, frame):
        pix = self.convert_cv_to_pixmap(frame, scale=500)
        self.bg_replace_display.setPixmap(pix)

    def update_blur_strength(self):
        self.blur_strength = self.blur_slider.value()

    @pyqtSlot(np.ndarray)
    def update_blur_display(self, frame):
        pix = self.convert_cv_to_pixmap(frame, scale=500)
        self.blur_display.setPixmap(pix)

    def load_blur_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video files (*.mp4 *.avi *.mov)")
        if not file:
            return
        self.blur_cap = cv2.VideoCapture(file)
        if not self.blur_cap.isOpened():
            QMessageBox.error(self, "Error", "Could not open video.")
            return
        self.blur_subtractor = cv2.createBackgroundSubtractorMOG2()

    def start_motion_blur(self):
        if not hasattr(self, 'blur_cap') or not self.blur_cap.isOpened():
            QMessageBox.warning(self, "Warning", "Load a video first.")
            return
        self.blur_thread = MotionBlurThread(self.blur_cap, self.blur_subtractor, self.blur_strength)
        self.blur_thread.frame_signal.connect(self.update_blur_display)
        self.blur_thread.start()

    def load_flow_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video files (*.mp4 *.avi *.mov)")
        if not file:
            return
        self.flow_cap = cv2.VideoCapture(file)
        if not self.flow_cap.isOpened():
            QMessageBox.error(self, "Error", "Could not open video.")
            return

    def start_optical_flow(self):
        if not hasattr(self, 'flow_cap') or not self.flow_cap.isOpened():
            QMessageBox.warning(self, "Warning", "Load a video first.")
            return
        self.flow_thread = OpticalFlowThread(self.flow_cap)
        self.flow_thread.frame_signal.connect(self.update_flow_display)
        self.flow_thread.start()

    @pyqtSlot(np.ndarray)
    def update_flow_display(self, frame):
        pix = self.convert_cv_to_pixmap(frame, scale=500)
        self.flow_display.setPixmap(pix)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
