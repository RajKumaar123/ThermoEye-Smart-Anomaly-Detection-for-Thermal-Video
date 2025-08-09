import cv2
import streamlit as st
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
from exp2 import detect_green_zones, detect_red_zones, detect_yellow_zones
from boundingbox import bounding_box_bet_frames

st.title("Video Playback with Graphs Below Both Small Videos")

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
start = st.button("Start Playback")

def resize_frame(frame, height=240):  # fixed height, scale width accordingly
    h, w = frame.shape[:2]
    scale = height / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, height))
    return resized

if start:
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap1 = cv2.VideoCapture(tfile.name)
        cap2 = cv2.VideoCapture(tfile.name)
        cap3 = cv2.VideoCapture(tfile.name)

        col1, col2, col3 = st.columns([1, 2, 1])

        # Placeholders for videos
        placeholder1 = col1.empty()
        placeholder2 = col2.empty()
        placeholder3 = col3.empty()

        # Placeholders for plots below left and right videos
        plot_placeholder_left = col1.empty()
        plot_placeholder_right = col3.empty()

        frame_count = 0
        x_vals_left = []
        y_vals_left = []

        x_vals_right = []
        y_vals_right = []
        prev_frame1 = None
        prev_frame3 = None

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            frame1, dead_pixel_for_frame1 = detect_red_zones(frame1)
            frame3, dead_pixel_for_frame3 = detect_yellow_zones(frame3)
            #print(dead_pixel_for_frame1, dead_pixel_for_frame3)
            if(isinstance(prev_frame1, np.ndarray)):
                temp = bounding_box_bet_frames(prev_frame1, frame1)
                if isinstance(temp, np.ndarray):
                    frame1 = temp
            if(isinstance(prev_frame3, np.ndarray)):
                temp = bounding_box_bet_frames(prev_frame3, frame3)
                if isinstance(temp, np.ndarray):
                    frame3 = temp


            prev_frame1 = frame1
            prev_frame3 = frame3


            if not (ret1 and ret2 and ret3):
                break

            frame_count += 1

            # Append some example data for plots (can be replaced with real data)
            x_vals_left.append(frame_count)
            #y_vals_left.append(frame_count % 30)
            y_vals_left.append(dead_pixel_for_frame1)



            x_vals_right.append(frame_count)
            #y_vals_right.append((frame_count * 2) % 30)
            y_vals_right.append(dead_pixel_for_frame3)
            

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

            frame1_resized = resize_frame(frame1, height=240)
            frame3_resized = resize_frame(frame3, height=240)

            frame2_large = resize_frame(frame2, height=480)
            frame2_large = cv2.resize(frame2_large, (frame2_large.shape[1]*2, frame2_large.shape[0]*2))

            max_height = max(frame1_resized.shape[0], frame2_large.shape[0], frame3_resized.shape[0])
            max_width = frame2_large.shape[1]
            canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            y_offset = (max_height - frame2_large.shape[0]) // 2
            canvas[y_offset:y_offset+frame2_large.shape[0], :frame2_large.shape[1], :] = frame2_large

            placeholder1.image(frame1_resized, caption="Window 1")
            placeholder2.image(canvas, caption="Window 2 (Large Center)")
            placeholder3.image(frame3_resized, caption="Window 3")

            dpi = 96
            # Left plot size matches left video
            height_px_left = frame1_resized.shape[0]
            width_px_left = frame1_resized.shape[1]
            fig_width_inch_left = width_px_left / dpi
            fig_height_inch_left = height_px_left / dpi

            fig_left, ax_left = plt.subplots(figsize=(fig_width_inch_left, fig_height_inch_left), dpi=dpi)
            ax_left.plot(x_vals_left, y_vals_left, color='blue')
            ax_left.set_title("Left Video Plot")
            ax_left.set_xlabel("Frame")
            ax_left.set_ylabel("Value")
            ax_left.grid(True)
            fig_left.tight_layout()
            plot_placeholder_left.pyplot(fig_left)
            plt.close(fig_left)

            # Right plot size matches right video
            height_px_right = frame3_resized.shape[0]
            width_px_right = frame3_resized.shape[1]
            fig_width_inch_right = width_px_right / dpi
            fig_height_inch_right = height_px_right / dpi

            fig_right, ax_right = plt.subplots(figsize=(fig_width_inch_right, fig_height_inch_right), dpi=dpi)
            ax_right.plot(x_vals_right, y_vals_right, color='green')
            ax_right.set_title("Right Video Plot")
            ax_right.set_xlabel("Frame")
            ax_right.set_ylabel("Value")
            ax_right.grid(True)
            fig_right.tight_layout()
            plot_placeholder_right.pyplot(fig_right)
            plt.close(fig_right)

            time.sleep(1/120)

        cap1.release()
        cap2.release()
        cap3.release()

    else:
        st.warning("Please upload a video file first.")
else:
    st.info("Upload a video and click 'Start Playback'.")
