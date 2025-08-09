import cv2
import streamlit as st
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

from exp2 import detect_green_zones, detect_red_zones, detect_yellow_zones
from boundingbox import bounding_box_bet_frames
from color_analysis import get_color_densities  # new import for color analysis

# Set title
st.set_page_config(layout="wide")
st.title("🎥 Real-Time Video Playback with Color Monitoring & Alerts")

# Upload video
video_file = st.file_uploader("📁 Upload a video file", type=["mp4", "avi", "mov"])
start = st.button("▶️ Start Playback")

# Resize frame helper
def resize_frame(frame, height=240):
    h, w = frame.shape[:2]
    scale = height / h
    return cv2.resize(frame, (int(w * scale), height))

# Detect sudden color jump
def detect_color_jump(curr, prev, threshold=0.25):
    alert_colors = []
    if prev:
        for color in curr:
            if abs(curr[color] - prev[color]) > threshold:
                alert_colors.append(color)
    return alert_colors

if start:
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap1 = cv2.VideoCapture(tfile.name)
        cap2 = cv2.VideoCapture(tfile.name)
        cap3 = cv2.VideoCapture(tfile.name)
        cap4 = cv2.VideoCapture(tfile.name)

        # Columns: 1 left, 2 center, 3 right, 4 color analysis
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        placeholder1 = col1.empty()
        placeholder2 = col2.empty()
        placeholder3 = col3.empty()
        placeholder4 = col4.empty()

        plot_placeholder_left = col1.empty()
        plot_placeholder_right = col3.empty()
        plot_placeholder_color = col4.empty()
        alert_placeholder = col4.empty()  # for jump alert

        frame_count = 0
        x_vals_left, y_vals_left = [], []
        x_vals_right, y_vals_right = [], []
        x_vals_color, red_vals, yellow_vals, green_vals, blue_vals = [], [], [], [], []
        color_data = []

        prev_frame1, prev_frame3 = None, None
        prev_intensity = None

        fps = cap1.get(cv2.CAP_PROP_FPS) or 30

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()

            if not (ret1 and ret2 and ret3 and ret4):
                break

            # Window 1: Red detection
            frame1, dead_pixel_for_frame1 = detect_red_zones(frame1)
            if isinstance(prev_frame1, np.ndarray):
                bb_frame = bounding_box_bet_frames(prev_frame1, frame1)
                if isinstance(bb_frame, np.ndarray):
                    frame1 = bb_frame
            prev_frame1 = frame1

            # Window 3: Yellow detection
            frame3, dead_pixel_for_frame3 = detect_yellow_zones(frame3)
            if isinstance(prev_frame3, np.ndarray):
                bb_frame = bounding_box_bet_frames(prev_frame3, frame3)
                if isinstance(bb_frame, np.ndarray):
                    frame3 = bb_frame
            prev_frame3 = frame3

            frame_count += 1
            timestamp = str(timedelta(seconds=frame_count / fps)).split(".")[0]

            # Graph data
            x_vals_left.append(frame_count)
            y_vals_left.append(dead_pixel_for_frame1)
            x_vals_right.append(frame_count)
            y_vals_right.append(dead_pixel_for_frame3)

            # Window 4: Color intensity
            r, y, g, b = get_color_densities(frame4)
            x_vals_color.append(frame_count)
            red_vals.append(r)
            yellow_vals.append(y)
            green_vals.append(g)
            blue_vals.append(b)

            curr_intensity = {"red": r, "yellow": y, "green": g, "blue": b}
            color_data.append({
                "Frame": frame_count,
                "Timestamp": timestamp,
                **curr_intensity
            })

            jump_alerts = detect_color_jump(curr_intensity, prev_intensity)
            prev_intensity = curr_intensity

            # Convert BGR to RGB for display
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            frame4_disp = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)

            # Resize for layout
            f1_resized = resize_frame(frame1)
            f2_large = resize_frame(frame2, height=480)
            f2_large = cv2.resize(f2_large, (f2_large.shape[1]*2, f2_large.shape[0]*2))
            f3_resized = resize_frame(frame3)
            f4_resized = resize_frame(frame4_disp)

            # Canvas center (Window 2)
            canvas = np.zeros((max(f1_resized.shape[0], f2_large.shape[0], f3_resized.shape[0]),
                               f2_large.shape[1], 3), dtype=np.uint8)
            y_off = (canvas.shape[0] - f2_large.shape[0]) // 2
            canvas[y_off:y_off+f2_large.shape[0], :f2_large.shape[1], :] = f2_large

            # Display windows
            placeholder1.image(f1_resized, caption="Window 1 - Red Zones")
            placeholder2.image(canvas, caption="Window 2 - Center View")
            placeholder3.image(f3_resized, caption="Window 3 - Yellow Zones")
            placeholder4.image(f4_resized, caption="Window 4 - Color Intensity Frame")

            # Left Graph
            fig_left, ax_left = plt.subplots(figsize=(4, 2.5))
            ax_left.plot(x_vals_left, y_vals_left, color='red')
            ax_left.set_title("Left Graph - Red Intensity")
            ax_left.set_xlabel("Frame")
            ax_left.set_ylabel("Intensity")
            ax_left.grid(True)
            plot_placeholder_left.pyplot(fig_left)
            plt.close(fig_left)

            # Right Graph
            fig_right, ax_right = plt.subplots(figsize=(4, 2.5))
            ax_right.plot(x_vals_right, y_vals_right, color='gold')
            ax_right.set_title("Right Graph - Yellow Intensity")
            ax_right.set_xlabel("Frame")
            ax_right.set_ylabel("Intensity")
            ax_right.grid(True)
            plot_placeholder_right.pyplot(fig_right)
            plt.close(fig_right)

            # Color Intensity Graph (Window 4)
            fig_color, ax_color = plt.subplots(figsize=(4, 3))
            ax_color.plot(x_vals_color, red_vals, color='red', label='Red')
            ax_color.plot(x_vals_color, yellow_vals, color='yellow', label='Yellow')
            ax_color.plot(x_vals_color, green_vals, color='green', label='Green')
            ax_color.plot(x_vals_color, blue_vals, color='blue', label='Blue')
            ax_color.set_title("Window 4 - Color Intensity Over Time")
            ax_color.set_xlabel("Frame")
            ax_color.set_ylabel("Normalized Intensity")
            ax_color.legend()
            ax_color.grid(True)
            plot_placeholder_color.pyplot(fig_color)
            plt.close(fig_color)

            # Jump Alert (Window 4)
            if jump_alerts:
                alert_msg = (
                    f"🚨 <b>ALERT</b>: Sudden spike in {', '.join(jump_alerts).upper()} "
                    f"(Frame {frame_count}, Time {timestamp})"
                )
                alert_placeholder.markdown(
                    f"<div style='color:red; font-weight:bold;'>{alert_msg}</div>",
                    unsafe_allow_html=True
                )
            else:
                alert_placeholder.markdown("")

            time.sleep(1 / 120)

        # Save intensity to CSV
        os.makedirs("output", exist_ok=True)
        out_path = f"output/color_intensity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(color_data).to_csv(out_path, index=False)
        st.success(f"✅ Color intensity data saved to `{out_path}`")

        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()
    else:
        st.warning("Please upload a video file first.")
else:
    st.info("Upload a video and click 'Start Playback'.")
