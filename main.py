import streamlit as st
import os
import uuid
# from ultralytics import YOLO
import glob
import sys

st.write("Python:", sys.version)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
st.set_page_config(page_title="Number Plate Detection", layout="centered")

st.title("🪖 Number Plate Detection")
st.write("Upload a video to detect Number Plates using YOLO")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
    # ------------------ SAVE INPUT VIDEO ------------------
    upload_dir = "uploads/videos"
    os.makedirs(upload_dir, exist_ok=True)

    video_name = f"{uuid.uuid4()}_{uploaded_video.name}"
    input_video_path = os.path.join(upload_dir, video_name)

    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.subheader("📥 Uploaded Video")
    st.video(input_video_path)


    # ------------------ BUTTON TO RUN DETECTION ------------------
    if st.button("Run Number Plate Detection"):
        st.subheader("🔍 Detecting Number Plates...")
        with st.spinner("Processing video... Please wait"):

            from ultralytics import YOLO
            model = YOLO("best.pt")
            st.success("Model loaded successfully ✅")

            results = model.predict(
                source=input_video_path,
                save=True,
                conf=0.4,
                project="runs",
                name="number_plate_detect",
                exist_ok=True
            )

        # ------------------ FIND OUTPUT VIDEO ------------------
        output_dir = results[0].save_dir  # YOLO output folder

        # YOLO usually saves as .avi or .mp4
        output_videos = glob.glob(os.path.join(output_dir, "*.avi")) + \
                        glob.glob(os.path.join(output_dir, "*.mp4"))

        if output_videos:
            output_video_path = output_videos[0]

            st.success("✅ Helmet detection completed successfully!")

            # ------------------ DOWNLOAD BUTTON ------------------
            with open(output_video_path, "rb") as file:
                video_bytes = file.read()

            st.download_button(
                label="⬇️ Download Detected Video",
                data=video_bytes,
                file_name=os.path.basename(output_video_path),
                mime="video/mp4"
            )

            st.info("You can now download the processed video. Preview is disabled to avoid browser issues.")

        else:
            st.error("Detection failed: Output video not found")
