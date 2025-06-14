import streamlit as st
import cv2
import torch
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="So sánh Face Recognition Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TẢI MODEL VÀ DỮ LIỆU (CACHE ĐỂ TĂNG TỐC) ---
@st.cache_resource
def load_all_models():
    """Tải tất cả các model AI một lần duy nhất."""
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from insightface.app import FaceAnalysis

    print("Đang tải models... (Chỉ chạy một lần)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mtcnn_model = MTCNN(keep_all=True, device=device)
    resnetv1_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    arcface_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'])
    arcface_model.prepare(ctx_id=0 if device.type == 'cuda' else -1)
    
    print("Tải models thành công.")
    return mtcnn_model, resnetv1_model, arcface_model, device

@st.cache_data
def load_known_face_data_from_file(model_name):
    """Tải dữ liệu khuôn mặt đã lưu từ các file .npy ban đầu."""
    print(f"Đang tải dữ liệu gốc cho {model_name}...")
    emb_file = f'known_{model_name}_embeddings.npy'
    name_file = f'known_{model_name}_names.npy'
    if os.path.exists(emb_file) and os.path.exists(name_file):
        embeddings = list(np.load(emb_file, allow_pickle=True))
        names = list(np.load(name_file, allow_pickle=True))
        return embeddings, names
    return [], []

def recognize_face(embedding, known_embeddings, known_names, threshold=0.6):
    if len(known_embeddings) == 0:
        return "Unknown", 0.0
    similarities = cosine_similarity([embedding], known_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return (known_names[max_idx], max_sim) if max_sim >= threshold else ("Unknown", max_sim)

# Tải tài nguyên
mtcnn, resnetv1, arcface_app, device = load_all_models()

# --- SỬ DỤNG SESSION STATE ĐỂ LƯU TRỮ DỮ LIỆU ---
if 'initialized' not in st.session_state:
    st.session_state.known_resnetv1_embeddings, st.session_state.known_resnetv1_names = load_known_face_data_from_file("facenet")
    st.session_state.known_arcface_embeddings, st.session_state.known_arcface_names = load_known_face_data_from_file("arcface")
    st.session_state.initialized = True
    st.sidebar.success("Tất cả các model và dữ liệu đã được tải.")

# --- LỚP XỬ LÝ VIDEO THỜI GIAN THỰC CHO STREAMLIT-WEBRTC ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold = 0.6

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Xử lý với ResnetV1
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            for box in boxes:
                face_tensor = mtcnn.extract(img_rgb, [box], save_path=None).to(device)
                embedding = resnetv1(face_tensor).detach().cpu().numpy()[0]
                name, sim = recognize_face(embedding, st.session_state.known_resnetv1_embeddings, st.session_state.known_resnetv1_names, self.threshold)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'R: {name} ({sim:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Xử lý với ArcFace
        faces = arcface_app.get(img)
        if len(faces) > 0:
            for face in faces:
                arc_embedding = face.embedding
                name, sim = recognize_face(arc_embedding, st.session_state.known_arcface_embeddings, st.session_state.known_arcface_names, self.threshold)
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f'A: {name} ({sim:.2f})', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return img

# --- GIAO DIỆN THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("Computer Vision Project")
    st.header("So sánh FaceNet và ArcFace")
    st.write("**Sinh viên thực hiện:** Thái")
    st.info("Chọn các tab bên dưới để xem chi tiết.")
    
    st.subheader("Số người trong CSDL:")
    st.write(f"**{len(np.unique(st.session_state.known_resnetv1_names))}** người")

# --- NỘI DUNG CHÍNH ---
st.title("Nhận diện khuôn mặt: So sánh ResNetV1 & ArcFace")

tab1, tab2, tab3 = st.tabs([
    "So sánh Tổng quan", 
    "Nhận diện thời gian thực",
    "Giới thiệu Dự án"
])

# --- TAB 1: SO SÁNH TỔNG QUAN ---
with tab1:
    st.header("So sánh Hiệu suất và Kiến trúc")
    # (Giữ nguyên nội dung tab 1)
    st.subheader("1. Bảng so sánh Kiến trúc")
    try:
        df_arch = pd.read_csv("architecture_comparison.csv")
        st.dataframe(df_arch.style.format(precision=2), use_container_width=True)
    except FileNotFoundError:
        st.error("Không tìm thấy file `architecture_comparison.csv`.")
    st.subheader("2. Bảng so sánh trên Benchmark")
    try:
        df_benchmark = pd.read_csv("benchmark_comparison.csv")
        st.dataframe(df_benchmark.style.format(precision=2), use_container_width=True)
    except FileNotFoundError:
        st.error("Không tìm thấy file `benchmark_comparison.csv`.")
    st.divider()
    st.subheader("3. Biểu đồ So sánh Tổng hợp")
    if os.path.exists("comparison_charts_full.png"):
        st.image("comparison_charts_full.png", caption="Biểu đồ so sánh hiệu suất, kiến trúc và benchmark.")
    else:
        st.warning("Không tìm thấy file 'comparison_charts_full.png'.")

# --- TAB 2: NHẬN DIỆN THỜI GIAN THỰC & QUẢN LÝ DỮ LIỆU ---
with tab2:
    st.header("Nhận diện thời gian thực từ webcam")
    st.info("Nhấn nút 'START' để bật webcam và xem kết quả nhận diện. Nhấn 'STOP' để dừng.")
    
    webrtc_streamer(
        key="realtime-recognition",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Thêm người mới")
        add_name = st.text_input("Nhập tên:", key="add_name")
        add_uploaded_img = st.file_uploader("Tải ảnh khuôn mặt", type=["jpg", "jpeg", "png"], key="add_img")
        
        if st.button("➕ Thêm"):
            if add_uploaded_img and add_name.strip():
                with st.spinner(f"Đang thêm {add_name}..."):
                    img = Image.open(add_uploaded_img).convert("RGB")
                    img_array = np.array(img)
                    
                    boxes, _ = mtcnn.detect(img_array)
                    if boxes is not None:
                        face_tensor = mtcnn.extract(img_array, [boxes[0]], save_path=None).to(device)
                        resnetv1_emb = resnetv1(face_tensor).detach().cpu().numpy()[0]
                        st.session_state.known_resnetv1_embeddings.append(resnetv1_emb)
                        st.session_state.known_resnetv1_names.append(add_name.strip())

                        faces = arcface_app.get(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                        if len(faces) > 0:
                            arc_emb = faces[0].embedding
                            st.session_state.known_arcface_embeddings.append(arc_emb)
                            st.session_state.known_arcface_names.append(add_name.strip())
                            
                            # Lưu file lên đĩa (trên server)
                            np.save('known_facenet_embeddings.npy', np.array(st.session_state.known_resnetv1_embeddings))
                            np.save('known_facenet_names.npy', np.array(st.session_state.known_resnetv1_names))
                            np.save('known_arcface_embeddings.npy', np.array(st.session_state.known_arcface_embeddings))
                            np.save('known_arcface_names.npy', np.array(st.session_state.known_arcface_names))
                            
                            st.success(f"Đã thêm '{add_name.strip()}'.")
                            st.rerun()
                        else:
                            st.warning("ArcFace không phát hiện được khuôn mặt.")
                            st.session_state.known_resnetv1_embeddings.pop()
                            st.session_state.known_resnetv1_names.pop()
                    else:
                        st.warning("Không phát hiện khuôn mặt.")
            else:
                st.warning("Vui lòng tải ảnh và nhập tên!")

    with col2:
        st.subheader("Xoá người đã đăng ký")
        if len(st.session_state.known_resnetv1_names) > 0:
            unique_names = sorted(list(np.unique(st.session_state.known_resnetv1_names)))
            to_delete = st.selectbox("Chọn tên để xoá", unique_names, key="delete_name", index=None, placeholder="Chọn một tên...")
            
            if st.button("❌ Xoá") and to_delete:
                new_resnet_emb, new_resnet_names = [], []
                new_arcface_emb, new_arcface_names = [], []
                
                for i, name in enumerate(st.session_state.known_resnetv1_names):
                    if name != to_delete:
                        new_resnet_emb.append(st.session_state.known_resnetv1_embeddings[i])
                        new_resnet_names.append(name)
                        new_arcface_emb.append(st.session_state.known_arcface_embeddings[i])
                        new_arcface_names.append(name)
                
                st.session_state.known_resnetv1_embeddings = new_resnet_emb
                st.session_state.known_resnetv1_names = new_resnet_names
                st.session_state.known_arcface_embeddings = new_arcface_emb
                st.session_state.known_arcface_names = new_arcface_names
                
                np.save('known_facenet_embeddings.npy', np.array(new_resnet_emb))
                np.save('known_facenet_names.npy', np.array(new_resnet_names))
                np.save('known_arcface_embeddings.npy', np.array(new_arcface_emb))
                np.save('known_arcface_names.npy', np.array(new_arcface_names))
                
                st.success(f"Đã xoá tất cả ảnh của '{to_delete}'.")
                st.rerun()
        else:
            st.write("Chưa có ai trong cơ sở dữ liệu.")

# --- TAB 3: GIỚI THIỆU DỰ ÁN ---
with tab3:
    st.header("Mục tiêu và Phương pháp")
    # (Giữ nguyên nội dung tab 3)
    st.markdown("""
    Dự án này được thực hiện trong khuôn khổ môn học Thị giác máy tính, nhằm mục đích so sánh hai mô hình nhận dạng khuôn mặt tiên tiến: **FaceNet (sử dụng kiến trúc InceptionResnetV1)** và **ArcFace (sử dụng mô hình buffalo_l)**.

    ### Phương pháp so sánh
    Chúng tôi đánh giá hai mô hình dựa trên các tiêu chí sau:
    1.  **Hiệu suất trên các bộ dữ liệu benchmark chuẩn.**
    2.  **Đặc điểm kiến trúc (Số tham số, FLOPs).**
    3.  **Hiệu suất trong thời gian thực (FPS, Độ ổn định).**
    4.  **Thử nghiệm định tính qua ảnh người dùng tải lên và webcam.**
    
    Toàn bộ ứng dụng demo này được xây dựng bằng **Streamlit**.
    """)
