import streamlit as st
import cv2
import torch
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="So sánh Face Recognition Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TẢI MODEL VÀ DỮ LIỆU (CACHE ĐỂ TĂNG TỐC) ---

# Sử dụng cache_resource cho các đối tượng không thể hash (models)
@st.cache_resource
def load_all_models():
    """Tải tất cả các model AI một lần duy nhất."""
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from insightface.app import FaceAnalysis

    print("Đang tải models... (Chỉ chạy một lần)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model 1: FaceNet (ResnetV1)
    mtcnn_model = MTCNN(keep_all=True, device=device)
    resnetv1_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Model 2: ArcFace
    arcface_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'])
    arcface_model.prepare(ctx_id=0 if device.type == 'cuda' else -1)
    
    print("Tải models thành công.")
    return mtcnn_model, resnetv1_model, arcface_model, device

# Sử dụng cache_data cho dữ liệu có thể hash (numpy arrays, lists)
@st.cache_data
def load_known_face_data(model_name):
    """Tải dữ liệu khuôn mặt đã lưu từ các file .npy."""
    print(f"Đang tải dữ liệu khuôn mặt cho {model_name}...")
    emb_file = f'known_{model_name}_embeddings.npy'
    name_file = f'known_{model_name}_names.npy'
    if os.path.exists(emb_file) and os.path.exists(name_file):
        embeddings = list(np.load(emb_file, allow_pickle=True))
        names = list(np.load(name_file, allow_pickle=True))
        return embeddings, names
    return [], []

# Hàm nhận diện (tái sử dụng từ cv.py)
def recognize_face(embedding, known_embeddings, known_names, threshold=0.6):
    if len(known_embeddings) == 0:
        return "Unknown", 0.0
    similarities = cosine_similarity([embedding], known_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return (known_names[max_idx], max_sim) if max_sim >= threshold else ("Unknown", max_sim)


# Tải tài nguyên
mtcnn, resnetv1, arcface_app, device = load_all_models()
known_resnetv1_embeddings, known_resnetv1_names = load_known_face_data("facenet")
known_arcface_embeddings, known_arcface_names = load_known_face_data("arcface")

st.sidebar.success("Tất cả các model và dữ liệu đã được tải.")

# --- GIAO DIỆN THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("Computer Vision Project")
    st.header("So sánh FaceNet và ArcFace")
    st.write("""
    **Sinh viên thực hiện:** Thái
    
    Ứng dụng này demo kết quả của dự án, so sánh hai mô hình nhận dạng khuôn mặt hàng đầu.
    """)
    st.info("Chọn các tab bên dưới để xem chi tiết.")

# --- NỘI DUNG CHÍNH ---
st.title("Phân tích và So sánh các mô hình Nhận dạng Khuôn mặt")

tab1, tab2, tab3 = st.tabs([
    "📊 So sánh Tổng quan", 
    "🖼️ Demo Nhận diện qua Ảnh",
    "📝 Giới thiệu Dự án"
])

# --- TAB 1: SO SÁNH TỔNG QUAN ---
with tab1:
    st.header("So sánh Hiệu suất và Kiến trúc")
    st.write("Phần này trình bày các kết quả so sánh định lượng giữa FaceNet (ResnetV1) và ArcFace (buffalo_l), được tạo ra bởi `cv.py`.")
    
    st.subheader("1. Bảng so sánh Kiến trúc")
    try:
        df_arch = pd.read_csv("architecture_comparison.csv")
        st.dataframe(df_arch.style.format(precision=2), use_container_width=True)
    except FileNotFoundError:
        st.error("Không tìm thấy file `architecture_comparison.csv`. Vui lòng chạy `cv.py` để tạo file.")

    st.subheader("2. Bảng so sánh trên Benchmark")
    try:
        df_benchmark = pd.read_csv("benchmark_comparison.csv")
        st.dataframe(df_benchmark.style.format(precision=2), use_container_width=True)
    except FileNotFoundError:
        st.error("Không tìm thấy file `benchmark_comparison.csv`. Vui lòng chạy `cv.py` để tạo file.")
            
    st.divider()
    
    st.subheader("3. Biểu đồ So sánh Tổng hợp")
    if os.path.exists("comparison_charts_full.png"):
        st.image("comparison_charts_full.png", caption="Biểu đồ so sánh hiệu suất, kiến trúc và benchmark.")
    else:
        st.warning("Không tìm thấy file 'comparison_charts_full.png'. Vui lòng chạy `cv.py` để tạo biểu đồ.")

# --- TAB 2: DEMO NHẬN DIỆN QUA ẢNH ---
with tab2:
    st.header("Thử nghiệm Nhận diện với ảnh của bạn")
    st.write("Tải lên một bức ảnh có chứa khuôn mặt để xem kết quả nhận diện từ cả hai mô hình.")

    uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        frame_rgb = np.array(image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Ảnh gốc', use_column_width=True)

        with st.spinner('Đang phân tích...'):
            # --- Xử lý với ResnetV1 ---
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None:
                for box in boxes:
                    face_tensor = mtcnn.extract(frame_rgb, [box], save_path=None).to(device)
                    embedding = resnetv1(face_tensor).detach().cpu().numpy()[0]
                    name, sim = recognize_face(embedding, known_resnetv1_embeddings, known_resnetv1_names)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f'ResnetV1: {name} ({sim:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Xử lý với ArcFace ---
            faces = arcface_app.get(frame_bgr)
            if len(faces) > 0:
                for face in faces:
                    arc_embedding = face.embedding
                    name, sim = recognize_face(arc_embedding, known_arcface_embeddings, known_arcface_names)
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_bgr, f'ArcFace: {name} ({sim:.2f})', (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        with col2:
            st.image(frame_bgr, caption='Kết quả Nhận diện', channels="BGR", use_column_width=True)


# --- TAB 3: GIỚI THIỆU DỰ ÁN ---
with tab3:
    st.header("Mục tiêu và Phương pháp")
    st.markdown("""
    Dự án này được thực hiện trong khuôn khổ môn học Thị giác máy tính, nhằm mục đích so sánh hai mô hình nhận dạng khuôn mặt tiên tiến: **FaceNet (sử dụng kiến trúc InceptionResnetV1)** và **ArcFace (sử dụng mô hình buffalo_l)**.

    ### Phương pháp so sánh
    Chúng tôi đánh giá hai mô hình dựa trên các tiêu chí sau:
    1.  **Hiệu suất trên các bộ dữ liệu benchmark chuẩn:** Đo lường độ chính xác trên LFW, CFP-FP, AgeDB-30, và IJB-C để có cái nhìn khách quan về hiệu năng của mỗi mô hình.
    2.  **Đặc điểm kiến trúc:** Phân tích các thông số như số lượng tham số và khối lượng tính toán (FLOPs) để đánh giá độ phức tạp và yêu cầu tài nguyên.
    3.  **Hiệu suất trong thời gian thực:** Thu thập và so sánh tốc độ xử lý (FPS) và độ ổn định của điểm tương đồng (similarity score) qua các lần nhận diện trực tiếp.
    4.  **Thử nghiệm định tính:** Cho phép người dùng tải lên ảnh để trực tiếp quan sát và so sánh kết quả nhận diện của hai mô hình trong các điều kiện khác nhau.
    
    Toàn bộ ứng dụng demo này được xây dựng bằng **Streamlit**, giúp trực quan hóa kết quả một cách hiệu quả.
    """)
