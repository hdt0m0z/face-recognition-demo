import streamlit as st
import cv2
import torch
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import io

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
# Điều này cho phép dữ liệu được cập nhật trong suốt phiên làm việc
if 'initialized' not in st.session_state:
    st.session_state.known_resnetv1_embeddings, st.session_state.known_resnetv1_names = load_known_face_data_from_file("facenet")
    st.session_state.known_arcface_embeddings, st.session_state.known_arcface_names = load_known_face_data_from_file("arcface")
    st.session_state.initialized = True
    st.sidebar.success("Tất cả các model và dữ liệu đã được tải.")

# --- GIAO DIỆN THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("Computer Vision Project")
    st.header("So sánh FaceNet và ArcFace")
    st.write("**Sinh viên thực hiện:** Thái")
    st.info("Chọn các tab bên dưới để xem chi tiết.")
    
    st.subheader("Số người trong CSDL (phiên hiện tại):")
    st.write(f"**{len(np.unique(st.session_state.known_resnetv1_names))}** người")


# --- NỘI DUNG CHÍNH ---
st.title("Phân tích và So sánh các mô hình Nhận dạng Khuôn mặt")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 So sánh Tổng quan", 
    "🖼️ Demo Nhận diện qua Ảnh",
    "➕ Thêm Dữ liệu Nhận dạng",
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
            # Dùng dữ liệu từ session_state để nhận diện
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None:
                for box in boxes:
                    face_tensor = mtcnn.extract(frame_rgb, [box], save_path=None).to(device)
                    embedding = resnetv1(face_tensor).detach().cpu().numpy()[0]
                    name, sim = recognize_face(embedding, st.session_state.known_resnetv1_embeddings, st.session_state.known_resnetv1_names)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f'ResnetV1: {name} ({sim:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            faces = arcface_app.get(frame_bgr)
            if len(faces) > 0:
                for face in faces:
                    arc_embedding = face.embedding
                    name, sim = recognize_face(arc_embedding, st.session_state.known_arcface_embeddings, st.session_state.known_arcface_names)
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_bgr, f'ArcFace: {name} ({sim:.2f})', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        with col2:
            st.image(frame_bgr, caption='Kết quả Nhận diện', channels="BGR", use_column_width=True)

# --- TAB 3: THÊM DỮ LIỆU NHẬN DẠNG ---
with tab3:
    st.header("Thêm dữ liệu người dùng mới")
    st.info("Tải lên ảnh của một người để thêm vào cơ sở dữ liệu nhận dạng cho phiên làm việc hiện tại.")

    person_name = st.text_input("Nhập tên người cần thêm (không dấu, không khoảng cách):")
    uploaded_images = st.file_uploader(
        "Tải lên một hoặc nhiều ảnh (rõ mặt)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Xử lý và Thêm vào CSDL", key="add_face"):
        if person_name and uploaded_images:
            with st.spinner(f"Đang xử lý ảnh cho '{person_name}'..."):
                for uploaded_file in uploaded_images:
                    image = Image.open(uploaded_file).convert('RGB')
                    frame_rgb = np.array(image)
                    
                    # Trích xuất embedding cho cả 2 model
                    boxes, _ = mtcnn.detect(frame_rgb)
                    if boxes is not None:
                        # ResnetV1
                        face_tensor = mtcnn.extract(frame_rgb, [boxes[0]], save_path=None).to(device)
                        resnetv1_emb = resnetv1(face_tensor).detach().cpu().numpy()[0]
                        st.session_state.known_resnetv1_embeddings.append(resnetv1_emb)
                        st.session_state.known_resnetv1_names.append(person_name)
                        
                        # ArcFace
                        faces_arc = arcface_app.get(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        if len(faces_arc) > 0:
                            arc_emb = faces_arc[0].embedding
                            st.session_state.known_arcface_embeddings.append(arc_emb)
                            st.session_state.known_arcface_names.append(person_name)
                            st.write(f"✅ Đã xử lý thành công ảnh: {uploaded_file.name}")
                        else:
                            st.warning(f"⚠️ ArcFace không tìm thấy mặt trong ảnh: {uploaded_file.name}")
                            st.session_state.known_resnetv1_embeddings.pop()
                            st.session_state.known_resnetv1_names.pop()
                    else:
                        st.warning(f"⚠️ FaceNet không tìm thấy mặt trong ảnh: {uploaded_file.name}")

            st.success(f"Hoàn tất! Đã thêm '{person_name}' vào CSDL của phiên này. Kiểm tra lại số người ở thanh bên.")
        else:
            st.error("Vui lòng nhập tên và tải lên ít nhất một ảnh.")

    st.divider()
    st.subheader("Lưu lại dữ liệu vĩnh viễn")
    st.warning("""
    **Lưu ý:** Dữ liệu bạn vừa thêm chỉ tồn tại trong phiên làm việc này.
    Để lưu vĩnh viễn, hãy nhấn các nút bên dưới để tải các file dữ liệu (.npy) đã cập nhật về máy, sau đó cam kết và đẩy chúng lên lại repository GitHub của bạn.
    """)

    col1, col2 = st.columns(2)
    with col1:
        # Chuyển list thành numpy array rồi thành bytes
        out_resnet_emb = io.BytesIO()
        np.save(out_resnet_emb, np.array(st.session_state.known_resnetv1_embeddings))
        st.download_button(
            label="Tải về `known_facenet_embeddings.npy`",
            data=out_resnet_emb.getvalue(),
            file_name="known_facenet_embeddings.npy"
        )
        out_arcface_emb = io.BytesIO()
        np.save(out_arcface_emb, np.array(st.session_state.known_arcface_embeddings))
        st.download_button(
            label="Tải về `known_arcface_embeddings.npy`",
            data=out_arcface_emb.getvalue(),
            file_name="known_arcface_embeddings.npy"
        )
    with col2:
        out_resnet_names = io.BytesIO()
        np.save(out_resnet_names, np.array(st.session_state.known_resnetv1_names))
        st.download_button(
            label="Tải về `known_facenet_names.npy`",
            data=out_resnet_names.getvalue(),
            file_name="known_facenet_names.npy"
        )
        out_arcface_names = io.BytesIO()
        np.save(out_arcface_names, np.array(st.session_state.known_arcface_names))
        st.download_button(
            label="Tải về `known_arcface_names.npy`",
            data=out_arcface_names.getvalue(),
            file_name="known_arcface_names.npy"
        )
# --- TAB 4: GIỚI THIỆU DỰ ÁN ---
with tab4:
    st.header("Mục tiêu và Phương pháp")
    st.markdown("""
    Dự án này được thực hiện trong khuôn khổ môn học Thị giác máy tính, nhằm mục đích so sánh hai mô hình nhận dạng khuôn mặt tiên tiến: **FaceNet (sử dụng kiến trúc InceptionResnetV1)** và **ArcFace (sử dụng mô hình buffalo_l)**.

    ### Phương pháp so sánh
    Chúng tôi đánh giá hai mô hình dựa trên các tiêu chí sau:
    1.  **Hiệu suất trên các bộ dữ liệu benchmark chuẩn.**
    2.  **Đặc điểm kiến trúc (Số tham số, FLOPs).**
    3.  **Hiệu suất trong thời gian thực (FPS, Độ ổn định).**
    4.  **Thử nghiệm định tính qua ảnh người dùng tải lên.**
    
    Toàn bộ ứng dụng demo này được xây dựng bằng **Streamlit**.
    """)
