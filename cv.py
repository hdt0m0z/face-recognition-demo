import cv2
import torch
import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis
import pandas as pd
from thop import profile

# ========== CẤU HÌNH ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Thiết bị:", device)

# --- Model 1: FaceNet (ResnetV1) ---
print("Đang tải model FaceNet (ResnetV1)...")
mtcnn = MTCNN(keep_all=True, device=device)
resnetv1 = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Tải model FaceNet thành công.")

# --- Model 2: ArcFace ---
print("Đang tải model ArcFace...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if device.type == 'cuda' else -1)
print("Tải model ArcFace thành công.")


# ========== LOAD DỮ LIỆU ==========
def load_known_faces(model_name):
    """Tải embeddings và tên đã lưu cho một model cụ thể."""
    emb_file = f'known_{model_name}_embeddings.npy'
    name_file = f'known_{model_name}_names.npy'
    if os.path.exists(emb_file) and os.path.exists(name_file):
        embeddings = list(np.load(emb_file, allow_pickle=True))
        names = list(np.load(name_file, allow_pickle=True))
        return embeddings, names
    return [], []

known_resnetv1_embeddings, known_resnetv1_names = load_known_faces("facenet")
known_arcface_embeddings, known_arcface_names = load_known_faces("arcface")
print(f"Đã tải {len(known_resnetv1_names)} khuôn mặt cho FaceNet và {len(known_arcface_names)} khuôn mặt cho ArcFace.")


# ========== NHẬN DIỆN ==========
def recognize_face(embedding, known_embeddings, known_names, threshold=0.6):
    """So sánh embedding với CSDL đã biết để tìm ra người khớp nhất."""
    if len(known_embeddings) == 0:
        return "Unknown", 0.0
    similarities = cosine_similarity([embedding], known_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return (known_names[max_idx], max_sim) if max_sim >= threshold else ("Unknown", max_sim)


# ========== KHỞI TẠO LOG ĐỂ VẼ BIỂU ĐỒ ==========
resnetv1_log, arcface_log = [], []


# ========== MỞ WEBCAM ==========
cap = cv2.VideoCapture(0)
print("\nNhấn: 'C' bật/tắt nhận diện, 'N' thêm người, 'X' xóa người, 'Q' thoát")
realtime_mode = False
prev_time = time.time()

os.makedirs('saved_faces/full_frames', exist_ok=True)

try:
    # (Vòng lặp while và các logic nhận diện, nhận phím giữ nguyên như cũ)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if realtime_mode:
            # ----- Xử lý với ResnetV1 (màu xanh lá) -----
            start_rv1 = time.time()
            boxes, _ = mtcnn.detect(img_rgb)
            if boxes is not None:
                for box in boxes:
                    face_tensor = mtcnn.extract(img_rgb, [box], save_path=None).to(device)
                    embedding = resnetv1(face_tensor).detach().cpu().numpy()[0]
                    name, sim = recognize_face(embedding, known_resnetv1_embeddings, known_resnetv1_names)
                    
                    resnetv1_log.append((time.time() - start_rv1, sim))
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ResnetV1: {name} ({sim:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ----- Xử lý với ArcFace (màu xanh dương) -----
            start_arc = time.time()
            faces = app.get(frame)
            if len(faces) > 0:
                for face in faces:
                    arc_embedding = face.embedding
                    name, sim = recognize_face(arc_embedding, known_arcface_embeddings, known_arcface_names)

                    arcface_log.append((time.time() - start_arc, sim))

                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'ArcFace: {name} ({sim:.2f})', (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # ----- FPS -----
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("So sanh ResnetV1 & ArcFace", frame)
        key = cv2.waitKey(1)

        # ========== NHẬN PHÍM ==========
        if key == ord('q'):
            break
        elif key == ord('c'):
            realtime_mode = not realtime_mode
            print("Realtime:", "BẬT" if realtime_mode else "TẮT")
        elif key == ord('n'):
            print("\n== Thêm người mới (cho cả 2 model) ==")
            name = input("Tên người: ").strip()
            if name:
                img_to_add = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(img_to_add)
                if boxes is not None:
                    face_tensor = mtcnn.extract(img_to_add, [boxes[0]], save_path=None).to(device)
                    resnetv1_emb = resnetv1(face_tensor).detach().cpu().numpy()[0]
                    known_resnetv1_embeddings.append(resnetv1_emb)
                    known_resnetv1_names.append(name)
                    np.save('known_facenet_embeddings.npy', np.array(known_resnetv1_embeddings))
                    np.save('known_facenet_names.npy', np.array(known_resnetv1_names))
                    faces_arc = app.get(frame)
                    if len(faces_arc) > 0:
                        arc_emb = faces_arc[0].embedding
                        known_arcface_embeddings.append(arc_emb)
                        known_arcface_names.append(name)
                        np.save('known_arcface_embeddings.npy', np.array(known_arcface_embeddings))
                        np.save('known_arcface_names.npy', np.array(known_arcface_names))
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"saved_faces/full_frames/{name}_combined_{timestamp}.jpg", frame)
                        print(f"Đã thêm '{name}' vào cả 2 cơ sở dữ liệu.")
                    else:
                        print("ArcFace không phát hiện được khuôn mặt, không thể thêm.")
                        known_resnetv1_embeddings.pop()
                        known_resnetv1_names.pop()
                else:
                    print("FaceNet không phát hiện được khuôn mặt, không thể thêm.")
            else:
                print("Tên không hợp lệ!")
        elif key == ord('x'):
            print("\n== Xóa người (khỏi cả 2 model) ==")
            if not known_resnetv1_names:
                print("Chưa có ai trong danh sách để xóa.")
                continue
            for i, n in enumerate(known_resnetv1_names):
                print(f"{i+1}. {n}")
            try:
                idx_str = input("STT cần xóa (0 để hủy): ")
                idx = int(idx_str)
                if idx == 0:
                    continue
                if 1 <= idx <= len(known_resnetv1_names):
                    removed_name = known_resnetv1_names.pop(idx - 1)
                    known_resnetv1_embeddings.pop(idx - 1)
                    known_arcface_names.pop(idx - 1)
                    known_arcface_embeddings.pop(idx - 1)
                    np.save('known_facenet_embeddings.npy', np.array(known_resnetv1_embeddings))
                    np.save('known_facenet_names.npy', np.array(known_resnetv1_names))
                    np.save('known_arcface_embeddings.npy', np.array(known_arcface_embeddings))
                    np.save('known_arcface_names.npy', np.array(known_arcface_names))
                    print(f"Đã xóa '{removed_name}' khỏi cả 2 cơ sở dữ liệu.")
                else:
                    print("Lỗi: STT không hợp lệ!")
            except ValueError:
                print("Lỗi: Vui lòng nhập một số hợp lệ.")
            except Exception as e:
                print(f"Lỗi không xác định: {e}")

finally:
    # --- DỌN DẸP ---
    print("\n✅ Đang giải phóng tài nguyên và thoát...")
    cap.release()
    cv2.destroyAllWindows()

    # --- TẠO CÁC BẢNG SO SÁNH ---
    # Phân tích kiến trúc
    print("\nĐang phân tích kiến trúc các model...")
    dummy_input_resnet = torch.randn(1, 3, 160, 160).to(device)
    total_ops, total_params = profile(resnetv1, (dummy_input_resnet,), verbose=False)
    depth_resnet = len(list(filter(lambda p: p.requires_grad, resnetv1.parameters())))
    arch_data = {
        'Model': ['FaceNet (ResnetV1)', 'ArcFace (buffalo_l)'],
        'Độ sâu mạng (ước tính)': [depth_resnet, 100],
        'Số lượng tham số (Triệu)': [total_params / 1e6, 25.6],
        'FLOPs (GFLOPs)': [total_ops / 1e9, 5.2],
        'Kích thước đầu vào': ['160x160', "112x112"]
    }
    df_arch = pd.DataFrame(arch_data)
    df_arch.to_csv("architecture_comparison.csv", index=False)
    print(f"✅ Đã lưu bảng so sánh kiến trúc tại: architecture_comparison.csv")
    print(df_arch)
    
    # Dữ liệu benchmark
    print("\nĐang tổng hợp hiệu suất trên các bộ dữ liệu benchmark...")
    benchmark_data = {
        'Dataset': ['LFW', 'CFP-FP', 'AgeDB-30', 'IJB-C (TAR@FAR=1e-4)'],
        'FaceNet (ResnetV1) Accuracy (%)': [99.30, 95.50, 97.40, 90.8],
        'ArcFace (buffalo_l) Accuracy (%)': [99.80, 98.37, 98.28, 97.1]
    }
    df_benchmark = pd.DataFrame(benchmark_data)
    df_benchmark.to_csv("benchmark_comparison.csv", index=False)
    print(f"✅ Đã lưu bảng so sánh benchmark tại: benchmark_comparison.csv")
    print(df_benchmark)


    # --- VẼ BIỂU ĐỒ SO SÁNH TỔNG HỢP (TẤT CẢ TRONG 1) ---
    print("\nĐang tạo biểu đồ so sánh tổng hợp...")
    
    # Tạo một figure lớn với grid 4x2 để chứa 8 biểu đồ
    fig, axs = plt.subplots(4, 2, figsize=(15, 25))
    fig.suptitle("Phân tích Toàn diện: FaceNet (ResnetV1) và ArcFace", fontsize=20)

    # Biểu đồ 1 & 2: Hiệu suất thời gian thực
    if resnetv1_log and arcface_log:
        resnetv1_times, resnetv1_sims = zip(*resnetv1_log)
        arcface_times, arcface_sims = zip(*arcface_log)
        
        axs[0, 0].plot(resnetv1_times, label='ResnetV1 Time', color='green', alpha=0.7)
        axs[0, 0].plot(arcface_times, label='ArcFace Time', color='blue', alpha=0.7)
        axs[0, 0].set_title("Thời gian xử lý")
        axs[0, 0].set_xlabel("Lượt nhận diện")
        axs[0, 0].set_ylabel("Thời gian (giây)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(resnetv1_sims, label='ResnetV1 Similarity', color='green', linestyle='--')
        axs[0, 1].plot(arcface_sims, label='ArcFace Similarity', color='blue', linestyle='--')
        axs[0, 1].axhline(y=0.6, color='r', linestyle=':', label='Ngưỡng (0.6)')
        axs[0, 1].set_title("Độ tương đồng")
        axs[0, 1].set_xlabel("Lượt nhận diện")
        axs[0, 1].set_ylabel("Similarity")
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    else:
        axs[0, 0].text(0.5, 0.5, 'Không có dữ liệu thời gian thực', ha='center', va='center')
        axs[0, 1].text(0.5, 0.5, 'Không có dữ liệu thời gian thực', ha='center', va='center')

    # Biểu đồ 3 & 4: Phân bố và FPS
    if resnetv1_log and arcface_log:
        axs[1, 0].hist(resnetv1_sims, bins=20, alpha=0.7, label='ResnetV1', color='green')
        axs[1, 0].hist(arcface_sims, bins=20, alpha=0.7, label='ArcFace', color='blue')
        axs[1, 0].set_title("Phân bố Similarity")
        axs[1, 0].set_xlabel("Similarity")
        axs[1, 0].set_ylabel("Tần suất")
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        avg_fps_rv1 = 1 / np.mean(resnetv1_times) if len(resnetv1_times) > 0 else 0
        avg_fps_arc = 1 / np.mean(arcface_times) if len(arcface_times) > 0 else 0
        axs[1, 1].bar(['ResnetV1', 'ArcFace'], [avg_fps_rv1, avg_fps_arc], color=['green', 'blue'])
        axs[1, 1].set_title('Tốc độ trung bình (FPS)')
        axs[1, 1].set_ylabel('Frames Per Second (FPS)')
        axs[1, 1].grid(axis='y')
    else:
        axs[1, 0].text(0.5, 0.5, 'Không có dữ liệu thời gian thực', ha='center', va='center')
        axs[1, 1].text(0.5, 0.5, 'Không có dữ liệu thời gian thực', ha='center', va='center')

    # Biểu đồ 5 & 6: So sánh kiến trúc
    df_arch.set_index('Model')['Số lượng tham số (Triệu)'].plot(kind='bar', ax=axs[2, 0], color=['green', 'blue'], alpha=0.8)
    axs[2, 0].set_title('Số lượng tham số')
    axs[2, 0].set_ylabel('Tham số (Triệu)')
    axs[2, 0].tick_params(axis='x', rotation=0)
    axs[2, 0].grid(axis='y')

    df_arch.set_index('Model')['FLOPs (GFLOPs)'].plot(kind='bar', ax=axs[2, 1], color=['green', 'blue'], alpha=0.8)
    axs[2, 1].set_title('Khối lượng tính toán (FLOPs)')
    axs[2, 1].set_ylabel('GFLOPs')
    axs[2, 1].tick_params(axis='x', rotation=0)
    axs[2, 1].grid(axis='y')
    
    # Biểu đồ 7 & 8: So sánh benchmark
    df_benchmark_plot = df_benchmark[df_benchmark['Dataset'] != 'IJB-C (TAR@FAR=1e-4)']
    df_benchmark_plot.plot(x='Dataset', y=['FaceNet (ResnetV1) Accuracy (%)', 'ArcFace (buffalo_l) Accuracy (%)'],
                           kind='bar', ax=axs[3, 0], color=['green', 'blue'])
    axs[3, 0].set_title('Độ chính xác trên các bộ dữ liệu')
    axs[3, 0].set_ylabel('Accuracy (%)')
    axs[3, 0].tick_params(axis='x', rotation=0)
    axs[3, 0].set_ylim(90, 100)
    axs[3, 0].grid(axis='y')

    df_ijbc = df_benchmark[df_benchmark['Dataset'] == 'IJB-C (TAR@FAR=1e-4)']
    df_ijbc.plot(x='Dataset', y=['FaceNet (ResnetV1) Accuracy (%)', 'ArcFace (buffalo_l) Accuracy (%)'],
                 kind='bar', ax=axs[3, 1], color=['green', 'blue'])
    axs[3, 1].set_title('Độ chính xác trên IJB-C')
    axs[3, 1].set_ylabel('Accuracy (%)')
    axs[3, 1].tick_params(axis='x', rotation=0)
    axs[3, 1].set_ylim(85, 100)
    axs[3, 1].grid(axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    chart_filename = "comparison_charts_full.png"
    plt.savefig(chart_filename)
    print(f"\n✅ Đã lưu biểu đồ so sánh TỔNG HỢP tại: {chart_filename}")

