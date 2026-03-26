import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array


page_bg = """
<style>
/* الخلفية */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000814, #001830);
    color: white;
    font-family: "Cairo", sans-serif;
}

/* الهيدر */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* الأدوات */
[data-testid="stToolbar"] {
    right: 2rem;
}

/* أداة رفع الملفات (البوكس الخارجي) */
[data-testid="stFileUploader"] {
    background-color: #000000 !important;
    color: white !important;
    border: 2px solid #ff6600 !important;  /* برتقالي */
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0px 0px 12px rgba(255,102,0,0.4);
}

/* نص داخل أداة رفع الملفات */
[data-testid="stFileUploader"] section {
    color: white !important;
}

/* زرار رفع الملف */
[data-testid="stFileUploader"] button {
    background-color: #000000 !important;
    color: white !important;
    border: 2px solid #ff6600 !important;  /* برتقالي */
    border-radius: 8px;
    font-weight: bold;
    box-shadow: 0px 0px 8px rgba(255,102,0,0.5);
    transition: all 0.3s ease-in-out;
}

/* تأثير عند المرور على الزرار */
[data-testid="stFileUploader"] button:hover {
    background-color: #111111 !important;
    box-shadow: 0px 0px 16px rgba(255,102,0,0.8);
}

/* النص اللي فوق زرار رفع الملف */
label[data-testid="stFileUploaderLabel"] {
    color: white !important;
    font-weight: bold;
    font-size: 16px;
}

/* الصندوق الخاص بالنتيجة */
.result-box {
    background-color: #000000;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    border: 2px solid #ff6600;  /* برتقالي */
    margin-top: 1rem;
    box-shadow: 0px 0px 12px rgba(255,102,0,0.4);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# تحميل الموديل
# ==============================
MODEL_PATH = r"C:\Users\ff\Desktop\project ML\BrainTumor\brain_tumor_model.keras"
model = load_model(MODEL_PATH)

# ==============================
# إعداد أسماء الكلاسات
# ==============================
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ==============================
# دالة تجهيز الصورة
# ==============================
def preprocess_image(uploaded_file, target_size=(150, 150)):
    image = load_img(uploaded_file, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ==============================
# واجهة Streamlit
# ==============================
st.title("🧠 Brain Tumor Classification")
st.markdown("### ارفع صورة أشعة MRI وسيقوم النموذج بالتنبؤ بنوع الورم.", unsafe_allow_html=True)

uploaded_file = st.file_uploader("ارفع صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    st.image(uploaded_file, caption="📷 الصورة التي تم رفعها", use_container_width=True)

    # تجهيز الصورة للتنبؤ
    image = preprocess_image(uploaded_file)

    # التنبؤ
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # عرض النتيجة بشكل منسق
    st.subheader("🔎 النتيجة:")
    st.markdown(
        f"""
        <div class="result-box">
            <b>الورم المتوقع:</b> {class_names[predicted_class]} <br>
            <b>نسبة الثقة:</b> {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )
