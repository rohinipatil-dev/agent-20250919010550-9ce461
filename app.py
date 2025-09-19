import os
from typing import List, Dict, Optional
from datetime import date

import streamlit as st
from openai import OpenAI


# ----------------------------
# OpenAI Client
# ----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI()


# ----------------------------
# System Prompt
# ----------------------------
def build_system_prompt(language: str = "id") -> str:
    if language == "en":
        return (
            "You are an AI advisor assisting a high school principal in Indonesia. "
            "Goals: provide practical, concise, and actionable guidance on school management, curriculum planning, teacher development, "
            "student affairs, school operations, parent/community communications, and event planning. "
            "Constraints: do not fabricate facts or legal references; when unsure, say so and suggest how to verify. "
            "Default response is concise with bullet points and step-by-step actions. "
            "Provide templates (letters, schedules, memos) with placeholders and clear instructions. "
            "If the user asks, you can switch to Indonesian or bilingual output. "
            "Tone: respectful, professional, supportive, and solution-oriented."
        )
    else:
        return (
            "Anda adalah asisten AI yang membantu kepala sekolah menengah atas di Indonesia. "
            "Tujuan: berikan panduan yang praktis, ringkas, dan dapat ditindaklanjuti terkait manajemen sekolah, perencanaan kurikulum, "
            "pengembangan guru, kesiswaan, operasional, komunikasi orang tua/komite, dan perencanaan kegiatan. "
            "Batasan: jangan mengada-ada fakta atau dasar hukum; jika ragu, sampaikan dengan jujur dan sarankan cara memverifikasi. "
            "Default jawaban: ringkas, dengan poin-poin dan langkah-langkah. "
            "Sediakan template (surat, jadwal, memo) dengan placeholder dan instruksi yang jelas. "
            "Jika diminta, Anda dapat menggunakan bahasa Inggris atau keluaran dwibahasa. "
            "Nada: hormat, profesional, suportif, dan berorientasi solusi."
        )


# ----------------------------
# Session State Helpers
# ----------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "language" not in st.session_state:
        st.session_state.language = "id"
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 800
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def clear_conversation():
    st.session_state.messages = []


# ----------------------------
# OpenAI Chat Interaction
# ----------------------------
def ask_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    # Compose messages: system + history + latest user input
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ----------------------------
# UI Components
# ----------------------------
def sidebar_controls():
    st.sidebar.header("Pengaturan")
    lang = st.sidebar.selectbox(
        "Bahasa / Language",
        options=[("id", "Bahasa Indonesia"), ("en", "English")],
        index=0,
        format_func=lambda x: x[1],
        key="language_selector",
    )
    st.session_state.language = lang[0]

    st.session_state.model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Pilih model bahasa. gpt-4 menghasilkan kualitas yang lebih baik."
    )

    st.session_state.temperature = st.sidebar.slider(
        "Kreativitas (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    st.session_state.max_tokens = st.sidebar.slider(
        "Batas Jawaban (max_tokens)",
        min_value=256,
        max_value=2000,
        value=800,
        step=32,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Mulai Percakapan Baru"):
        clear_conversation()
        st.sidebar.success("Percakapan direset.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Pastikan variabel lingkungan OPENAI_API_KEY telah disetel.")


def template_builder():
    st.subheader("Template Cepat")
    tab1, tab2, tab3 = st.tabs(["Surat Edaran", "RKS/RKAS", "Jadwal & Tugas"])

    with tab1:
        with st.form("form_surat_edaran"):
            topik = st.text_input("Topik Surat", value="Informasi Ujian Akhir Semester")
            audiens = st.text_input("Audiens", value="Orang Tua/Wali Siswa")
            tanggal = st.date_input("Tanggal", value=date.today())
            penandatangan = st.text_input("Penandatangan", value="Kepala Sekolah")
            gaya = st.selectbox("Gaya Bahasa", ["Formal", "Semi-formal"])
            bilingual = st.checkbox("Butuh versi bilingual (ID-EN)?", value=False)
            submitted = st.form_submit_button("Susun Surat dengan AI")
            if submitted:
                prompt = build_prompt_surat_edaran(
                    topik, audiens, str(tanggal), penandatangan, gaya, bilingual
                )
                st.session_state.pending_prompt = prompt
                st.success("Prompt untuk surat edaran telah disiapkan. Lihat kolom chat di bawah.")

    with tab2:
        with st.form("form_rks"):
            periode = st.text_input("Periode", value="Juli-Desember 2025")
            fokus = st.text_area(
                "Fokus Program (pisahkan dengan baris)",
                value="Peningkatan Literasi\nPenguatan Projek P5\nPengembangan Kompetensi Guru\nPerbaikan Sarana/Prasarana",
            )
            indikator = st.text_area(
                "Indikator Keberhasilan (opsional)",
                value="Nilai literasi meningkat 10%\nMinimal 2 pelatihan guru per semester\nKetersediaan perpustakaan digital",
            )
            submitted = st.form_submit_button("Susun RKS/RKAS dengan AI")
            if submitted:
                prompt = build_prompt_rks(periode, fokus, indikator)
                st.session_state.pending_prompt = prompt
                st.success("Prompt untuk RKS/RKAS telah disiapkan. Lihat kolom chat di bawah.")

    with tab3:
        with st.form("form_jadwal"):
            jenis = st.selectbox("Jenis Jadwal/Tugas", ["Jadwal Mengajar", "Jadwal Piket", "Pembagian Tugas Guru"])
            kondisi = st.text_area(
                "Kondisi/Kebijakan (opsional)",
                value="Prioritaskan pemerataan beban mengajar.\nPerhatikan sertifikasi dan keahlian guru.\nHindari bentrok dengan jadwal wali kelas.",
            )
            submitted = st.form_submit_button("Susun dengan AI")
            if submitted:
                prompt = build_prompt_jadwal(jenis, kondisi)
                st.session_state.pending_prompt = prompt
                st.success("Prompt untuk jadwal/tugas telah disiapkan. Lihat kolom chat di bawah.")


def build_prompt_surat_edaran(
    topik: str,
    audiens: str,
    tanggal: str,
    penandatangan: str,
    gaya: str,
    bilingual: bool,
) -> str:
    lang_note = "Tulis bilingual (Indonesia dan Inggris) dengan dua bagian terpisah dan konsisten." if bilingual else "Gunakan Bahasa Indonesia yang jelas dan formal."
    return (
        f"Susun draf Surat Edaran sekolah tentang '{topik}'.\n"
        f"- Audiens: {audiens}\n"
        f"- Tanggal: {tanggal}\n"
        f"- Penandatangan: {penandatangan}\n"
        f"- Gaya bahasa: {gaya}\n"
        f"- {lang_note}\n"
        f"Strukturkan: Kop/Identitas, Nomor, Perihal, Salam pembuka, Isi utama (tujuan, rincian, waktu/tempat jika ada), "
        f"Instruksi/harapan, Penutup, Tanda tangan, Tembusan (jika perlu). "
        f"Sertakan placeholder yang jelas untuk nomor surat, lampiran, dan kontak."
    )


def build_prompt_rks(periode: str, fokus: str, indikator: str) -> str:
    return (
        "Buat RKS/RKAS ringkas untuk tingkat SMA dengan format tabel/poin yang mudah dipahami.\n"
        f"- Periode: {periode}\n"
        f"- Fokus Program:\n{fokus}\n"
        f"- Indikator Keberhasilan (opsional):\n{indikator}\n"
        "Cantumkan: tujuan, kegiatan utama, PIC, timeline, kebutuhan sumber daya/anggaran (perkiraan), indikator & cara evaluasi, risiko & mitigasi. "
        "Gunakan placeholder untuk angka anggaran dan sesuaikan dengan regulasi sekolah."
    )


def build_prompt_jadwal(jenis: str, kondisi: str) -> str:
    return (
        f"Susun {jenis.lower()} untuk SMA. "
        "Berikan langkah penyusunan, asumsi, dan keluaran akhir berupa tabel/poin yang rapi. "
        f"Pertimbangkan ketentuan berikut (jika relevan):\n{kondisi}\n"
        "Sertakan catatan tentang cara menyesuaikan jika terjadi konflik jadwal."
    )


def render_chat(client: Optional[OpenAI]):
    st.subheader("Asisten AI Kepala Sekolah")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # If a template prepared a prompt, show it as a prefill hint
    if st.session_state.pending_prompt:
        with st.expander("Prompt siap digunakan dari Template", expanded=True):
            st.code(st.session_state.pending_prompt)
        prefill = st.session_state.pending_prompt
    else:
        prefill = ""

    user_input = st.chat_input(
        "Ketik pertanyaan atau perintah Anda di sini...",
    )

    # If user hasn't typed but we have pending prompt, auto-fill on button
    if prefill and not user_input:
        if st.button("Gunakan Prompt di atas"):
            user_input = prefill
            st.session_state.pending_prompt = ""

    if user_input:
        add_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)

        # Generate assistant response
        try:
            if client is None:
                raise RuntimeError("OpenAI client is not initialized. Set OPENAI_API_KEY.")

            system_prompt = build_system_prompt(st.session_state.language)
            reply = ask_openai(
                client=client,
                model=str(st.session_state.model),
                system_prompt=system_prompt,
                history=st.session_state.messages[:-1],  # exclude the just-added user message? No—include full history except system
                user_input=user_input,
                temperature=float(st.session_state.temperature),
                max_tokens=int(st.session_state.max_tokens),
            )
            add_message("assistant", reply)
            with st.chat_message("assistant"):
                st.write(reply)
        except Exception as e:
            error_text = f"Terjadi kesalahan saat memanggil model: {e}"
            add_message("assistant", error_text)
            with st.chat_message("assistant"):
                st.error(error_text)

    # Utilities: download transcript
    if st.session_state.messages:
        transcript = []
        for m in st.session_state.messages:
            role = "USER" if m["role"] == "user" else "ASSISTANT"
            transcript.append(f"{role}:\n{m['content']}\n")
        st.download_button(
            "Unduh Transkrip (.txt)",
            data="\n".join(transcript),
            file_name="transkrip_asisten_kepsek.txt",
            mime="text/plain",
        )


# ----------------------------
# Main App
# ----------------------------
def main():
    st.set_page_config(page_title="Asisten AI Kepala Sekolah (Indonesia)", page_icon="🎓", layout="wide")
    init_session_state()

    st.title("🎓 Asisten AI Kepala Sekolah")
    st.caption(
        "Didesain untuk membantu kepala SMA di Indonesia: manajemen sekolah, pengembangan guru, kesiswaan, dan komunikasi. "
        "Gunakan dengan kebijakan internal dan regulasi yang berlaku."
    )

    sidebar_controls()
    template_builder()

    api_key_exists = bool(os.environ.get("OPENAI_API_KEY"))
    if not api_key_exists:
        st.warning(
            "OPENAI_API_KEY belum disetel pada environment. "
            "Set sebelum menggunakan Asisten AI."
        )
        client = None
    else:
        client = get_openai_client()

    render_chat(client)


if __name__ == "__main__":
    main()