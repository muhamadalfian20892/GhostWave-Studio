GHOSTWAVE STUDIO (VERSI 1.7.0)
PANDUAN PENGGUNA DAN PETUNJUK MEMULAI

Situs Web Asli: http://technokerslab.blogspot.com/

Selamat datang di GhostWave Studio! Jika Anda mencari cara yang mudah dan
praktis untuk menyiapkan lagu dan lirik agar lolos dari pemblokiran otomatis di
platform seperti Suno, Anda berada di tempat yang tepat. Program ini saya rancang
khusus agar Anda tidak perlu paham bahasa pemrograman, baris perintah, atau
teori pemrosesan sinyal untuk mendapatkan hasil terbaik.


APA YANG SEBENARNYA DILAKUKAN APLIKASI INI

Saat Anda mengunggah lagu ke layanan musik kecerdasan buatan, sistem mereka akan
menjalankan pemindai otomatis untuk mencocokkan file Anda dengan lagu berhak cipta
yang sudah ada di database. Sistem mendengarkan titik penting audio, lekukan melodi,
dan ciri khas suara. Jika sistem menganggap lagu Anda terlalu mirip dengan lagu
komersial yang sudah beredar, unggahan Anda akan langsung ditolak.

GhostWave Studio mengambil lagu asli Anda dan menerapkan penyesuaian akustik
halus yang mengaburkan sidik jari titik penting tersebut. Di telinga manusia,
lagu Anda tetap terdengar seperti aslinya dengan ketukan dan ritme yang utuh,
tetapi bagi pemindai otomatis, kecocokan sidik jari lagu menjadi hilang.

Semua kunci API pribadi dan pengaturan Anda disimpan secara aman di komputer
lokal dalam berkas profil bernama ghostwave.sn. Tidak ada data yang dikirim
ke server tanpa izin, dan Anda tidak perlu mengunduh model AI berukuran raksasa
hanya untuk menjalankan aplikasi ini.


DUKUNGAN BAHASA DAN PENGGANTIAN BAHASA REALTIME

Mulai versi 1.4.0, GhostWave Studio mendukung penuh dua bahasa: Bahasa Indonesia
dan Bahasa Inggris.

Saat Anda membuka aplikasi ini untuk pertama kalinya, jendela pemilihan bahasa
akan langsung menyambut Anda, sehingga Anda bisa memilih bahasa yang paling
nyaman digunakan. Anda juga dapat mengganti bahasa antarmuka kapan saja melalui
menu Bahasa di menu bar. Seluruh tampilan aplikasi akan langsung berganti saat
itu juga tanpa perlu menutup atau memulai ulang aplikasi.


PUSAT TIKET DUKUNGAN DAN BANTUAN

Jika Anda menemukan masalah, menemukan format file audio yang gagal diproses,
atau memiliki ide fitur baru, Anda bisa langsung mengirimkan tiket bantuan
langsung dari dalam aplikasi tanpa perlu membuka browser.

1. Tekan Ctrl+T pada keyboard atau buka menu Bantuan > Tiket Dukungan.
2. Pilih kategori laporan Anda: Laporan Kerusakan, Permintaan Fitur, Pertanyaan
dan Bantuan, atau Lainnya.
3. Masukkan judul dan deskripsi masalah secara jelas.
4. Biarkan kotak centang diagnosa sistem tetap aktif agar saya selaku pengembang dapat
mengetahui versi Windows dan versi aplikasi untuk memperbaiki kendala dengan cepat.
5. Klik tombol Kirim Tiket.

Tiket Anda akan terkirim dengan aman ke sistem saya. Anda bisa membuka
tab "Tiket Saya" kapan saja untuk membaca balasan dari pengembang dan mengirimkan
pesan balasan lanjutan langsung di jendela tersebut.

Anda juga bisa menghubungi pengembang langsung melalui email di hafiyanajah@gmail.com
untuk pertanyaan atau masukan lainnya.


PANDUAN CEPAT: MEMBERSIHKAN AUDIO DALAM TIGA LANGKAH

Langkah 1: Pilih berkas audio Anda.
Klik tombol Jelajahi atau tekan Alt+B pada keyboard. Pilih berkas lagu Anda
(format WAV, MP3, FLAC, M4A, atau OGG). GhostWave Studio akan memeriksa berkas
tersebut dan menampilkan durasi, frekuensi sampel, serta formatnya langsung di
layar. Anda juga bisa langsung menyeret dan menjatuhkan file lagu dari File Explorer
ke dalam jendela aplikasi!

Langkah 2: Pilih preset sanitasi.
Bagi sebagian besar pengguna, preset bawaan "Complete Sanitization" memberikan
tingkat perlindungan tertinggi. Jika Anda ingin suara lagu tetap sangat mendekati
rekaman asli, Anda bisa memilih preset "Balanced Audio Quality" dari menu pilihan.
Jika Anda tidak ingin pusing dengan angka teknis, abaikan saja opsi lanjutan.
Namun jika Anda ingin mengatur pergeseran nada, penyesuaian tempo, atau simulasi
ruang studio, klik tombol "Tampilkan Pengaturan Lanjutan" untuk membukanya.

Langkah 3: Proses dan simpan berkas Anda.
Klik tombol "Proses Audio" atau tekan Alt+P. Tentukan lokasi untuk
menyimpan berkas baru. Saya sangat menyarankan menyimpan hasil dalam format
16-bit PCM WAV. Bilah progres akan berjalan seiring berjalannya setiap tahapan,
dan setelah selesai, Anda mendapatkan berkas audio bersih yang siap diunggah.
Jika Anda perlu membatalkan di tengah jalan, cukup klik tombol "Batal".


PANDUAN LENGKAP FITUR BARU VERSI 1.7.0 (MODULAR & FLEKSIBEL)

Di versi 1.7.0, GhostWave Studio dibuat jauh lebih ramah untuk siapa saja. Anda tidak
lagi terkurung di mode kaku. Sekarang Anda bisa mengatur sendiri setiap elemen
menggunakan penggeser (slider) dan kotak centang sesuai kebutuhan lagu Anda.

1. Preset Platform Sasaran
Di bagian atas panel Audio, Anda bisa memilih tujuan unggahan Anda:
- Suno v4-v6 Cover Mode: Dioptimalkan khusus untuk melewati pemindai terbaru Suno
  (Audible Magic & pencocokan melodi CQT).
- Google Lyria 3.5: Mengaktifkan pembersih watermark SynthID dan difusi fase.
- Udio AI: Menyesuaikan pergeseran tempo dan modulasi harmonik untuk arsitektur Udio.
- YouTube Content ID: Berfokus pada pergeseran puncak frekuensi dan akustik ruangan.
- TikTok & Reels: Penyesuaian cepat untuk video pendek.
- Custom (Pengaturan Mandiri): Memberi kebebasan penuh bagi Anda untuk mengatur semua slider.

2. Penggeser (Slider) Modular yang Mudah Dipahami
- Pergeseran Nada (Pitch Shift): Menggeser tinggi rendah nada lagu dalam satuan
  semitone (misalnya +2 atau -2). Menggeser 2 nada membuat lagu tetap terdengar
  merdu dan alami, tetapi membuat algoritma pencocokan melodi langsung gagal mengenali lagu.
- Karakter Vokal (Formant Shift): Mengubah warna suara vokal (menjadikannya sedikit
  lebih tebal atau lebih ramping) tanpa mengubah nada lagu sama sekali. Ini sangat
  efektif mengelabui pemindai vokal AI.
- Tempo Lagu (Speed / BPM): Memperlambat atau mempercepat lagu sedikit saja
  (misalnya 98% atau 103%). Perubahan sekecil ini hampir tidak terasa di telinga,
  tetapi memutus sinkronisasi garis waktu pemindai otomatis.
- Gema Ruang Studio (Reverb): Memberikan sedikit pantulan akustik alami seolah-olah
  lagu direkam ulang di dalam studio fisik.
- Redam Vokal Tengah (Vocal Ducking): Menurunkan volume vokal utama di bagian tengah
  stereo sehingga instrumen pengiring lebih menonjol.

3. Kotak Centang Modular
- Bersihkan Watermark AI (SynthID): Jika audio Anda berasal atau melewati model AI
  (seperti Google Lyria atau pembuat vokal AI), centang opsi ini untuk menetralkan
  jejak frekuensi tak terdengar yang ditanam oleh sistem watermark.
- Topeng Psikoakustik (Masking): Menyisipkan titik sinyal pengalih di frekuensi yang
  secara alami tidak bisa didengar oleh telinga manusia (berdasarkan ambang pendengaran
  Terhardt ATH). Pemindai akan kebingungan membaca pola lagu, sementara kualitas audio
  tetap jernih di telinga Anda.
- Pelembut Artikulasi (Whisper Scrambler): Mengaburkan desah konsonan vokal agar
  pemindai lirik otomatis (seperti OpenAI Whisper) kesulitan mentranskripsi kata.
- Potong Vokal Tengah (Vocal Cut): Menghilangkan vokal tengah secara langsung.

4. Deteksi Tangga Nada (Key) dan Ketukan (BPM) Otomatis
Begitu Anda memilih berkas lagu, GhostWave Studio akan langsung mendeteksi tangga nada
dan temponya, lalu menampilkannya di atas panel (misalnya: "Tangga Nada: G Major | Tempo: 124 BPM").
Di sebelahnya, ada tombol "[Gunakan Rekomendasi Nada]". Jika diklik, aplikasi akan
otomatis menggeser slider nada ke +2 atau -2 semitone yang paling selaras secara musikal.

5. Visualizer Gelombang Ganda & Titik Pengecoh
Di bawah informasi file, terdapat visualizer yang menampilkan:
- Garis Biru: Gelombang audio asli Anda.
- Garis Hijau Zamrud: Gelombang audio yang sudah diproses.
- Titik Emas: Lokasi titik pengecoh (decoy peaks) yang disisipkan untuk membuyarkan sidik jari.
Tampilan ini juga dilengkapi deskripsi teks yang bisa dibaca oleh pembaca layar.

6. Pemutar Uji Dengar Langsung (A/B & Delta Player)
Setelah audio diproses, Anda tidak perlu repot membuka pemutar musik eksternal. Gunakan
tombol pemutar di panel:
- [Putar Asli]: Mendengarkan suara sebelum diproses.
- [Putar Hasil]: Mendengarkan suara sesudah diproses.
- [Putar Delta]: Mendengarkan HANYA perbedaan atau efek yang ditambahkan oleh GhostWave Studio.
- [Stop]: Menghentikan pemutaran seketika.

7. Simulator Keamanan Sebelum Unggah (Pre-Flight Simulator - Alt+F)
Sebelum Anda membuang waktu mengunggah file ke platform, buka simulator ini lewat tombol
"Uji Keamanan Sebelum Unggah" atau tekan Alt+F. Simulator akan memeriksa file Anda dari
empat sisi:
- Kesamaan Melodi (CQT Chroma)
- Kesamaan Sidik Jari Titik Puncak (Acoustic Landmarks)
- Keberadaan Jejak AI (Watermark SynthID / AudioSeal)
- Pergeseran Harmoni
Anda akan mendapatkan Skor Keamanan (0-100%) dan saran perbaikan langsung yang ditulis
dalam bahasa santai dan mudah dipahami.

8. Antrean Berkas Banyak (Batch) & Pemantau Folder (Folder Watcher - Ctrl+Shift+B)
Punya banyak lagu yang ingin dibersihkan sekaligus? Tekan Ctrl+Shift+B:
- Tab Antrean Pemrosesan: Tambahkan puluhan lagu sekaligus, centang apakah ingin
  langsung dipotong per 22 detik, lalu klik "Mulai Pemrosesan Antrean".
- Tab Pemantau Folder: Tentukan folder masuk dan folder keluar. Aplikasi akan terus
  memantau folder tersebut di latar belakang. Setiap kali Anda memasukkan lagu baru ke
  folder masuk, aplikasi akan otomatis memprosesnya dan menyimpannya di folder keluar.

9. Pembuat Prompt Gaya Musik & Struktur Lagu Suno / Udio
Di tab Sanitasi Lirik (Ctrl+Tab), ada tombol baru: "[Buat Prompt Gaya Suno / Udio...]".
Fitur ini membantu Anda merangkai deskripsi gaya lagu (misalnya: "Indie Pop, 120 BPM, G Major, Upbeat")
dan menyediakan tombol cepat untuk menyisipkan tag struktur lagu seperti:
[Intro], [Verse 1], [Chorus - Catchy Melodic Hook], [Drop], [Bridge], dan [Outro].
Hasilnya bisa langsung disalin atau disisipkan ke kotak lirik Anda.

10. Penjaga Rima pada Sanitasi Lirik
Di tab Sanitasi Lirik, terdapat kotak centang baru: "Pertahankan rima akhir kalimat".
Saat aktif, sanitasi lirik tidak akan mengubah kata-kata di ujung baris yang bersajak,
sehingga irama lagu tetap enak dinyanyikan oleh model AI.


RAHASIA MENGUNGGAH KE SUNO AGAR TIDAK DIBLOKIR

Dari pengujian intensif selama berbulan-bulan, saya menemukan bahwa cara Anda
mengunggah berkas ke Suno sama pentingnya dengan isi audionya. Ingat selalu
empat aturan emas berikut:

1. Selalu simpan sebagai WAV 16-bit murni, jangan MP3.
Berkas MP3 memiliki header dan metadata penyandi yang sangat mudah dilacak oleh
pemindai sidik jari otomatis. Berkas WAV tidak memiliki metadata tambahan yang
membuat pemindai jauh lebih sulit mengenali pola aslinya.

2. Jangan mengunggah langsung di halaman Create atau Studio.
Halaman Create Suno menjalankan pemindaian serentak yang sangat ketat dan sering
langsung menolak berkas. Sebagai gantinya, buka halaman Library di akun Suno Anda,
lalu klik tombol "Upload Audio" di sana. Antrean unggahan Library diproses oleh
sistem latar belakang yang aturannya jauh lebih longgar. Setelah klip lagu Anda
muncul di Library, klik ikon titik tiga pada lagu tersebut lalu pilih "Extend"
atau "Create with Audio".

3. Buat potongan klip audio antara 20 sampai 24 detik.
Lagu berdurasi penuh memberikan ratusan titik pencocokan berurutan bagi pemindai.
Klip pendek memutus garis waktu tersebut. Jika lagu Anda panjang, gunakan fitur
"ABS Audio Slicer" di GhostWave Studio (atau tekan Alt+S di tab Audio) untuk memotong
lagu menjadi bagian 22 detik secara otomatis. Unggah bagian pertama ke Library Suno,
dan Anda siap melanjutkan kreasi.

4. Jika Suno menolak unggahan, bersihkan cookie atau tunggu sebentar.
Jika Suno menampilkan pesan bahwa audio Anda cocok dengan rekaman yang sudah ada,
jangan langsung mencoba mengunggah ulang berkali-kali. Sesi browser Anda akan
diberi tanda peringatan sementara selama tiga puluh hingga enam puluh menit. Buka
jendela penyamaran pribadi (incognito), hapus cache browser, atau istirahat
sejenak sebelum mencoba lagi dengan berkas yang sudah diproses.


MEMBERSIHKAN DAN MEMFORMAT LIRIK LAGU

Jika Anda mengetik atau menempelkan lirik ke Suno dan ditolak karena memuat
nama musisi terkenal, kata bermerek, atau filter hak cipta, pindah ke tab
Sanitasi Lirik dengan menekan Ctrl+Tab.

1. Tempelkan lirik asli Anda ke kotak bagian atas.
2. Pilih strategi sanitasi. Opsi "Phonetic Spelling Variation" adalah yang
paling populer karena mengganti frasa sensitif dengan ejaan fonetik yang ramah
vokal dan pemenggalan suku kata tanpa mengubah cara vokalis menyanyikan liriknya.
3. Klik tombol "Sanitasi Lirik" atau tekan Alt+S.
4. Klik tombol "Salin ke Papan Klip" atau tekan Alt+C, lalu tempelkan hasilnya
langsung ke kotak prompt Suno.


PINTASAN KEYBOARD UNTUK SCREEN READER

GhostWave Studio dirancang dari awal agar ramah bagi pembaca layar (screen reader),
termasuk NVDA, JAWS, dan Windows Narrator. Setiap tombol, penggeser, dan kotak status
memiliki label suara yang jelas dan tombol pintasan.

Tekan F1 kapan saja untuk membuka dialog Panduan Aksesibilitas. Anda bisa membaca
isi panduan baris demi baris menggunakan tombol panah, menekan Escape untuk
menutupnya, atau menekan tombol Salin untuk menyimpan seluruh daftar pintasan ke
papan klip Anda.

Pintasan utama yang perlu diingat:
Tab dan Shift+Tab: Pindah maju dan mundur antar tombol atau input.
Alt+1 / Alt+2 / Ctrl+Tab: Beralih antara tab Audio Sanitizer dan Lyrics Sanitizer.
Alt+B atau Ctrl+O: Memilih berkas audio masukan.
Alt+P atau Ctrl+P: Memproses dan mengekspor audio.
Alt+F: Membuka Simulator Keamanan Sebelum Unggah (Pre-Flight Evasion Simulator).
Alt+E atau Ctrl+E: Membuka laporan Audit Kemiripan Audio.
Alt+S: Di tab Audio untuk membuka ABS Audio Slicer. Di tab Lirik untuk memproses lirik.
Alt+A atau Ctrl+K: Membuka pengaturan Cloud API.
Ctrl+T: Membuka Pusat Tiket Dukungan.
Ctrl+U: Membuka pemotong ABS Audio Slicer dari tab mana saja.
Ctrl+Shift+B: Membuka Antrean Pemrosesan Batch dan Pemantau Folder Otomatis.
Ctrl+B: Membuka editor daftar hitam musisi dan selebritas.
Ctrl+H atau Shift+F1: Membuka panduan unggah Suno.
Ctrl+Shift+S: Mengekspor profil pengaturan (.sn).
Ctrl+Shift+O: Mengimpor profil pengaturan (.sn).
Ctrl+A: Memilih seluruh teks di dalam kotak teks.
F1: Membuka Panduan Aksesibilitas dan daftar pintasan.
F2: Membuka jendela Catatan Rilis / Changelog terbaru (tersedia offline dan online).
Escape: Menutup jendela dialog yang sedang aktif.


PEMBARUAN APLIKASI OTOMATIS

Anda tidak perlu memeriksa situs web secara manual untuk mengetahui apakah ada
versi baru yang dirilis. Setiap kali GhostWave Studio dibuka, aplikasi dapat
memeriksa pembaruan secara mandiri di latar belakang. Jika ada pembaruan yang
tersedia, jendela ramah akan muncul menanyakan apakah Anda ingin mengunduhnya.

Anda bisa mengklik "Lihat Yang Baru" untuk membaca catatan perubahan di jendela
yang sama, atau mengklik "Unduh Sekarang" agar aplikasi mengunduh berkas installer
baru dan memandu proses pembaruan secara praktis.


OPEN SOURCE DAN KONTRIBUSI KOMUNITAS

GhostWave Studio adalah perangkat lunak open source seutuhnya. Jika Anda ingin
melihat kode sumber, melaporkan kendala teknis, memberi masukan perbaikan, atau
berkontribusi kode, silakan kunjungi repository GitHub resmi saya:
https://github.com/muhamadalfian20892/GhostWave-Studio
