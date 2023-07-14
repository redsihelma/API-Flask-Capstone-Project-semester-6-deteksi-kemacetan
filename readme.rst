###################
API Deteksi Kemacetan Kendaraan di Jalan Raya
###################

API Deteksi Kemacetan di buat dengan flask versi 2.3.2

##########################################
aplikasi ini memiliki fitur/endpoint sebagai berikut:
##########################################

1. upload video: di endpoint ini user dapat upload video dengan format 
   .mp4 dan .avi, setelah user upload video maka klik tombol execute
   untuk memprediksi video tersebut dengan model mobilenetv1. setelah 
   sistem telah selesai mempredict sistem akan memunculkan tombol 
   download agar user bisa mendownload video hasil deteksi 

2. predict video realtime: di endpoint ini memiliki 3 define/definisi
   yaitu def post, def get, dan def delete. pada define post user klik
   tombol execute untuk mengaktifkan kamera internal pada masing masing devic.
   untuk define get user klik tombol execute untuk memulai predict 
   dari kamera internal  masing masing device. untuk define delete
   user klik tombol execute untuk mengakhiri semua yang berjalan di 
   define post dan get dan sistem akan memunculkan tombol download seperti
   endpoint upload video user download video hasil predict dengan real time
   dan format namanya yaitu output.mp4
   
3. register: di endpoint ini user memasukkan nama depan, nama belakang, 
   email, password, dan konfirmasi password untuk di masukkan ke 
   dalam database MYSQL dan di endpoint ini mengirimkan otp/Verifikasi
   email yang sudah di daftarkan sebelumnya dan otp tersebut 
   akan di masukkan ke dalam endopint verifikasi 

4. Verifikasi: di endpoint ini user di haruskan untuk memasukkan otp
   yang sudah di berikan di email yang sudah di daftarkan sebelumnya
   untuk melanjutkan endpoint login jika user tidak memasukkan otp
   terlebih dahulu maka user tidak akan bisa masuk/login

5. login: di endpoint ini user harus memasukkan email dan password 
   untuk login dan jika user telah memasukkan email dan password 
   dengan benar maka user akan mendapatkan bearer/Token Auth.
   token ini untuk mengubah username/nama pertama dan nama terakhir, 
   dan mengubah password

6. edit password: di endopint ini user dapat mengubah passwordnya
   dengan memasukkan current password/password lama, 
   new password/passwrod baru yang ingin di ubah, dan 
   autorization atau token yang sudah di buat pada saat melakukan login

7. edit user: di endpoint ini sama seperti edit password, di endpoint
   edit user dapat mengubah nama depan, dan nama belakang selain itu 
   user harus menambahkan token/autorization yang sudah di berikan pada
   saat user login

##############################
cara menjalankan aplikasi ini:
##############################

1.  di aplikasi ini menggunakan Flask versi 2.3.2 bisa mengintall 
    atau menggunakan Python versi 3.7 atau yang lebih baru, dalam 
    aplikasi ini menggunakan Python versi 3.9.12 
   
2. di aplikasi ini menggunakan MYSQL untuk databasenya, untuk masuk 
   ke database MYSQL kita membutuhkan XAMPP, di aplikasi ini 
   menggunakan XAMPP versi 3.3.0 unduh XAMPP versi 3.3.0 dan pasang 
   di path C:\ setelah di pasang tempatkan aplikasinya di C:\xampp\htdocs

3. buka aplikasi XAMPP dan klik tombol start apache dan MYSQL jika 
   apache dan MYSQL sudah berwarna hijau klik tombol admin di baris 
   MYSQL untuk masuk ke localhost phpMyAdmin. jika sudah di phpMyAdmin 
   buat database baru dengan nama db_absensi selanjutnya import 
   deteksi_kemacetan_kendaraan.sql 

4. membuat environment untuk menyimpan/menginstall modul modul yang 
   di butuhkan di aplikasi sebelumnya ubah pathnya di penyimpanan 
   aplikasi yang tersimpan untuk menginstall modul modul yang di butuhkan
   untuk installnya menggunakan pipenv, sebelum install modul di pipenv 
   install terlebih dahulu pipenvnya dengan mengetik pip install pipenv
   di CMD, setelah install pipenv selanjutnya install modul 
   dengan cara ketik pipenv install <nama-modul>

5. untuk run/menjalankan aplikasinya menggunakan CMD dengan mengubah
   pathnya di penyimpanan aplikasi yang tersimpan dan mengaktifkan
   environment yang telah dibuat sebelumnya, selanjutnya ketik 
   pipenv run python app.py jika berhasil akan menampilkan link localhost 
   dan link host IP klik salah satu untuk masuk ke halman API, beda 
   lagi jika aplikasi sudah di hosting/di onlinekan