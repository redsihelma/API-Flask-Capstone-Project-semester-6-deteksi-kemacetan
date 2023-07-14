CREATE DATABASE IF NOT EXISTS deteksi_kemacetan_kendaraan;

USE deteksi_kemacetan_kendaraan;

CREATE TABLE IF NOT EXISTS User (
    id INT(11) AUTO_INCREMENT PRIMARY KEY,
    firstname VARCHAR(35) NOT NULL,
    lastname VARCHAR(35) NOT NULL,
    email VARCHAR(65) UNIQUE NOT NULL,
    password VARCHAR(123) NOT NULL,
    is_verified TINYINT(1) NOT NULL,
    createdAt DATE,
    updatedAt DATE
);
