#include "../inc/OpenCV-Face-Recognition.h"

void databaseInitialization() {
	try {
		// Создаем или открываем базу данных "example.db"
		sqlite::database db("example.db");

		// Создаем таблицу, если она не существует
		db << "CREATE TABLE IF NOT EXISTS attendance ("
			"id INTEGER PRIMARY KEY AUTOINCREMENT, "
			"name TEXT, "
			"entryTime INTEGER, "
			"exitTime INTEGER);";

		// Пример использования
		addRecord(db, "Alice", 1609459200);  // Добавляем запись для Alice с временем входа
		addExitTimeToRecord(db, "Alice", 1609466400);  // Обновляем время выхода для Alice
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "SQLite error: " << e.what() << std::endl;
	}
}

// Функция для добавления новой записи с именем и временем входа
void addRecord(sqlite::database & db, const std::string & name, int entryTime) {
	try {
		db << "INSERT INTO attendance (name, entryTime) VALUES (?, ?);"
			<< name << entryTime;
		std::cout << "Record added: " << name << " вошел в " << entryTime << std::endl;
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "Ошибка при добавлении записи: " << e.what() << std::endl;
	}
}

// Функция для обновления времени выхода для последней записи по имени
void addExitTimeToRecord(sqlite::database& db, const std::string& name, int exitTime) {
	try {
		// Обновляем последнюю запись с переданным именем, у которой еще не заполнено поле exitTime
		db << "UPDATE attendance SET exitTime = ? WHERE name = ? AND exitTime IS NULL ORDER BY id DESC LIMIT 1;"
			<< exitTime << name;
		std::cout << "Exit time updated for " << name << ": " << exitTime << std::endl;
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "Ошибка при обновлении времени выхода: " << e.what() << std::endl;
	}
}
