#include "../inc/OpenCV-Face-Recognition.h"

void databaseInitialization() {
	try {
		// ������� ��� ��������� ���� ������ "example.db"
		sqlite::database db("example.db");

		// ������� �������, ���� ��� �� ����������
		db << "CREATE TABLE IF NOT EXISTS attendance ("
			"id INTEGER PRIMARY KEY AUTOINCREMENT, "
			"name TEXT, "
			"entryTime INTEGER, "
			"exitTime INTEGER);";

		// ������ �������������
		addRecord(db, "Alice", 1609459200);  // ��������� ������ ��� Alice � �������� �����
		addExitTimeToRecord(db, "Alice", 1609466400);  // ��������� ����� ������ ��� Alice
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "SQLite error: " << e.what() << std::endl;
	}
}

// ������� ��� ���������� ����� ������ � ������ � �������� �����
void addRecord(sqlite::database & db, const std::string & name, int entryTime) {
	try {
		db << "INSERT INTO attendance (name, entryTime) VALUES (?, ?);"
			<< name << entryTime;
		std::cout << "Record added: " << name << " ����� � " << entryTime << std::endl;
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "������ ��� ���������� ������: " << e.what() << std::endl;
	}
}

// ������� ��� ���������� ������� ������ ��� ��������� ������ �� �����
void addExitTimeToRecord(sqlite::database& db, const std::string& name, int exitTime) {
	try {
		// ��������� ��������� ������ � ���������� ������, � ������� ��� �� ��������� ���� exitTime
		db << "UPDATE attendance SET exitTime = ? WHERE name = ? AND exitTime IS NULL ORDER BY id DESC LIMIT 1;"
			<< exitTime << name;
		std::cout << "Exit time updated for " << name << ": " << exitTime << std::endl;
	}
	catch (sqlite::sqlite_exception& e) {
		std::cerr << "������ ��� ���������� ������� ������: " << e.what() << std::endl;
	}
}
