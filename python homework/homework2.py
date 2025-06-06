import json

class DormitoryManagement:
    def __init__(self):
        self.students = {}
        self.load_data()
    
    def load_data(self):
        try:
            with open("students.json", "r", encoding="utf-8") as file:
                self.students = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.students = {}
    
    def save_data(self):
        with open("students.json", "w", encoding="utf-8") as file:
            json.dump(self.students, file, ensure_ascii=False, indent=4)
    
    def add_student(self, student_id, name, gender, room, phone):
        if student_id in self.students:
            print("学号已存在，无法重复添加！")
            return
        self.students[student_id] = {
            "姓名": name,
            "性别": gender,
            "宿舍房间号": room,
            "联系电话": phone
        }
        self.save_data()
        print("学生信息添加成功！")
    
    def find_student(self, student_id):
        if student_id in self.students:
            print("学生信息如下：")
            for key, value in self.students[student_id].items():
                print(f"{key}: {value}")
        else:
            print("未找到该学号的学生信息！")
    
    def show_all_students(self):
        if not self.students:
            print("当前没有学生信息！")
            return
        print("所有学生信息如下：")
        for student_id, info in self.students.items():
            print(f" 学号: {student_id}")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print("------------------")
    
    def run(self):
        while True:
            print("\n宿舍管理系统")
            print("1. 添加学生信息")
            print("2. 查找学生信息")
            print("3. 显示所有学生信息")
            print("4. 退出系统")
            choice = input("请选择操作（1-4）：")
            
            if choice == "1":
                student_id = input("请输入学号：")
                name = input("请输入姓名：")
                gender = input("请输入性别（男/女）：")
                room = input("请输入宿舍房间号：")
                phone = input("请输入联系电话：")
                self.add_student(student_id, name, gender, room, phone)
            elif choice == "2":
                student_id = input("请输入要查找的学号：")
                self.find_student(student_id)
            elif choice == "3":
                self.show_all_students()
            elif choice == "4":
                print("确认退出？(Y/N)")
                confirm = input().strip().lower()
                if confirm == "y":
                    print("退出系统...")
                    break
            else:
                print("无效的选择，请重新输入！")

if __name__ == "__main__":
    system = DormitoryManagement()
    system.run()
