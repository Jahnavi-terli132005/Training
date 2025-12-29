try:
    username = input("Enter username: ")
    password = input("Enter password: ")
    if password != "admin123":
        raise ValueError("Incorrect password")  
    print("Login successful")
except ValueError as e:
    print(e)