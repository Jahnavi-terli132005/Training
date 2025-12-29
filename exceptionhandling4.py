try:
    f=open('non_existent_file.txt', 'r')
    print(f.read())
except FileNotFoundError:
    print("Error: The file does not exist.")   
finally:
    print("Execution completed.")
    try:
        a=int(input())
        print(10/a)
    except (ValueError, ZeroDivisionError):
        print("Error: Invalid input or division by zero.")