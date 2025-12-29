with open('example.txt', 'w') as f:
    f.write("Python file handling\n")

with open('example.txt', 'w') as f:
    f.write("Easy and Powerful\n")

with open('example.txt', 'r') as f:
    print(f.read())
#create and write to a file
file=open('example.txt', 'w')
file.write("Name : Haritha\n")
file.close()

#append data to a file
file=open('example.txt', 'a')
file.write("Grade : A\n")
file.close()

print("Data Appended.")