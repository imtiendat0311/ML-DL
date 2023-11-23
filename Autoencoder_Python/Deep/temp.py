def bai1():
    a = int(input("Nhap n: "))
    for i in range(1, a+1):
        print(i if i % 2 == 0 else " ", end="")


def bai2():
    a = int(input("Nhap n: "))
    b = 0
    c = 0
    for i in range(a):
        if i == 0:
            print(0, end=" ")
            c = 0
        if i == 1:
            print(1, end=" ")
            b = 1
        if i > 1:
            prev_B = b
            b = b + c
            c = prev_B
            print(b, end=" ")


def bai2_cach2():
    a = int(input("Nhap n:"))
    fibonachi(a)


def fibonachi(n, a=0, b=1):
    if n == 0:
        return
    else:
        if (a != None):
            print(a, end=" ")
        fibonachi(n-1, b, a+b)


def bai2_cach3():
    a = int(input("Nhap n:"))
    fib = [0, 1]+[0]*(a-1)
    for i in range(2, a):
        fib[i] = fib[i-1]+fib[i-2]
    print(fib[:a])


# bai1()
# bai2()
bai2_cach3()
