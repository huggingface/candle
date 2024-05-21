sh1 = 1
sh2 = 1
sh3 = 3
sh4 = 2
sh5 = 4

st1 = 1
st2 = 1
st3 = 4
st4 = 12
st5 = 1


shapes1 = sh5
shapes2 =  (shapes1 * sh4)
shapes3 =  (shapes2 * sh3)
shapes4 =  (shapes3 * sh2)
shapes5 =  (shapes4 * sh1)

for index in range(0,2*3*4):
    s1 = (index // shapes4)
    s2 = (index // shapes3) % (shapes4 // shapes3)
    s3 = (index // shapes2) % (shapes3 // shapes2)
    s4 = (index // shapes1) % (shapes2 // shapes1)
    s5 = index             % (shapes1)

    new_index = s1 * st1 + s2 * st2 + s3 * st3 + s4 * st4 + s5 * st5
    print(f"index:{index}=>{new_index}")


sum = 0
for i in range(0,1000000):
    sum += i

print(sum)