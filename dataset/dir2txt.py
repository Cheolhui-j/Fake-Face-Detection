import os


root = './data/pggan_train'

class_dir = os.listdir(root)

fake = os.listdir(root + '/' + class_dir[0])

real = os.listdir(root + '/' + class_dir[1])


with open('./pggan_train.txt', 'w') as f:
    for _ in range(20):
        for i in range(500):
            f.write('{} {}\n'.format(fake[i], class_dir[0]))
        for j in range(500):
            f.write('{} {}\n'.format(real[j], class_dir[1]))